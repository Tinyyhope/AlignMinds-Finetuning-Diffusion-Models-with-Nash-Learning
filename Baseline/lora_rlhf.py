import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import wandb
from tqdm import tqdm
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from diffusers.models.attention_processor import AttnProcessor
from accelerate import Accelerator
from typing import Dict, List, Tuple, Optional
from diffusers.loaders import AttnProcsLayers
# Environment Variables
os.environ["WANDB_API_KEY"] = "wandb_api here"
SCRATCH_DIR = "Your data Path"
cache_dirs = {
    "HF_HOME": os.path.join(SCRATCH_DIR, "hf_home"),
    "TRANSFORMERS_CACHE": os.path.join(SCRATCH_DIR, "hf_cache"),
    "WANDB_CACHE_DIR": os.path.join(SCRATCH_DIR, "wandb_cache"),
    "HF_DATASETS_CACHE": os.path.join(SCRATCH_DIR, "hf_datasets"),
    "HF_METRICS_CACHE": os.path.join(SCRATCH_DIR, "hf_metrics"),
    "XDG_CACHE_HOME": os.path.join(SCRATCH_DIR, "xdg_cache"),
}
for env_var, path in cache_dirs.items():
    os.makedirs(path, exist_ok=True)
    os.environ[env_var] = path

# Define a simple MLP reward head that maps CLIP embeddings to a scalar score
class RewardModelHead(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.dense1 = nn.Linear(embedding_dim * 2, 256)
        self.dense2 = nn.Linear(256, 64)
        self.dense3 = nn.Linear(64, 1)

    def forward(self, image_embeds, text_embeds):
        x = torch.cat([image_embeds, text_embeds], dim=1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        return self.dense3(x).squeeze(-1)

# Define a custom dataset that tokenizes prompts from JSONL for training
class PreferenceDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=77):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        prompt = example["prompt"]
        tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "prompt": prompt,
            "input_ids": tokenized.input_ids.squeeze(0),
            "attention_mask": tokenized.attention_mask.squeeze(0),
        }

# Custom attention processor that fixes the dimension mismatch issues
class FixedCrossAttnProcessor(AttnProcessor):
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        batch_size, sequence_length, _ = hidden_states.shape

        # Prepare attention mask based on input size
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # Project hidden states to query vectors
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        # Prepare encoder hidden states
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # Project encoder hidden states to key and value vectors
        try:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        except RuntimeError as e:
            if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                # Resize encoder_hidden_states to match expected dimensions
                # This approach uses a simple projection to the expected size
                proj = torch.nn.Linear(
                    encoder_hidden_states.shape[-1], 
                    768  # Target dimension inferred from the error message
                ).to(encoder_hidden_states.device)
                encoder_hidden_states = proj(encoder_hidden_states)
                
                # Retry projection after fixing dimension mismatch
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
            else:
                # Raise if the error is unrelated
                raise e
            
        # Reshape key and value tensors for batched attention computation
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Compute scaled dot-product attention
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Apply output linear projection
        hidden_states = attn.to_out[0](hidden_states)

        # Apply output dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class RLHFTrainer:
    def __init__(self, config):
        self.config = config
        self.accelerator = Accelerator(
            mixed_precision=None,  # Disable mixed precision entirely
            gradient_accumulation_steps=1,
            log_with="wandb" if self.config.use_wandb else None,
        )

        if self.accelerator.is_main_process and self.config.use_wandb:
            wandb.init(project=self.config.wandb_project_name, name=self.config.wandb_run_name)

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # Set up reference model (SFT model) and policy model
        self._setup_models()
        
        # Load or initialize the reference model (SFT)
        self._initialize_reference_model()
        
        self._setup_dataset()

        self.trainable_params = []
        for n, p in self.unet.named_parameters():
            if p.requires_grad:
                self.trainable_params.append(p)
                
        self.unet, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader
        )

    def _setup_models(self):
        from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel
        self.device = self.accelerator.device

        
        # Load tokenizer and text encoder (frozen CLIP backbone)
        print("Loading tokenizer and models...")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
        
        # Load pretrained UNet model to act as the policy network π_θ
        self.unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="unet",
            low_cpu_mem_usage=False
        )
        
        # Replace all cross-attention modules with a patched processor to fix shape mismatches
        attn_processors = {}
        for name, attn_module in self.unet.attn_processors.items():
            if name.endswith("attn2.processor"):
                # For cross-attention, use our fixed processor
                attn_processors[name] = FixedCrossAttnProcessor()
            else:
                # For everything else, keep the default
                attn_processors[name] = AttnProcessor()
                
        self.unet.set_attn_processor(attn_processors)
        
        # Set scheduler
        self.scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        
        # Load reward model (CLIP) trained on human preferences
        self.reward_model = CLIPModel.from_pretrained("/scratch/rl4789/cjj/reward_model/reward-model-best/clip")

        # Load reward head (MLP) trained on top of frozen CLIP embeddings
        self.reward_head = RewardModelHead(embedding_dim=self.reward_model.config.projection_dim)
        self.reward_head.load_state_dict(torch.load("/scratch/rl4789/cjj/reward_model/reward-model-best/reward_head.pt"))

        # Freeze reward model and reward head during RLHF training
        self.reward_model.requires_grad_(False)
        self.reward_head.requires_grad_(False)
        self.reward_model.to(self.device)
        self.reward_head.to(self.device)
        
        # Freeze all parameters of reward model
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.reward_model.requires_grad_(False)
        
        # Set device
        self.device = self.accelerator.device
        self.text_encoder.to(self.device)
        self.vae.to(self.device)
        self.reward_model.to(self.device)
        
        # Make UNet parameters trainable
        trainable_modules = []
        param_count = 0
        
        # Selectively unfreeze parameters to reduce risk of dimension mismatch
        for name, param in self.unet.named_parameters():
            if "mid_block" in name:
                param.requires_grad = True
                trainable_modules.append(name)
                param_count += param.numel()
        
        print(f"Trainable modules: {len(trainable_modules)}")
        for name in trainable_modules[:10]:
            print(f"  - {name}")
        if len(trainable_modules) > 10:
            print(f"  ... and {len(trainable_modules) - 10} more")
        print(f"Total trainable parameters: {param_count:,}")
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.unet.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )
        
        if getattr(self.config, "lora_path", None) is not None:
            print(f"Loading LoRA adapter from {self.config.lora_path}")
            try:
                lora_attn_procs = AttnProcsLayers.load_pretrained(
                    pretrained_model_name_or_path=self.config.lora_path,
                    subfolder="attn_procs" if os.path.isdir(os.path.join(self.config.lora_path, "attn_procs")) else None,
                )
                self.unet.set_attn_processor(lora_attn_procs)
            except Exception as e:
                print(f"Failed to load LoRA from {self.config.lora_path}: {e}")

    def _initialize_reference_model(self):
        """Initialize the reference model (SFT model) for KL regularization"""
        print("Initializing reference model...")
        
        # Create a copy of the UNet for the reference model (μ)
        self.ref_unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="unet"
        )
        
        # Apply the same attention processors
        attn_processors = {}
        for name, attn_module in self.ref_unet.attn_processors.items():
            if name.endswith("attn2.processor"):
                attn_processors[name] = FixedCrossAttnProcessor()
            else:
                attn_processors[name] = AttnProcessor()
                
        self.ref_unet.set_attn_processor(attn_processors)
        
        # Reference model should be frozen
        self.ref_unet.requires_grad_(False)
        self.ref_unet.to(self.device)
        
        print("Reference model initialized.")

    def _encode_prompt(self, input_ids, attention_mask=None):
        """Encode the prompt into text embeddings."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )[0]
        
        return prompt_embeds

    def _setup_dataset(self):
        dataset = PreferenceDataset(
            jsonl_path=self.config.json_path,
            tokenizer=self.tokenizer,
        )
        self.train_dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def _compute_reward(self, images, prompts):
        """Compute reward using CLIP similarity between images and prompts"""
        with torch.no_grad():
            # Process images for CLIP
            # Convert to BGR for CLIP
            images = images.permute(0, 2, 3, 1)  # BCHW -> BHWC
            
            # Resize to CLIP expected size (224x224)
            images_resized = F.interpolate(
                images.permute(0, 3, 1, 2),  # BHWC -> BCHW
                size=(224, 224),
                mode="bicubic"
            )
            
            # Normalize with CLIP mean and std
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).reshape(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).reshape(1, 3, 1, 1)
            images_normalized = (images_resized - mean) / std
            
            # Tokenize prompts
            text_inputs = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get image and text features
            image_features = self.reward_model.get_image_features(images_normalized)
            text_features = self.reward_model.get_text_features(**text_inputs)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            # Compute similarity/reward (cosine similarity)
            rewards = self.reward_head(image_features, text_features)

            
        return rewards

    def _kl_divergence(self, p_logits, q_logits):
        """Compute KL divergence between policy and reference model logits"""
        p = F.softmax(p_logits, dim=1)
        q = F.softmax(q_logits, dim=1)
        
        kl_div = torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10)), dim=1)
        return kl_div

    def _latents_to_images(self, latents):
        """Convert latents to RGB images"""
        with torch.no_grad():
            # Scale latents
            latents = 1 / 0.18215 * latents
            
            # Decode latents to images
            images = self.vae.decode(latents).sample
            
            # Normalize to [0, 1]
            images = (images / 2 + 0.5).clamp(0, 1)
            
        return images

    def _generate_denoised_latents(self, latents, timesteps, text_embeddings, use_ref_model=False):
        """Generate denoised latents using policy or reference model"""
        model = self.ref_unet if use_ref_model else self.unet
        
        # Ensure all tensors are on the same device
        device = latents.device
        timesteps = timesteps.to(device)
        text_embeddings = text_embeddings.to(device)
        
        with torch.set_grad_enabled(not use_ref_model):
            # Get noise prediction from model
            noise_pred = model(
                latents,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample
            
            # Perform one step of denoising - make sure all tensor operations stay on same device
            timesteps_cpu = timesteps.cpu()  # Move to CPU for indexing scheduler arrays
            prev_timestep_value = max(0, self.scheduler.config.num_train_timesteps // self.config.num_inference_steps)
            prev_timestep = torch.clamp(timesteps_cpu - prev_timestep_value, min=0)
            
            # Get alpha values and move to appropriate device
            alpha_prod_t = torch.tensor(self.scheduler.alphas_cumprod.numpy()[timesteps_cpu.numpy()], device=device)
            alpha_prod_t_prev = torch.tensor(self.scheduler.alphas_cumprod.numpy()[prev_timestep.numpy()], device=device)
            
            # Compute the coefficient for the predicted noise
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            
            # Current prediction for x_0
            pred_original_sample = (latents - beta_prod_t.sqrt().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * noise_pred) / alpha_prod_t.sqrt().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
            # Direction pointing to x_t
            dir_xt = (1. - alpha_prod_t_prev).sqrt().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * noise_pred
            
            # Random noise
            noise = torch.randn_like(latents)
            
            # Compute the previous noisy sample
            prev_sample = alpha_prod_t_prev.sqrt().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * pred_original_sample + dir_xt + torch.sqrt(beta_prod_t_prev).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * noise
            
        return prev_sample, noise_pred

    def train(self):
        global_step = 0
        num_successful_steps = 0
        
        print(f"Starting RLHF training with KL coefficient τ={self.config.kl_coeff}")
        
        for epoch in range(self.config.num_epochs):
            self.unet.train()
            self.ref_unet.eval()
            
            progress_bar = tqdm(total=len(self.train_dataloader), disable=not self.accelerator.is_local_main_process)

            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    try:
                        # Transfer to device
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        prompts = batch["prompt"]

                        # Encode text with error catching
                        try:
                            text_embeddings = self._encode_prompt(input_ids, attention_mask)
                        except Exception as e:
                            print(f"Error encoding prompt: {e}")
                            self.optimizer.zero_grad()
                            progress_bar.update(1)
                            continue
                            
                        # Generate initial latents
                        batch_size = input_ids.shape[0]
                        latents = torch.randn((batch_size, 4, 64, 64), device=self.device)
                        
                        # Choose timestep in the early-to-mid range of diffusion process
                        # This gives better signal for the policy gradient
                        t_value = torch.randint(400, 700, (batch_size,), device=self.device)
                        
                        # Ensure t_value is on the same device as other tensors
                        t_value = t_value.to(self.device)
                        
                        # Add noise to latents according to timestep
                        noise = torch.randn_like(latents)
                        noisy_latents = self.scheduler.add_noise(latents, noise, t_value)
                        
                        # Make sure embeddings don't require gradients
                        text_embeddings = text_embeddings.detach()
                        
                        # Get policy and reference predictions
                        with torch.set_grad_enabled(True):
                            denoised_latents, policy_noise_pred = self._generate_denoised_latents(
                                noisy_latents, t_value.to(noisy_latents.device), text_embeddings, use_ref_model=False
                            )
                        
                        with torch.no_grad():
                            _, ref_noise_pred = self._generate_denoised_latents(
                                noisy_latents, t_value.to(noisy_latents.device), text_embeddings, use_ref_model=True
                            )
                        
                        # Convert denoised latents to images for reward computation
                        images = self._latents_to_images(denoised_latents)
                        
                        # Compute reward
                        rewards = self._compute_reward(images, prompts)
                        
                        # Compute KL divergence
                        kl_div = self._kl_divergence(policy_noise_pred.flatten(1), ref_noise_pred.flatten(1))
                        
                        # RLHF loss: reward - τ * KL(π_θ(·|x), μ(·|x))
                        # Note: We want to maximize reward and minimize KL divergence, so we negate both for gradient descent
                        policy_loss = -rewards + self.config.kl_coeff * kl_div
                        policy_loss = policy_loss.mean()
                        
                        # Backward and optimize
                        self.accelerator.backward(policy_loss)
                        self.accelerator.clip_grad_norm_(self.trainable_params, 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        
                        num_successful_steps += 1
                        
                    except Exception as e:
                        print(f"Error in training step: {e}")
                        self.optimizer.zero_grad()

                # Update progress
                progress_bar.update(1)
                global_step += 1

                # Logging
                if global_step % 10 == 0 and num_successful_steps > 0:
                    logs = {
                        "policy_loss": policy_loss.detach().item(),
                        "rewards": rewards.mean().detach().item(),
                        "kl_divergence": kl_div.mean().detach().item(),
                        "step": global_step,
                        "successful_steps": num_successful_steps
                    }
                    if self.accelerator.is_main_process:
                        print(f"Step {global_step}: loss={policy_loss.detach().item():.4f}, reward={rewards.mean().detach().item():.4f}, KL={kl_div.mean().detach().item():.4f}")
                    if self.config.use_wandb:
                        wandb.log(logs)

                # Save periodically
                if global_step % 100 == 0 and num_successful_steps > 0:
                    self._save_checkpoint(global_step)

            # Save at end of epoch
            if num_successful_steps > 0:
                self._save_checkpoint(f"epoch-{epoch}")

        # Final save
        if num_successful_steps > 0:
            self._save_checkpoint("final")
        
        print(f"Training completed with {num_successful_steps} successful steps out of {global_step} total steps")
        
        if self.config.use_wandb:
            wandb.finish()

    def _save_checkpoint(self, step):
        """Save a checkpoint of the model."""
        if not self.accelerator.is_main_process:
            return
        
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save the whole policy model (UNet)
        self.unet.save_pretrained(os.path.join(checkpoint_dir, "policy_model"))
        
        print(f"Saved checkpoint at {checkpoint_dir}")


class Args:
    json_path = "dataset path..."
    output_dir = "output path..."
    batch_size = 2  
    num_epochs = 5
    learning_rate = 1e-5
    use_wandb = True
    wandb_project_name = "sd-rlhf"
    wandb_run_name = "sd-rlhf-policy-gradient"
    kl_coeff = 0.05  # KL regularization coefficient 
    num_inference_steps = 50  # For denoising steps
    seed = 42
    lora_path = "lora_adapter_path..."

if __name__ == "__main__":
    args = Args()
    os.makedirs(args.output_dir, exist_ok=True)
    trainer = RLHFTrainer(args)
    trainer.train()