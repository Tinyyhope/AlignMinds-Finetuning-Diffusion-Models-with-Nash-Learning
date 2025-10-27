#!/usr/bin/env python
"""
This file implements a modified and extended version of the Nash Learning from Human Feedback (NLHF) algorithm,
originally proposed in [NLHF: http://arxiv.org/abs/2312.00886].

This implementation is adapted specifically for fine-tuning text-to-image diffusion models using LoRA, 
which differs significantly from the original NLHF paper designed for language modeling 
(including the loss function and approximaiton tech, etc).

Our goal is to align diffusion-based generative models with human preferences by combining 
KL-regularized policy optimization with pairwise preference supervision.

Key features:
- Supports current and reference policies defined over diffusion trajectories (UNet noise prediction).
- Replaces language modeling input/output with prompt-conditioned image generation in latent space.
- Defines a novel loss function inspired by Nash Learning from Human Feedback (NLHF) and Direct Preference Optimization (DPO), 
  tailored for the diffusion setting.
- Instead of relying on scalar reward models, we use pairwise human preference labels as direct supervision.
- We approximate the preference function via offline comparison rather than training a parameterized neural preference model, 
  which we show is more efficient and stable for image-level alignment.

Attribution:
- Portions of this implementation, including KL regularization structure, noise prediction scoring, and alternative policy design, 
  are adapted and extended from the official NLHF repository.

Please cite the original NLHF paper if you build upon this implementation.
"""

####Note: You need GPU(s) to run this program #####


from diffusers.models.attention_processor import LoRAAttnProcessor
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, LambdaLR
from tqdm import tqdm
import logging
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.loaders import AttnProcsLayers
import random
from copy import deepcopy
from datasets import load_dataset
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch_ema
import argparse 
from datetime import datetime
import os.path as osp
import json
from typing import Dict, List, Tuple, Union, Optional, Any

# Optional accelerate support for multi-GPU
try:
    from accelerate import Accelerator
    from accelerate.utils import find_executable_batch_size
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

# =============================================================# 
# Utility funcitons
# =============================================================#
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("diffusion_nlhf.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DiffusionNLHF")
def check_parameter_updates(model, step):
    """
    Check if LoRA parameters are being updated correctly during training.
    """
    if not hasattr(check_parameter_updates, "previous_params"):
        check_parameter_updates.previous_params = {}
        # First time: store current parameters
        for name, module in model.pipe.unet.attn_processors.items():
            for param_name, param in module.named_parameters():
                full_name = f"{name}.{param_name}"
                if param.requires_grad:
                    check_parameter_updates.previous_params[full_name] = param.data.clone().detach()
        return {"changed": False, "unchanged": 0, "total": 0}

    changes = {"changed": False, "unchanged": 0, "total": 0}
    for name, module in model.pipe.unet.attn_processors.items():
        for param_name, param in module.named_parameters():
            full_name = f"{name}.{param_name}"
            if param.requires_grad:
                changes["total"] += 1
                if full_name in check_parameter_updates.previous_params:
                    is_same = torch.allclose(param.data, check_parameter_updates.previous_params[full_name])
                    if is_same:
                        changes["unchanged"] += 1
                    else:
                        changes["changed"] = True
                    # Update stored param
                    check_parameter_updates.previous_params[full_name] = param.data.clone().detach()
    
    if changes["unchanged"] == changes["total"]:
        logger.warning(f"STEP {step}: NO parameters changed in this step!")
    else:
        logger.info(f"STEP {step}: {changes['total'] - changes['unchanged']}/{changes['total']} parameters changed")
    
    return changes


def ensure_model_on_device(model, device):
    """Recursively ensures all parameters and buffers are on the specified device."""
    model = model.to(device)
    # Double-check parameters
    for param in model.parameters():
        if param.device != device:
            param.data = param.data.to(device)
    # Double-check buffers
    for buffer_name, buffer in model.named_buffers():
        if buffer.device != device:
            buffer.data = buffer.data.to(device)
    return model

# =============================================================#
# Task type selection (set exactly one to 1)
# =============================================================#

USE_ATARI     = 0
USE_NLP       = 0
USE_DIFFUSION = 1  #this project focuses on diffusion-based image generation

assert sum([USE_ATARI, USE_NLP, USE_DIFFUSION]) == 1, \
    "Exactly one of USE_ATARI, USE_NLP, USE_DIFFUSION must be set to 1"


# =============================================================#
# Algorithm and Optimizer Selection
# =============================================================#

USE_IPO_MD = 0
USE_NLHF = int(not USE_IPO_MD)


USE_ADAMW_ON_LION = 0
USE_ADAMW         = ~USE_ADAMW_ON_LION

#############################################
# In case the dataset is missing
#############################################
class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return {
            "prompt": "a dummy prompt",
            "image_a": torch.zeros(3, 512, 512),
            "image_b": torch.zeros(3, 512, 512),
            "preference": torch.tensor(1, dtype=torch.long)
        }

# =============================================================#
# Hardware and Debugging Config
# =============================================================#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16

torch.autograd.set_detect_anomaly(True)
debugging_is_on = False


# Debug helper function
def print_tensor_info(tensor_name, tensor):
    tensor_float = tensor.float() if not tensor.is_floating_point() else tensor
    info = {
        "shape": tuple(tensor.shape),
        "min/max": (tensor.min().item(), tensor.max().item()),
        "mean": tensor_float.mean().item(),
        "std": tensor_float.std().item()
    }
    print(f"{tensor_name} = {tensor}")
    for key, value in info.items():
        print(f"{key}: {value}")


# =============================================================#
# Hyperparameters (context-aware). We do ablation study here.
# =============================================================#

# learning rate
if USE_IPO_MD:
    lr=5e-5
elif USE_NLHF:
    lr = 5e-5 # ← default 

# tau is a temperature parameter controlling the degree of KL regularization toward π_ref
if USE_IPO_MD:
    tau = 0.001
elif USE_NLHF:
    tau = 0.001

# Additional override for diffusion-based tuning
if USE_DIFFUSION:
    lr = 5e-5  # safer for LoRA tuning
    tau = 0.001  # may need stronger KL regularization for image distribution

# Mixing ratio between current policy and reference policy 
# Trade-off between exploration and exploitation
beta = 0.125

# Linear combination weight for preference loss 
alpha = 0.5

# =============================================================#
# NLHF Config Structure
# =============================================================#

class NLHFConfig:
    def __init__(self,
                 use_ema=True,
                 use_md_pg=True,
                 use_variance_reduction=True,
                 kl_mode='single',   # 'single' or 'multi'
                 n_kl_steps=1,
                 beta=beta,
                 tau=tau,
                 lora_r=8,          # LoRA rank
                 mixed_precision=True,
                 output_dir="checkpoints",
                 save_steps=500,
                 gradient_accumulation_steps=1,
                 resume_from_checkpoint=None,
                 use_accelerate=False,
                 loss_pref_current_weight=0.5,  # Weight for current policy preference loss
                 loss_pref_alt_weight=0.5,      # Weight for alternative policy preference loss
                 return_latents=False):         # Whether to return latents in forward pass
        self.use_ema = use_ema
        self.use_md_pg = use_md_pg
        self.use_variance_reduction = use_variance_reduction
        self.kl_mode = kl_mode
        self.n_kl_steps = n_kl_steps
        self.beta = beta
        self.tau = tau
        self.lora_r = lora_r
        self.mixed_precision = mixed_precision
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.resume_from_checkpoint = resume_from_checkpoint
        self.use_accelerate = use_accelerate and HAS_ACCELERATE
        self.loss_pref_current_weight = loss_pref_current_weight
        self.loss_pref_alt_weight = loss_pref_alt_weight
        self.return_latents = return_latents
        
    def save(self, path):
        """Save config to JSON file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            # Convert to dict and save only serializable values
            config_dict = {k: v for k, v in self.__dict__.items() 
                           if not k.startswith('_') and not callable(v)}
            json.dump(config_dict, f, indent=2)
            
    @classmethod
    def load(cls, path):
        """Load config from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


if USE_DIFFUSION:

    class PolicyNetwork(nn.Module):
        def __init__(self, model_name="runwayml/stable-diffusion-v1-5", lora_path=None, lora_r=16):
            super().__init__()
            
            # Determine data type based on device capability
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Initializing PolicyNetwork with target device: {target_device}, dtype: {dtype}")
            
            # base model
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
                variant="fp16",
                safety_checker=None
            ).to(target_device)
            
            # Initialize a fixed noise scheduler to ensure consistency
            self.pipe.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
            
        
            # Move all pipeline components to GPU 
            if torch.cuda.is_available():
                logger.info(f"Moving entire pipeline to {target_device}")
                self.pipe.to(target_device) 
                
                # Double-check key components explicitly
                self.pipe.text_encoder = self.pipe.text_encoder.to(target_device)
                self.pipe.unet = self.pipe.unet.to(target_device)
                self.pipe.vae = self.pipe.vae.to(target_device)
                
                logger.info(f"Text encoder device: {next(self.pipe.text_encoder.parameters()).device}")
                logger.info(f"UNet device: {next(self.pipe.unet.parameters()).device}")
                            
                # Initialize and apply LoRA attention processors

            def apply_lora_to_unet(unet, lora_r=4):
                lora_attn_procs = {}
                for name in unet.attn_processors.keys():
                    cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

                    if name.startswith("mid_block"):
                        hidden_size = unet.config.block_out_channels[-1]
                    elif name.startswith("up_blocks"):
                        block_id = int(name[len("up_blocks."):].split(".")[0])
                        hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                    elif name.startswith("down_blocks"):
                        block_id = int(name[len("down_blocks."):].split(".")[0])
                        hidden_size = unet.config.block_out_channels[block_id]
                    else:
                        raise ValueError(f"Unknown attention processor name: {name}")

                    lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=lora_r
                )

                unet.set_attn_processor(lora_attn_procs)

            if lora_path is None:
                logger.info(f"Initializing new LoRA with rank {lora_r}")
                apply_lora_to_unet(self.pipe.unet, lora_r)
            else:
                logger.info(f"Loading LoRA from {lora_path}")
                self.pipe.unet.load_attn_procs(lora_path)

                # Make loaded LoRA parameters trainable
                for name, module in self.pipe.unet.named_modules():
                    if 'lora' in name.lower():
                        for param in module.parameters():
                            param.requires_grad = True

        
        def save_lora(self, save_path):
            """Save only the LoRA parameters to disk"""
            os.makedirs(save_path, exist_ok=True)
            logger.info(f"Saving LoRA weights to {save_path}")
            
            # Get attention processors dictionary
            attn_procs_dict = {}
            for name, module in self.pipe.unet.attn_processors.items():
                if isinstance(module, LoRAAttnProcessor):
                    attn_procs_dict[name] = module
            
            # Extract and save only the LoRA parameters
            self.pipe.unet.save_attn_procs(save_path)

            logger.info(f"LoRA weights saved successfully to {save_path}")

            
        def save_checkpoint(self, save_dir, global_step, optimizer=None, scheduler=None, ema=None):
            """
            Save a complete training checkpoint including optimizer state
            for resuming training later
            """
            os.makedirs(save_dir, exist_ok=True)
            
            # Create checkpoint directory with step number
            checkpoint_dir = os.path.join(save_dir, f"checkpoint-{global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save LoRA weights
            lora_path = os.path.join(checkpoint_dir, "lora_weights")
            self.save_lora(lora_path)

            
            # Save optimizer and scheduler states if provided
            checkpoint_dict = {
                "global_step": global_step,
            }
            
            if optimizer is not None:
                checkpoint_dict["optimizer"] = optimizer.state_dict()
                
            if scheduler is not None:
                checkpoint_dict["scheduler"] = scheduler.state_dict()
                
            if ema is not None:
                checkpoint_dict["ema"] = ema.state_dict()
                
            # Save checkpoint metadata
            checkpoint_path = os.path.join(checkpoint_dir, "training_state.pt")
            torch.save(checkpoint_dict, checkpoint_path)
            
            logger.info(f"Saved full checkpoint to {checkpoint_dir}")
            return checkpoint_dir
            
        @classmethod
        def from_checkpoint(cls, checkpoint_dir, optimizer=None, scheduler=None, ema=None):
            """
            Resume from a checkpoint directory
            Returns the model, global_step, and updated optimizer/scheduler/ema if provided
            """
            # Load LoRA weights
            lora_path = os.path.join(checkpoint_dir, "lora_weights")
            model = cls(lora_path=lora_path, lora_r=16)

    
            # Load training state
            training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
            if os.path.exists(training_state_path):
                logger.info(f"Loading training state from {training_state_path}")
                training_state = torch.load(training_state_path)
                
                global_step = training_state["global_step"]
                
                # Restore optimizer state if provided
                if optimizer is not None and "optimizer" in training_state:
                    optimizer.load_state_dict(training_state["optimizer"])
                    
                # Restore scheduler state if provided
                if scheduler is not None and "scheduler" in training_state:
                    scheduler.load_state_dict(training_state["scheduler"])
                
                # Restore EMA state if provided
                if ema is not None and "ema" in training_state:
                    ema.load_state_dict(training_state["ema"])
                    
                logger.info(f"Resumed from step {global_step}")
                return model, global_step, optimizer, scheduler, ema
            else:
                logger.warning(f"No training state found in {checkpoint_dir}, starting from step 0")
                return model, 0, optimizer, scheduler, ema

        @torch.no_grad()
        def forward(self, prompts, guidance_scale=7.5, num_inference_steps=25, seed=None, return_latents=False):
            """
            Args:
                prompts: list of strings (batch of text prompts)
                guidance_scale: classifier-free guidance (default 7.5)
                num_inference_steps: diffusion steps (default 25)
                seed: random seed for deterministic generation (optional)
                return_latents: whether to return latents instead of PIL images
            Returns:
                If return_latents=False: List[PIL.Image] (generated images)
                If return_latents=True: Tuple[List[PIL.Image], torch.Tensor] (images and latents)
            """
            # ---  Fix: always get dynamic device ---
            unet_device = next(self.pipe.unet.parameters()).device

            # Match batch size for seed generation
            if seed is not None:
                generator = torch.Generator(device=unet_device).manual_seed(seed)
            else:
                seed = random.randint(0, 999999)
                generator = torch.Generator(device=unet_device).manual_seed(seed)
            
            # Ensure consistent number of prompts
            if isinstance(prompts, str):
                prompts = [prompts]
                
            # Handle tokenizer mismatch by padding/truncating
            try:
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    if return_latents:
                        # For return_latents, we need access to the latents
                        latents = torch.randn(
                            (len(prompts), 4, 64, 64),
                            generator=generator,
                            device=unet_device
                        )
                        
                        output = self.pipe(
                            prompts,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            latents=latents,
                            return_dict=True,
                            output_type="pt" if return_latents else "pil"
                        )
                        
                        return output.images, latents  # Return both final images and initial latents
                    else:
                        output = self.pipe(
                            prompts,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            return_dict=True
                        )
                    
                        return output.images  # List of PIL.Image
                    
            except Exception as e:
                logger.error(f"Error during inference: {str(e)}")
                # Return black images if generation fails
                black_images = [Image.new('RGB', (512, 512), color='black') for _ in range(len(prompts))]
                if return_latents:
                    # Create empty latents as fallback
                    empty_latents = torch.zeros((len(prompts), 4, 64, 64), device=unet_device)
                    return black_images, empty_latents
                return black_images

# =============================================================#
# Reference and Alternative Policy Utilities
# =============================================================#


def get_alternative_noise(eps_pred_current, eps_pred_reference, cfg: NLHFConfig):
    """
    Returns π′ in denoising space, either MD-PG (geometric mix) or EMA.
    """
    # Ensure both inputs are on the same device and have the same dtype
    if eps_pred_current.device != eps_pred_reference.device:
        eps_pred_reference = eps_pred_reference.to(device=eps_pred_current.device)
        
    if eps_pred_current.dtype != eps_pred_reference.dtype:
        eps_pred_reference = eps_pred_reference.to(dtype=eps_pred_current.dtype)
        
    if cfg.use_md_pg:
        return (1 - cfg.beta) * eps_pred_current + cfg.beta * eps_pred_reference
    else:
        return eps_pred_reference



def preference_model_single_step(
    prompt_batch, preference_labels, image_a_batch, image_b_batch,
    current_policy, reference_policy, cfg: NLHFConfig, global_step=0
):
    batch_size = len(prompt_batch)
    scheduler = current_policy.pipe.scheduler
    scheduler.set_timesteps(50)

    # Move models to correct device
    unet_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_policy.pipe.unet = ensure_model_on_device(current_policy.pipe.unet, unet_device)
    reference_policy.pipe.unet = ensure_model_on_device(reference_policy.pipe.unet, unet_device)
    current_policy.pipe.text_encoder = ensure_model_on_device(current_policy.pipe.text_encoder, unet_device)
    reference_policy.pipe.text_encoder = ensure_model_on_device(reference_policy.pipe.text_encoder, unet_device)
    current_policy.pipe.vae = ensure_model_on_device(current_policy.pipe.vae, unet_device)
    reference_policy.pipe.vae = ensure_model_on_device(reference_policy.pipe.vae, unet_device)

    # Encode prompts
    try:
        input_ids = current_policy.pipe.tokenizer(
            prompt_batch, return_tensors="pt",
            padding="max_length", max_length=77, truncation=True
        ).input_ids.to(unet_device)
        text_embeddings = current_policy.pipe.text_encoder(input_ids)[0]
    except Exception as e:
        logger.error(f"Error during tokenization: {str(e)}")
        prompts_fixed = [p[:200] for p in prompt_batch]
        input_ids = current_policy.pipe.tokenizer(
            prompts_fixed, return_tensors="pt",
            padding="max_length", max_length=77, truncation=True
        ).input_ids.to(unet_device)
        text_embeddings = current_policy.pipe.text_encoder(input_ids)[0]

    # Encode real images into latents
    with torch.no_grad():
        latents_a = current_policy.pipe.vae.encode(image_a_batch.to(unet_device, dtype=current_policy.pipe.vae.dtype)).latent_dist.sample()
        latents_b = current_policy.pipe.vae.encode(image_b_batch.to(unet_device, dtype=current_policy.pipe.vae.dtype)).latent_dist.sample()
        latents_a = latents_a * 0.18215
        latents_b = latents_b * 0.18215

    # Sample different noise for each image
    batch_seed = 42 + global_step
    torch.manual_seed(batch_seed)
    noise_a = torch.randn_like(latents_a, device=unet_device)
    noise_b = torch.randn_like(latents_b, device=unet_device)

    timestep_weights = torch.linspace(1.0, 0.1, scheduler.config.num_train_timesteps).to(unet_device)
    timestep_indices = torch.multinomial(timestep_weights, batch_size, replacement=True)
    timesteps = timestep_indices.to(unet_device).long()

    with torch.autocast(device_type="cuda" if cfg.mixed_precision and torch.cuda.is_available() else "cpu",
                        enabled=cfg.mixed_precision):
        # Add noise
        noisy_latents_a = scheduler.add_noise(latents_a, noise_a, timesteps)
        noisy_latents_b = scheduler.add_noise(latents_b, noise_b, timesteps)

        # Predict noise
        eps_pred_current_a = current_policy.pipe.unet(noisy_latents_a, timesteps, encoder_hidden_states=text_embeddings).sample
        eps_pred_current_b = current_policy.pipe.unet(noisy_latents_b, timesteps, encoder_hidden_states=text_embeddings).sample

        eps_pred_reference_a = reference_policy.pipe.unet(noisy_latents_a, timesteps, encoder_hidden_states=text_embeddings).sample
        eps_pred_reference_b = reference_policy.pipe.unet(noisy_latents_b, timesteps, encoder_hidden_states=text_embeddings).sample

        # Move to correct dtype
        eps_pred_current_a = eps_pred_current_a.to(unet_device, dtype=noise_a.dtype)
        eps_pred_current_b = eps_pred_current_b.to(unet_device, dtype=noise_b.dtype)
        eps_pred_reference_a = eps_pred_reference_a.to(unet_device, dtype=noise_a.dtype)
        eps_pred_reference_b = eps_pred_reference_b.to(unet_device, dtype=noise_b.dtype)

        # Alternative policies
        eps_pred_alt_a = get_alternative_noise(eps_pred_current_a, eps_pred_reference_a, cfg).to(unet_device, dtype=noise_a.dtype)
        eps_pred_alt_b = get_alternative_noise(eps_pred_current_b, eps_pred_reference_b, cfg).to(unet_device, dtype=noise_b.dtype)

        # Compute scores on both image A and B
        score_current_a = -F.mse_loss(eps_pred_current_a, noise_a, reduction='none').mean(dim=[1, 2, 3])
        score_alt_a     = -F.mse_loss(eps_pred_alt_a,     noise_a, reduction='none').mean(dim=[1, 2, 3])
        score_current_b = -F.mse_loss(eps_pred_current_b, noise_b, reduction='none').mean(dim=[1, 2, 3])
        score_alt_b     = -F.mse_loss(eps_pred_alt_b,     noise_b, reduction='none').mean(dim=[1, 2, 3])

        # Use labels to select which image’s score to use
        preference_labels = preference_labels.to(unet_device, dtype=torch.float32)
        score_current = preference_labels * score_current_a + (1.0 - preference_labels) * score_current_b
        score_alt     = preference_labels * score_alt_a     + (1.0 - preference_labels) * score_alt_b

        # DPO-style preference loss
        logits = score_current - score_alt
        
        target = torch.ones_like(logits, device=unet_device)
        loss_pref = F.binary_cross_entropy_with_logits(logits, target)

    # KL loss between current and reference (on image_a only)
    eps_pred_current_float = eps_pred_current_a.to(dtype=torch.float32)
    eps_pred_reference_float = eps_pred_reference_a.to(dtype=torch.float32)
    loss_kl = F.mse_loss(eps_pred_current_float, eps_pred_reference_float)

    # LoRA Regularization
    l2_reg = 0.0
    for module in current_policy.pipe.unet.attn_processors.values():
        if isinstance(module, LoRAAttnProcessor):
            for param in module.parameters():
                if param.requires_grad:
                    l2_reg += param.norm(2)

    total_loss = loss_pref + cfg.tau * loss_kl + 1e-5 * l2_reg
    total_loss = total_loss.to(unet_device)

    return total_loss, {
        "loss_pref": loss_pref.item(),
        "loss_kl": loss_kl.item(),
        "loss_total": total_loss.item(),
        "l2_reg": l2_reg.item()
    }


def preference_model(prompt_batch, preference_labels, image_a_batch, image_b_batch,
                     current_policy, reference_policy, cfg: NLHFConfig, global_step=0):
    if cfg.kl_mode == "single":
        return preference_model_single_step(
            prompt_batch, preference_labels, image_a_batch, image_b_batch,
            current_policy, reference_policy, cfg, global_step
        )
    
    else:
        raise ValueError(f"Unknown kl_mode: {cfg.kl_mode}")


class Hpdv2ImageDataset(Dataset):
    def __init__(self, dataset, image_folder):
        self.dataset = dataset
        self.image_folder = image_folder
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        prompt = entry["prompt"]
        preference_list = entry["human_preference"]
        
        # Decide preference: if [0,1] → preference=1 (choose 2nd image), if [1,0] → preference=0 (choose 1st image)
        if preference_list == [0, 1]:
            preference = 1
        elif preference_list == [1, 0]:
            preference = 0
        else:
            raise ValueError(f"Invalid human_preference {preference_list} at idx {idx}")
        
        try:
            # Correctly load images by joining path
            image_a_path = os.path.join(self.image_folder, entry["image_path"][0])
            image_b_path = os.path.join(self.image_folder, entry["image_path"][1])

            image_a = self.transform(Image.open(image_a_path).convert("RGB"))
            image_b = self.transform(Image.open(image_b_path).convert("RGB"))
            image_a = image_a.to(device)
            image_b = image_b.to(device)

        except Exception as e:
            logger.error(f"Error loading images for index {idx}: {str(e)}")
            image_a = torch.zeros(3, 512, 512, device=device)
            image_b = torch.zeros(3, 512, 512, device=device)

        return {
            "prompt": prompt,
            "image_a": image_a,
            "image_b": image_b,
            "preference": torch.tensor(preference, dtype=torch.long, device=device)  # 🔥 加 device
        }

def load_hpdv2_dataloaders(batch_size=8, val_ratio=0.1, seed=42, dataset_jsonl=None, image_folder=None):
    logger.info("Loading dataset...")

    if dataset_jsonl is not None:
        logger.info(f"Loading local dataset from {dataset_jsonl}")
        dataset = load_dataset("json", data_files=dataset_jsonl, split="train")
        
        split = dataset.train_test_split(test_size=val_ratio, seed=seed)
        train_data = Hpdv2ImageDataset(split["train"], image_folder)
        val_data = Hpdv2ImageDataset(split["test"], image_folder)
        test_data = Hpdv2ImageDataset(dataset, image_folder)
    else:
        raise ValueError("For local data, must provide dataset_jsonl and image_folder")
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader




from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable

class Lion(Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99),
                    weight_decay=0.0, differentiable=True):
        """Initialize the hyperparameters.

        Args:
            params (iterable): iterable of parameters to optimize or
                                dicts defining parameter groups
            lr (float, optional): learning rate (default: 1e-4)
            betas (Tuple[float, float], optional): coefficients used
                            for computing running averages of gradient
                            and its square (default: (0.9, 0.99))
            weight_decay (float, optional): weight decay coefficient
                                            (default: 0)
        """

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.defaults['differentiable'] = differentiable  # Initialize

        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'
                                .format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'
                                .format(betas[1]))

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates
                                        the model and returns the loss.

        Returns:
            the loss.
        """
        loss = None
        updates = []
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)

                # Store the update
                updates.append(-update.sign_())

                p.add(update.sign_(), alpha=-group['lr'])

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return updates  # Return updates for all parameters


# Credit : https://github.com/egg-west/AdamW-pytorch/blob/master/adamW.py
class AdamW(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        step_sizes = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                step_sizes.append(step_size)

                p.data.addcdiv_(torch.mul(p.data, group['weight_decay']), denom, value=-step_size).add_(exp_avg, alpha=-step_size)

        return step_sizes


class AdamW_on_Lion_Optimizer(Optimizer):
    def __init__(self, params, lr=1e-3, adam_betas=(0.9, 0.999),
                    lion_betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

        self.params = list(params)

        # Define the Adam and Lion optimizers
        self.adamW = AdamW(self.params, lr=lr, betas=adam_betas,
                                    eps=eps, weight_decay=weight_decay)
        self.lion = Lion(self.params, lr=lr, betas=lion_betas,
                            weight_decay=weight_decay)

        self.scheduler_adamW = CosineAnnealingWarmRestarts(self.adamW, T_0=5, T_mult=2)
        self.scheduler_lion = CosineAnnealingWarmRestarts(self.lion, T_0=5, T_mult=2)

        defaults = dict(lr=lr, adam_betas=adam_betas,
                        lion_betas=lion_betas, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(self.params, defaults)

    def get_current_lr(self, optimizer):
        """Retrieves the current learning rate, considering potential schedulers.
        """
        # Typically, the learning rate is stored in the first param_group
        # assuming all param_groups have the same lr if they exist
        return optimizer.param_groups[0]['lr']

    def step(self, lr=1e-3, max_iter=25, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates
                                        the model and returns the loss.

        Returns:
            the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Retrieve current learning rates from the optimizers
        lr_adamW = self.get_current_lr(self.adamW)
        lr_lion = self.get_current_lr(self.lion)

        for i in range(max_iter):
            # Apply the Lion and Adam optimizer
            lion_updates = self.lion.step()
            adamW_updates = self.adamW.step()

            scaled_updates = []
            for lion_update, adamW_update in zip(lion_updates, adamW_updates):
                # Implement scaling logic with individual 'lion_update' and 'adamw_update'

                # See [Learning Rate Grafting Transferability of Optimizer Tuning]
                # (https://openreview.net/forum?id=FpKgG31Z_i9)
                # Grafting adamW#lion: update direction from lion, update magnitude from adamW
                # scaled_update = lion_step * (adamW_norms / lion_norms)
                # Incorporate learning rates into the scaling factor (lr_adamW / lr_lion)
                if isinstance(lion_update, torch.Tensor) and isinstance(adamW_update, torch.Tensor):
                    lion_norm = torch.norm(lion_update) + 1e-10
                    adamW_norm = torch.norm(adamW_update)
                    scaled_update = (lr_adamW / lr_lion) * lion_update * (adamW_norm / lion_norm)
                else:
                    # Fallback if not tensors
                    scaled_update = adamW_update

                scaled_updates.append(scaled_update)

            # Update model weights
            for param, update in zip(self.params, scaled_updates):
                if isinstance(update, torch.Tensor):
                    param.data.add_(update, alpha=-self.defaults['lr'])

            # Step the schedulers
            self.scheduler_adamW.step()
            self.scheduler_lion.step()

        return scaled_updates


# =============================================================
# Training and Evaluation Functions
# =============================================================


def train_one_step(batch, current_policy, reference_policy, optimizer, ema, cfg, 
                   step, accumulation_step=0, accelerator=None):
    current_policy.train()

    cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_device = next(current_policy.pipe.unet.parameters()).device
    
    if torch.cuda.is_available() and unet_device.type != "cuda":
        logger.warning(f"UNet not on CUDA in train_one_step! Moving it now.")
        current_policy.pipe.unet = current_policy.pipe.unet.to(cuda_device)
        reference_policy.pipe.unet = reference_policy.pipe.unet.to(cuda_device)
        unet_device = cuda_device

    prompt_batch = batch["prompt"]
    preference_labels = batch["preference"].to(unet_device).float()
    image_a_batch = batch["image_a"].to(unet_device)
    image_b_batch = batch["image_b"].to(unet_device)

    with torch.autocast(device_type="cuda" if cfg.mixed_precision and torch.cuda.is_available() else "cpu", 
                        enabled=cfg.mixed_precision):
        loss, log_dict = preference_model(
            prompt_batch=prompt_batch,
            preference_labels=preference_labels,
            image_a_batch=image_a_batch,
            image_b_batch=image_b_batch,
            current_policy=current_policy,
            reference_policy=reference_policy,
            cfg=cfg,
            global_step=step
        )
           
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
    
    if torch.cuda.is_available() and loss.device.type != "cuda":
        logger.warning(f"Loss not on CUDA before backward! Moving from {loss.device} to {cuda_device}")
        loss = loss.to(cuda_device)

    # Backward pass
    if accelerator:
        accelerator.backward(loss)
    else:
        loss.backward()

    # === (1) Check gradient after backward
    nonzero_grads = []
    for name, module in current_policy.pipe.unet.attn_processors.items():
        for param_name, param in module.named_parameters():
            full_name = f"{name}.{param_name}"
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum().item() > 0:
                    nonzero_grads.append(full_name)

    # === (2) Optimizer step
    if (accumulation_step + 1) % cfg.gradient_accumulation_steps == 0:
        # (2.1) Gradient clipping
        trainable_params = []
        for module in current_policy.pipe.unet.attn_processors.values():
            trainable_params.extend([p for p in module.parameters() if p.requires_grad])

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)

        # (2.2) Debug top grad norms
        grad_norms = []
        for name, module in current_policy.pipe.unet.attn_processors.items():
            for param_name, param in module.named_parameters():
                full_name = f"{name}.{param_name}"
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append((full_name, grad_norm))
                    
                    if param.grad.abs().mean().item() < 1e-8:
                        logger.warning(f"Near-zero gradient detected for {full_name}: {param.grad.abs().mean().item()}")
                    
                    noise_scale = 1e-5 * param.grad.abs().mean().item()
                    param.grad.data.add_(torch.randn_like(param.grad) * noise_scale)

        grad_norms.sort(key=lambda x: x[1], reverse=True)
        optimizer.step()
        optimizer.zero_grad()

        # (2.4) EMA update and synchronize reference_policy
        if cfg.use_ema and ema:
            ema.update()
            with ema.average_parameters():
                for p_ref, p_cur in zip(reference_policy.parameters(), current_policy.parameters()):
                    p_ref.data.copy_(p_cur.data)

    return loss.item() * cfg.gradient_accumulation_steps, log_dict



def evaluate(current_policy, reference_policy, val_loader, cfg):
    current_policy.eval()
    total_val_loss = 0.0
    num_batches = 0

    for batch in tqdm(val_loader, desc="Evaluating"):
        prompt_batch = batch["prompt"]
        preference_labels = batch["preference"].to(device).float()
        image_a_batch = batch["image_a"].to(device)
        image_b_batch = batch["image_b"].to(device)

        with torch.no_grad():
            val_loss, _ = preference_model(
                prompt_batch=prompt_batch,
                preference_labels=preference_labels,
                image_a_batch=image_a_batch,
                image_b_batch=image_b_batch,
                current_policy=current_policy,
                reference_policy=reference_policy,
                cfg=cfg
            )

        total_val_loss += val_loss.item()
        num_batches += 1
    
    avg_val_loss = total_val_loss / num_batches
    return avg_val_loss



def test_lemma1_diffusion(current_policy, reference_policy, prompt_batch, cfg):
    if torch.cuda.is_available():
        unet_device = torch.device("cuda")
        # Ensure the UNet is on CUDA using our helper function
        current_policy.pipe.unet = ensure_model_on_device(current_policy.pipe.unet, unet_device)
        reference_policy.pipe.unet = ensure_model_on_device(reference_policy.pipe.unet, unet_device)
    else:
        unet_device = next(current_policy.pipe.unet.parameters()).device
    
    # Ensure text_encoder is on the same device as UNet
    current_policy.pipe.text_encoder = ensure_model_on_device(current_policy.pipe.text_encoder, unet_device)
    reference_policy.pipe.text_encoder = ensure_model_on_device(reference_policy.pipe.text_encoder, unet_device)
    
    scheduler = current_policy.pipe.scheduler
    scheduler.set_timesteps(50)

    try:
        input_ids = current_policy.pipe.tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True
        ).input_ids.to(unet_device) 
        
        text_embeddings = current_policy.pipe.text_encoder(input_ids)[0]
        text_embeddings = text_embeddings.to(unet_device)

    except Exception as e:
        logger.error(f"Error in test_lemma1: {str(e)}")
        fallback_prompts = ["a photo of a cat"] * len(prompt_batch)
        input_ids = current_policy.pipe.tokenizer(
            fallback_prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True
        ).input_ids.to(unet_device)  
        
        text_embeddings = current_policy.pipe.text_encoder(input_ids)[0]
        text_embeddings = text_embeddings.to(unet_device)

    batch_size = len(prompt_batch)
    latents = torch.randn((batch_size, 4, 64, 64), device=unet_device)
    noise = torch.randn_like(latents, device=unet_device) 
    timesteps = torch.randint(
        low=0,
        high=scheduler.config.num_train_timesteps,
        size=(batch_size,),
        device=unet_device
    ).long()
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    with torch.autocast(device_type="cuda" if cfg.mixed_precision and torch.cuda.is_available() else "cpu",
                        enabled=cfg.mixed_precision):
        eps_pi = current_policy.pipe.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
        eps_mu = reference_policy.pipe.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
        
        # Ensure consistent device for all tensors
        eps_pi = eps_pi.to(device=unet_device)
        eps_mu = eps_mu.to(device=unet_device)
        
        eps_pi_mu = (1 - cfg.beta) * eps_pi + cfg.beta * eps_mu
        eps_random = current_policy.pipe.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
        eps_random = eps_random.to(device=unet_device)

    eps_pi = eps_pi.to(dtype=torch.float32)
    eps_mu = eps_mu.to(dtype=torch.float32)
    eps_pi_mu = eps_pi_mu.to(dtype=torch.float32)
    eps_random = eps_random.to(dtype=torch.float32)

    def kl(a, b):
        return 0.5 * ((a - b) ** 2).mean()

    eta_tau = cfg.beta
    lhs = kl(eps_random, eps_pi_mu)
    rhs = eta_tau * kl(eps_random, eps_mu) + (1 - eta_tau) * kl(eps_random, eps_pi) - eta_tau * kl(eps_pi_mu, eps_mu)

    logger.info(f"[Lemma 1] KL(π, π^μ) = {lhs.item():.6f}, RHS = {rhs.item():.6f}")
    return lhs.item() <= rhs.item() + 1e-6
def test_lemma2_diffusion(current_policy, reference_policy, prompt_batch, cfg):
    
    if torch.cuda.is_available():
        unet_device = torch.device("cuda")
        current_policy.pipe.unet = ensure_model_on_device(current_policy.pipe.unet, unet_device)
        reference_policy.pipe.unet = ensure_model_on_device(reference_policy.pipe.unet, unet_device)
    else:
        unet_device = next(current_policy.pipe.unet.parameters()).device
    
    # Ensure text_encoder is on the same device as UNet
    current_policy.pipe.text_encoder = ensure_model_on_device(current_policy.pipe.text_encoder, unet_device)
    reference_policy.pipe.text_encoder = ensure_model_on_device(reference_policy.pipe.text_encoder, unet_device)
    
    scheduler = current_policy.pipe.scheduler
    scheduler.set_timesteps(50)

    try:
        input_ids = current_policy.pipe.tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True
        ).input_ids.to(unet_device) 
        
        text_embeddings = current_policy.pipe.text_encoder(input_ids)[0]

        text_embeddings = text_embeddings.to(unet_device)

    except Exception as e:
        logger.error(f"Error in test_lemma2: {str(e)}")
        fallback_prompts = ["a photo of a cat"] * len(prompt_batch)
        input_ids = current_policy.pipe.tokenizer(
            fallback_prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True
        ).input_ids.to(unet_device) 
        
        text_embeddings = current_policy.pipe.text_encoder(input_ids)[0]
        text_embeddings = text_embeddings.to(unet_device)

    batch_size = len(prompt_batch)
    latents = torch.randn((batch_size, 4, 64, 64), device=unet_device)
    noise = torch.randn_like(latents, device=unet_device) 
    timesteps = torch.randint(
        low=0,
        high=scheduler.config.num_train_timesteps,
        size=(batch_size,),
        device=unet_device
    ).long()
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    with torch.autocast(device_type="cuda" if cfg.mixed_precision and torch.cuda.is_available() else "cpu",
                        enabled=cfg.mixed_precision):
        eps_current = current_policy.pipe.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
        eps_reference = reference_policy.pipe.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
        
        # Ensure consistent device for all tensors
        eps_current = eps_current.to(device=unet_device)
        eps_reference = eps_reference.to(device=unet_device)
        
        eps_mixture = (1 - cfg.beta) * eps_current + cfg.beta * eps_reference
        eps_random = current_policy.pipe.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
        eps_random = eps_random.to(device=unet_device)

    eps_current = eps_current.to(dtype=torch.float32)
    eps_reference = eps_reference.to(dtype=torch.float32)
    eps_mixture = eps_mixture.to(dtype=torch.float32)
    eps_random = eps_random.to(dtype=torch.float32)
    noise = noise.to(dtype=torch.float32)

    def kl(a, b):
        return 0.5 * ((a - b) ** 2).mean()

    KL_pi_t1 = kl(eps_random, eps_current)
    KL_pi_tmu = kl(eps_random, eps_mixture)

    score_pi = -F.mse_loss(eps_random, noise, reduction="none").mean(dim=[1, 2, 3])
    score_mixture = -F.mse_loss(eps_mixture, noise, reduction="none").mean(dim=[1, 2, 3])
    preference = torch.sigmoid(score_pi - score_mixture)

    preference_gap = ((eps_mixture - eps_random) ** 2).mean(dim=[1, 2, 3]) * preference
    preference_term = preference_gap.mean()

    rhs = KL_pi_tmu + cfg.beta * preference_term + 2 * cfg.beta ** 2

    logger.info(f"[Lemma 2] KL(π, π_t+1) = {KL_pi_t1.item():.6f}, RHS = {rhs.item():.6f}")
    return KL_pi_t1.item() <= rhs.item() + 1e-6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default=None, help="Path to folder containing images")
    parser.add_argument("--dataset_jsonl", type=str, default=None, help="Path to local JSONL dataset file")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to saved LoRA weights")
    parser.add_argument("--kl_mode", type=str, default="single", choices=["single", "multi"], help="KL regularization mode")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA for reference policy")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lora_r", type=int, default=16, help="Rank for LoRA adaptation")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps for gradient accumulation")
    parser.add_argument("--use_accelerate", action="store_true", help="Use Accelerate for multi-GPU training")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--loss_pref_current_weight", type=float, default=0.6, help="Weight for current policy preference loss")
    parser.add_argument("--loss_pref_alt_weight", type=float, default=0.2, help="Weight for alternative policy preference loss")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="diffusion-nlhf", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--final_model_path", type=str, default=None, help="Path to save final model")
    parser.add_argument("--lemma_log_path", type=str, default="lemma_verification.jsonl", help="Path to save lemma verification results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.final_model_path is None:
        final_model_dir = os.path.join("/scratch/jc11815", "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        args.final_model_path = os.path.join(final_model_dir, f"final_model_{timestamp}.pt")

    if args.use_wandb:
        try:
            import wandb
            run_name = args.wandb_run_name or f"run_{timestamp}"
            os.environ["WANDB_API_KEY"] = "bb90870b5a6b7eb37aaa386960b322190d582914"
            wandb.login(key=os.environ["WANDB_API_KEY"])
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
            logger.info(f"Initialized W&B run: {run_name}")
        except ImportError:
            logger.warning("W&B logging requested but wandb package is not installed.")
            args.use_wandb = False
        except Exception as e:
            logger.warning(f"Error initializing W&B: {str(e)}")
            args.use_wandb = False

    cfg = NLHFConfig(
        use_ema=args.use_ema,
        kl_mode=args.kl_mode,
        n_kl_steps=4,
        beta=beta,
        tau=tau,
        lora_r=args.lora_r,
        mixed_precision=args.mixed_precision,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        use_accelerate=args.use_accelerate and HAS_ACCELERATE,
        loss_pref_current_weight=args.loss_pref_current_weight,
        loss_pref_alt_weight=args.loss_pref_alt_weight
    )
    config_path = os.path.join(args.output_dir, f"config_{timestamp}.json")
    cfg.save(config_path)
    logger.info(f"Saved configuration to {config_path}")

    # === Lemma log ===
    lemma_log_path = args.lemma_log_path
    with open(lemma_log_path, 'w') as f:
        f.write("# Lemma Verification Results\n")

    # === ccelerator ===
    if cfg.use_accelerate:
        try:
            from accelerate import Accelerator
            accelerator = Accelerator(
                mixed_precision="fp16" if cfg.mixed_precision else "no",
                gradient_accumulation_steps=cfg.gradient_accumulation_steps
            )
            logger.info(f"Initialized Accelerator with {accelerator.num_processes} processes")
        except Exception as e:
            logger.error(f"Error initializing Accelerator: {str(e)}")
            accelerator = None
            cfg.use_accelerate = False
    else:
        accelerator = None

    # === Dataloader ===
    train_loader, val_loader, test_loader = load_hpdv2_dataloaders(
        batch_size=args.batch_size,
        dataset_jsonl=args.dataset_jsonl,
        image_folder=args.image_folder
    )

    # === Initialize / Resume Model ===
    global_step = 0
    if cfg.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {cfg.resume_from_checkpoint}")
        current_policy, global_step, optimizer, scheduler, ema = PolicyNetwork.from_checkpoint(
            cfg.resume_from_checkpoint
        )
    else:
        # ===  Policy ===
        logger.info("Initializing policy networks...")
        current_policy = PolicyNetwork(lora_path=args.lora_path, lora_r=args.lora_r)
        
        if torch.cuda.is_available():
            logger.info("Moving UNet to CUDA explicitly")
            current_policy.pipe.unet = current_policy.pipe.unet.to("cuda")
        
        current_policy.pipe.unet.enable_gradient_checkpointing()

        # If no accelerator, models to(cuda)
        if not accelerator and torch.cuda.is_available():
            logger.info("Moving entire policy to CUDA")
            current_policy = current_policy.to("cuda")
        for name, param in current_policy.pipe.unet.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False

        # === optimizer  ===
        trainable_params = []
        for module in current_policy.pipe.unet.attn_processors.values():
            if isinstance(module, LoRAAttnProcessor):
                trainable_params.extend(list(module.parameters()))
        logger.info(f"Training {sum(p.numel() for p in trainable_params)} parameters")

        optimizer = optim.AdamW(trainable_params, lr=lr)

        scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs, eta_min=lr * 0.1)

        # === EMA ===
        if cfg.use_ema:
            from torch_ema import ExponentialMovingAverage
            ema = ExponentialMovingAverage(trainable_params, decay=0.9995)  # Increased decay
        else:
            ema = None

    reference_policy = deepcopy(current_policy).eval()
    with torch.no_grad():
        diff_sum = 0.0
        for p1, p2 in zip(current_policy.parameters(), reference_policy.parameters()):
            diff_sum += (p1 - p2).abs().sum().item()


    # Freeze reference policy
    for param in reference_policy.parameters():
        param.requires_grad = False

    if not accelerator and torch.cuda.is_available():
        reference_policy = reference_policy.to("cuda")

    # === accelerator.prepare
    if accelerator:
        if torch.cuda.is_available():
            current_policy.pipe.text_encoder = ensure_model_on_device(current_policy.pipe.text_encoder, "cuda")
            current_policy.pipe.unet = ensure_model_on_device(current_policy.pipe.unet, "cuda")
            
            if not cfg.use_ema:
                reference_policy.pipe.text_encoder = ensure_model_on_device(reference_policy.pipe.text_encoder, "cuda")
                reference_policy.pipe.unet = ensure_model_on_device(reference_policy.pipe.unet, "cuda")
        current_policy, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            current_policy, optimizer, train_loader, val_loader, scheduler
        )
        
        if cfg.use_ema and ema is not None:
            logger.info("Synchronizing EMA parameters to reference_policy after Accelerator prepare.")
            with ema.average_parameters():
                for p_ref, p_cur in zip(reference_policy.parameters(), current_policy.parameters()):
                    p_ref.data.copy_(p_cur.data)
        else:
            logger.info("Synchronizing current_policy to reference_policy after Accelerator prepare.")
            for p_ref, p_cur in zip(reference_policy.parameters(), current_policy.parameters()):
                p_ref.data.copy_(p_cur.data)


        if torch.cuda.is_available():
            current_policy.pipe.text_encoder = ensure_model_on_device(current_policy.pipe.text_encoder, "cuda")
            current_policy.pipe.unet = ensure_model_on_device(current_policy.pipe.unet, "cuda")
            reference_policy.pipe.text_encoder = ensure_model_on_device(reference_policy.pipe.text_encoder, "cuda")
            reference_policy.pipe.unet = ensure_model_on_device(reference_policy.pipe.unet, "cuda")


    


    # === Training Loop ===
    logger.info("Starting training...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 3  
    early_stopping = False

    lemma_results = []
    
    for epoch in range(args.epochs):
        if early_stopping:
            logger.info("Early stopping triggered. Exiting training loop.")
            break

        current_policy.train()
        total_loss = 0.0
        epoch_steps = 0
        # Reset epoch metrics
        epoch_logs = {
            "loss_pref": 0.0,
            "loss_kl": 0.0,
            "loss_total": 0.0,
            "l2_reg": 0.0  
        }
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            accumulation_step = global_step % cfg.gradient_accumulation_steps
            loss, log = train_one_step(
                batch, current_policy, reference_policy, optimizer, ema, cfg,
                step=global_step, accumulation_step=accumulation_step, 
                accelerator=accelerator
            )

            total_loss += loss
            epoch_steps += 1
            for k, v in log.items():
                epoch_logs[k] += v

            if (accumulation_step + 1) % cfg.gradient_accumulation_steps == 0:
                global_step += 1
                scheduler.step()

                if args.use_wandb and step % 10 == 0:
                    wandb.log({
                        "train/loss": loss,
                        "train/loss_pref": log["loss_pref"],
                        "train/loss_kl": log["loss_kl"],
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/step": global_step,
                        "train/epoch": epoch,
                    })

                if global_step % 100 == 0:
                    with torch.no_grad():
                        prompt_batch = batch["prompt"]
                        lemma1_result = test_lemma1_diffusion(current_policy, reference_policy, prompt_batch, cfg)
                        lemma2_result = test_lemma2_diffusion(current_policy, reference_policy, prompt_batch, cfg)
                        lemma_data = {
                            "step": global_step,
                            "epoch": epoch,
                            "lemma1_verified": lemma1_result,
                            "lemma2_verified": lemma2_result,
                            "timestamp": datetime.now().isoformat()
                        }
                        lemma_results.append(lemma_data)
                        with open(lemma_log_path, 'a') as f:
                            f.write(json.dumps(lemma_data) + "\n")

                        if args.use_wandb:
                            wandb.log({
                                "lemma/lemma1_verified": lemma1_result,
                                "lemma/lemma2_verified": lemma2_result,
                                "lemma/step": global_step
                            })

        avg_loss = total_loss / epoch_steps
        for k in epoch_logs:
            epoch_logs[k] /= epoch_steps

        logger.info(f" Epoch {epoch} complete | Avg Loss = {avg_loss:.4f}")

        if args.use_wandb:
            wandb.log({
                "epoch/loss": avg_loss,
                "epoch/loss_pref": epoch_logs["loss_pref"],  # UPDATED
                "epoch/loss_kl": epoch_logs["loss_kl"],
                "epoch/epoch": epoch,
            })

        # === Evaluate on validation set ===
        avg_val_loss = evaluate(current_policy, reference_policy, val_loader, cfg)
        logger.info(f" Validation loss after Epoch {epoch}: {avg_val_loss:.4f}")

        if args.use_wandb:
            wandb.log({
                "val/loss": avg_val_loss,
                "val/epoch": epoch,
            })

        # === Early Stopping and Best Model Saving ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0

            best_model_path = os.path.join(args.output_dir, f"best_model_val{best_val_loss:.4f}")
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            current_policy.save_lora(best_model_path)
            logger.info(f" New best model saved to {best_model_path} with val_loss {best_val_loss:.4f}")

            if args.use_wandb:
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_model_path"] = best_model_path
                wandb.log({"event/best_model_saved": 1, "event/epoch": epoch})

        else:
            epochs_no_improve += 1
            logger.info(f"No improvement in val loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= early_stopping_patience:
                logger.info(f" Early stopping triggered after {epochs_no_improve} epochs without improvement.")
                early_stopping = True
                if args.use_wandb:
                    wandb.log({"event/early_stopped": 1, "event/epoch": epoch})
                break

        # === Save epoch checkpoint ===
        checkpoint_dir = current_policy.save_checkpoint(
            cfg.output_dir,
            global_step=global_step,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema
        )
        logger.info(f"Saved epoch checkpoint to {checkpoint_dir}")

        # === End of epoch ===
        avg_loss = total_loss / epoch_steps
        
        # Calculate epoch averages
        for k in epoch_logs:
            epoch_logs[k] /= epoch_steps
        
        logger.info(f" Epoch {epoch} complete | Avg Loss = {avg_loss:.4f}")

        
        
        # Log epoch metrics to W&B
        if args.use_wandb:
            wandb.log({
                "epoch/loss": avg_loss,
                "epoch/loss_pref": epoch_logs["loss_pref"],  # UPDATED
                "epoch/loss_kl": epoch_logs["loss_kl"],
                "epoch/epoch": epoch,
            })

        # === Early Stopping and Best Model Saving ===
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0

        # Save best model
        best_model_path = os.path.join(args.output_dir, f"best_model_val{best_val_loss:.4f}")
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        current_policy.save_lora(best_model_path)
        logger.info(f" New best model saved to {best_model_path} with val_loss {best_val_loss:.4f}")

        if args.use_wandb:
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_model_path"] = best_model_path
            wandb.log({"event/best_model_saved": 1, "event/epoch": epoch})

    else:
        epochs_no_improve += 1
        logger.info(f"No improvement in val loss for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
            early_stopping = True
            if args.use_wandb:
                wandb.log({"event/early_stopped": 1, "event/epoch": epoch})

        
        # Save epoch checkpoint
        checkpoint_dir = current_policy.save_checkpoint(
            cfg.output_dir, 
            global_step=global_step,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema
        )
        logger.info(f"Saved epoch checkpoint to {checkpoint_dir}")

    # === Save final model ===
    os.makedirs(os.path.dirname(args.final_model_path), exist_ok=True)
    current_policy.save_lora(args.final_model_path)
    logger.info(f"Saved final model to {args.final_model_path}")

    
    # Finalize W&B run
    if args.use_wandb:
        # Upload lemma verification results as artifact
        wandb.save(lemma_log_path)
        wandb.finish()
    
    logger.info("Training complete!")


# Define a custom function for sampling different noises during image generation
def sample_with_different_noise(policy_network, prompts, batch_size=2, seed1=None, seed2=None):
    """
    Generate a pair of images for each prompt using different noise seeds.
    
    Args:
        policy_network: The PolicyNetwork instance
        prompts: List of prompts
        batch_size: Batch size for processing
        seed1: Random seed for first image set (optional)
        seed2: Random seed for second image set (optional)
        
    Returns:
        List of paired images [(prompt, img1, img2), ...]
    """
    if seed1 is None:
        seed1 = random.randint(0, 999999)
    if seed2 is None:
        seed2 = random.randint(0, 999999)
        
    # Ensure seeds are different
    while seed2 == seed1:
        seed2 = random.randint(0, 999999)
    
    results = []
    
    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # Generate first set of images with seed1
        images1 = policy_network.forward(batch_prompts, seed=seed1)
        images2 = policy_network.forward(batch_prompts, seed=seed2)

        # Pair results
        for j, prompt in enumerate(batch_prompts):
            if j < len(images1) and j < len(images2):
                results.append((prompt, images1[j], images2[j]))
    
    return results


def generate_and_visualize_image_pairs(policy_network, prompts, output_dir, run_name=None):
    """
    Generate image pairs with different noise seeds and save them as a grid for comparison.
    Also log to wandb if available.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        from PIL import Image, ImageDraw, ImageFont
        import matplotlib.pyplot as plt
        import numpy as np
        
        # ✏️ 只取前5个 prompts
        prompts = prompts[:5]

        # Generate image pairs with different noise
        image_pairs = sample_with_different_noise(policy_network, prompts)
        
        # Create a grid of images (2 columns x N rows)
        n_pairs = len(image_pairs)
        fig, axes = plt.subplots(n_pairs, 2, figsize=(12, 5 * n_pairs))
        
        # Handle single row case
        if n_pairs == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for i, (prompt, img1, img2) in enumerate(image_pairs):
            axes[i][0].imshow(np.array(img1))
            axes[i][0].set_title(f"Seed 1: {prompt[:50]}")
            axes[i][0].axis('off')
            
            axes[i][1].imshow(np.array(img2))
            axes[i][1].set_title(f"Seed 2: {prompt[:50]}")
            axes[i][1].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_pairs_{run_name or 'test'}_{timestamp}.png"
        fig_path = os.path.join(output_dir, filename)
        plt.savefig(fig_path)
        plt.close()
        
        logger.info(f"Saved image pairs visualization to {fig_path}")
        
        # Log to wandb if available
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"image_pairs": wandb.Image(fig_path)})
        except:
            pass
            
        return fig_path

    except Exception as e:
        logger.error(f"Error generating image pairs: {str(e)}")
        return None



if __name__ == "__main__":
    import sys
    sys.argv = [
        "diffusion_nlhf.py",
        "--image_folder", "selected_images_path",
        "--dataset_jsonl", "dataset_path",
        "--epochs", "10",
        "--batch_size", "2",
        "--gradient_accumulation_steps", "1",
        "--lora_r", "4",
        "--mixed_precision",
        "--kl_mode", "single",
        "--use_ema",
        "--output_dir", "output_path",
        "--final_model_path", "final_model_save_path",
        "--save_steps", "1000",
        "--use_accelerate",
        "--use_wandb",
        "--wandb_project", "diffusion-nlhf",
        "--wandb_run_name", "diffusion_nlhf_test_run",
        
    ]
    main()
