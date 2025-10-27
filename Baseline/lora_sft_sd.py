# Import Package
import os
import json
import datetime
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
from peft import LoraConfig
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from torch.optim.lr_scheduler import LambdaLR

# Environment Variables
os.environ["HF_HOME"] = "hf_home here "
os.environ["TRANSFORMERS_CACHE"] = "/scratch/xc2615/hf_cache"
os.environ["WANDB_API_KEY"] = "wandb_api"
base_checkpoint_dir = "saved_checkpoints_path"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_dir = os.path.join(base_checkpoint_dir, f"run_{timestamp}")
os.makedirs(checkpoint_dir, exist_ok=True)

# Define a dataset for paired images, prompts and human preference labels
class PairwiseDataset(Dataset):
    """
    Loads paired images and prompts along with human preference labels
    for supervised preference fine-tuning.
    
    Args:
        json_path (str): Path to the JSONL file containing prompt and image pair information.
        image_dir (str): Directory where image files are stored.
        tokenizer (Tokenizer): Tokenizer for processing text prompts.
        image_processor (callable): Image transformation function (e.g., resizing, normalization).
        max_samples (int, optional): Maximum number of samples to load. Defaults to 3000.
    """
    def __init__(self, json_path, image_dir, tokenizer, image_processor, max_samples=3000):
        # Initialize dataset by loading sample information from JSONL file
        self.samples = []
        with open(json_path, 'r') as f:
            for idx, line in enumerate(f):
                if idx >= max_samples:
                    break
                data = json.loads(line)
                self.samples.append({
                    "prompt": data["prompt"],
                    "img0": os.path.join(image_dir, data["image_path"][0]),
                    "img1": os.path.join(image_dir, data["image_path"][1]),
                    "label": data["human_preference"]
                })
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.samples)

    def __getitem__(self, idx):
        # Fetch and process a single sample: two images, a prompt, and a label
        sample = self.samples[idx]
        prompt = sample["prompt"]
        raw_label = sample["label"]
        label = int(raw_label[1]) if isinstance(raw_label, (list, tuple)) else int(raw_label)
        label = torch.clamp(torch.tensor(label, dtype=torch.float), 0, 1)

        # Try loading and preprocessing images
        try:
            img0 = self.image_processor(Image.open(sample["img0"]).convert("RGB"))
        except Exception:
            img0 = torch.zeros(3, 512, 512)

        try:
            img1 = self.image_processor(Image.open(sample["img1"]).convert("RGB"))
        except Exception:
            img1 = torch.zeros(3, 512, 512)

        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)

        return {
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "img0": img0,
            "img1": img1,
            "label": label
        }

# Utility function to save model, optimizer, and scheduler states
def save_checkpoint(pipe, optimizer, scheduler, epoch, best_val_loss, global_step, filename):
    """
    Save model checkpoint including LoRA adapter weights, optimizer state, and scheduler state.
    
    Args:
        pipe (StableDiffusionPipeline): The diffusion model pipeline containing LoRA modules.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        epoch (int): Current epoch number.
        best_val_loss (float): Best validation loss achieved so far.
        global_step (int): Current global training step.
        filename (str): Filename to save the checkpoint.
    
    Returns:
        None
    """
    save_path = os.path.join(checkpoint_dir, filename)
    checkpoint = {
        "unet": {k: v.state_dict() for k, v in pipe.unet.attn_processors.items() if isinstance(v, LoRAAttnProcessor)},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "global_step": global_step
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at {save_path}")

# Utility function to load the latest model, optimizer, and scheduler states
def load_checkpoint(pipe, optimizer, scheduler):
    """
    Load the latest available checkpoint and restore model, optimizer, and scheduler states.
    
    Args:
        pipe (StableDiffusionPipeline): The diffusion model pipeline containing LoRA modules.
        optimizer (torch.optim.Optimizer): The optimizer to be resumed.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to be resumed.
    
    Returns:
        tuple:
            int: The next epoch number to start from.
            float: The best validation loss recorded.
            int: The current global training step.
    """
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")])
    if not checkpoints:
        return 0, float('inf'), 0
    latest = os.path.join(checkpoint_dir, checkpoints[-1])
    checkpoint = torch.load(latest)
    for name, module in pipe.unet.attn_processors.items():
        if isinstance(module, LoRAAttnProcessor) and name in checkpoint["unet"]:
            module.load_state_dict(checkpoint["unet"][name])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    print(f"Resumed from {latest}")
    return checkpoint["epoch"] + 1, checkpoint["best_val_loss"], checkpoint["global_step"]

# Define the main training loop for supervised fine-tuning the diffusion model with LoRA adapters
def train():
    """
    Fine-tune the UNet of a Stable Diffusion model using LoRA adapters and human preference data.

    This function handles:
        - Loading the pretrained model and scheduler
        - Adding and freezing LoRA modules
        - Setting up dataset, dataloaders, optimizer, and scheduler
        - Running the training and validation loops
        - Logging and saving checkpoints

    Args:
        None

    Returns:
        None
    """
    accelerator = Accelerator()
    wandb.init(project="lora-unet-preference-finetune", name="run-sd-v1.5")
    device = accelerator.device

    # Load the pretrained Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        use_auth_token=os.environ["HF_TOKEN"],
        cache_dir="/scratch/xc2615/hf_cache"
    ).to(device)

    # Set up the noise scheduler
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Add LoRA adapters for fine-tuning
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        init_lora_weights="gaussian"
    )
    pipe.unet.add_adapter(lora_config)

    # Freeze non-LoRA parameters 
    for name, param in pipe.unet.named_parameters():
        if "lora" not in name:
            param.requires_grad = False

    # Prepare tokenizer and image preprocessing pipeline
    tokenizer = pipe.tokenizer
    image_processor = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load the dataset of paired images and prompts
    dataset = PairwiseDataset(
        json_path="/scratch/xc2615/nlhf/selected_package/selected_samples.jsonl",
        image_dir="/scratch/xc2615/nlhf/selected_package/selected_images",
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_samples=3000
    )

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=True)

    # Set up optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, pipe.unet.parameters()), lr=2e-5)

    def lr_lambda(current_step):
        warmup_steps = 200
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler()

    # Resume training if a checkpoint exists
    start_epoch, best_val_loss, global_step = load_checkpoint(pipe, optimizer, scheduler)
    patience = 5
    counter = 0

    # Training loop
    for epoch in range(start_epoch, 50):
        pipe.unet.train()
        total_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch} [Train]"):
            optimizer.zero_grad()
            img0, img1, label = batch["img0"].to(device), batch["img1"].to(device), batch["label"].to(device)

            with torch.cuda.amp.autocast():
                # Encode images into latent space
                with torch.no_grad():
                    latents0 = pipe.vae.encode(img0).latent_dist.sample() * 0.18215
                    latents1 = pipe.vae.encode(img1).latent_dist.sample() * 0.18215

                # Add noise to the latents
                noise0, noise1 = torch.randn_like(latents0), torch.randn_like(latents1)
                timesteps = torch.randint(0, 1000, (latents0.shape[0],), device=device).long()

                noisy_latents0 = pipe.scheduler.add_noise(latents0, noise0, timesteps)
                noisy_latents1 = pipe.scheduler.add_noise(latents1, noise1, timesteps)

                # Encode prompts
                encoder_hidden_states = pipe.text_encoder(
                    batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device)
                )[0]

                # Predict noise
                noise_pred0 = torch.nan_to_num(pipe.unet(noisy_latents0, timesteps, encoder_hidden_states).sample)
                noise_pred1 = torch.nan_to_num(pipe.unet(noisy_latents1, timesteps, encoder_hidden_states).sample)

                # Compute preference loss
                loss0 = F.mse_loss(noise_pred0, noise0, reduction="none").mean(dim=[1,2,3])
                loss1 = F.mse_loss(noise_pred1, noise1, reduction="none").mean(dim=[1,2,3])

                preference = 2 * (label - 0.5)
                logits = (loss0 - loss1) * preference
                loss = F.softplus(logits).mean()

            # Backpropagation with mixed precision
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            wandb.log({"train/loss": loss.item()}, step=global_step)
            global_step += 1

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch} - Avg Train Loss: {avg_loss:.6f}")

        # Validation loop
        pipe.unet.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch} [Validation]"):
                img0, img1, label = batch["img0"].to(device), batch["img1"].to(device), batch["label"].to(device)

                latents0 = pipe.vae.encode(img0).latent_dist.sample() * 0.18215
                latents1 = pipe.vae.encode(img1).latent_dist.sample() * 0.18215

                noise0, noise1 = torch.randn_like(latents0), torch.randn_like(latents1)
                timesteps = torch.randint(0, 1000, (latents0.shape[0],), device=device).long()

                noisy_latents0 = pipe.scheduler.add_noise(latents0, noise0, timesteps)
                noisy_latents1 = pipe.scheduler.add_noise(latents1, noise1, timesteps)

                encoder_hidden_states = pipe.text_encoder(
                    batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device)
                )[0]

                noise_pred0 = torch.nan_to_num(pipe.unet(noisy_latents0, timesteps, encoder_hidden_states).sample)
                noise_pred1 = torch.nan_to_num(pipe.unet(noisy_latents1, timesteps, encoder_hidden_states).sample)

                loss0 = F.mse_loss(noise_pred0, noise0, reduction="none").mean(dim=[1,2,3])
                loss1 = F.mse_loss(noise_pred1, noise1, reduction="none").mean(dim=[1,2,3])

                preference = 2 * (label - 0.5)
                logits = (loss0 - loss1) * preference
                loss = F.softplus(logits).mean()

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch} - Avg Val Loss: {avg_val_loss:.6f}")
        wandb.log({"epoch/loss": avg_loss, "epoch/val_loss": avg_val_loss}, step=global_step)

        # Save checkpoint after each epoch
        save_checkpoint(pipe, optimizer, scheduler, epoch, best_val_loss, global_step, "last_checkpoint.pth")

        # Update best checkpoint if validation improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(pipe, optimizer, scheduler, epoch, best_val_loss, global_step, "best_checkpoint.pth")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    wandb.finish()

# Run the training procedure
if __name__ == "__main__":
    train()
