# Import packages
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import wandb
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, get_linear_schedule_with_warmup
from accelerate import Accelerator

# Environment variables 
os.environ["WANDB_API_KEY"] = "wanddb_api"
SCRATCH_DIR = "/scratch/rl4789/cjj"
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

# Reward Model Head: MLP taking concatenated image and text embeddings to predict reward 
class RewardModelHead(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.dense1 = nn.Linear(embedding_dim * 2, 256)
        self.dense2 = nn.Linear(256, 64)
        self.dense3 = nn.Linear(64, 1)
        
    def forward(self, image_embeds, text_embeds):
        combined = torch.cat([image_embeds, text_embeds], dim=1)
        x = F.relu(self.dense1(combined))
        x = F.relu(self.dense2(x))
        reward = self.dense3(x)
        return reward.squeeze(-1)

# Full Reward Model: frozen CLIP + trainable reward head 
class CLIPRewardModel(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.reward_head = RewardModelHead(embedding_dim=self.clip.config.projection_dim)
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.eval()
        
    def forward(self, image_inputs, text_inputs):
        with torch.no_grad():
            image_features = self.clip.get_image_features(pixel_values=image_inputs["pixel_values"])
            text_features = self.clip.get_text_features(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"]
            )
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return self.reward_head(image_features, text_features)

# Custom Dataset class that loads human preference annotations and paired images
class PreferenceDataset(Dataset):
    def __init__(self, preference_data, image_dir):
        self.image_dir = image_dir
        self.data = preference_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        paths = item["image_path"]
        pref_idx, dispref_idx = item["human_preference"]

        preferred_img = Image.open(os.path.join(self.image_dir, paths[pref_idx])).convert("RGB")
        dispreferred_img = Image.open(os.path.join(self.image_dir, paths[dispref_idx])).convert("RGB")

        return {
            "prompt": prompt,
            "preferred_img": preferred_img,
            "dispreferred_img": dispreferred_img
        }

# Collate function that processes batch of prompts and images into CLIP inputs
def collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    preferred_imgs = [item["preferred_img"] for item in batch]
    dispreferred_imgs = [item["dispreferred_img"] for item in batch]

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    preferred_inputs = processor(text=prompts, images=preferred_imgs, return_tensors="pt", padding=True, truncation=True)
    dispreferred_inputs = processor(text=prompts, images=dispreferred_imgs, return_tensors="pt", padding=True, truncation=True)

    return {
        "prompts": prompts,
        "preferred_img_inputs": preferred_inputs,
        "dispreferred_img_inputs": dispreferred_inputs
    }

# Load and split JSONL file into training and validation data 
def prepare_preference_data(input_jsonl, train_ratio=0.9):
    with open(input_jsonl, 'r') as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} preference samples from {input_jsonl}")
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]

# Trainer class for training reward model with pairwise Bradley-Terry loss 
class RewardModelTrainer:
    def __init__(self, config):
        self.config = config
        self.accelerator = Accelerator(
            mixed_precision=None,
            gradient_accumulation_steps=1,
            log_with="wandb" if self.config.use_wandb else None,
        )
        if self.accelerator.is_main_process and self.config.use_wandb:
            wandb.init(project=self.config.wandb_project_name, name=self.config.wandb_run_name)

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        self.train_data, self.eval_data = prepare_preference_data(self.config.input_jsonl)
        self._setup_model()
        self._setup_dataset()
        self.model, self.optimizer, self.scheduler, self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, self.train_dataloader, self.eval_dataloader
        )

    # Initialize model and optimizer
    def _setup_model(self):
        print("Initializing model...")
        self.model = CLIPRewardModel(self.config.clip_model_name)
        self.model.reward_head.train()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.config.learning_rate, weight_decay=0.01)
    
    # Load datasets and initialize dataloaders and LR scheduler 
    def _setup_dataset(self):
        print("Loading datasets...")
        train_dataset = PreferenceDataset(self.train_data, self.config.image_dir)
        eval_dataset = PreferenceDataset(self.eval_data, self.config.image_dir)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
        total_steps = len(self.train_dataloader) * self.config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # Training loop 
    def train(self):
        print(f"Training for {self.config.num_epochs} epochs...")
        global_step = 0
        best_accuracy = 0.0

        for epoch in range(self.config.num_epochs):
            self.model.train()
            train_loss = 0.0
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                with self.accelerator.accumulate(self.model):
                    r_pref = self.model(batch["preferred_img_inputs"], {"input_ids": batch["preferred_img_inputs"]["input_ids"], "attention_mask": batch["preferred_img_inputs"]["attention_mask"]})
                    r_dispref = self.model(batch["dispreferred_img_inputs"], {"input_ids": batch["dispreferred_img_inputs"]["input_ids"], "attention_mask": batch["dispreferred_img_inputs"]["attention_mask"]})
                    loss = -F.logsigmoid(r_pref - r_dispref).mean()  # Pairwise loss: encourages higher reward for preferred image
                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    train_loss += loss.item()
                    global_step += 1
                    if global_step % self.config.logging_steps == 0 and self.config.use_wandb:
                        wandb.log({"train_loss": loss.item(), "step": global_step})

            eval_metrics = self.evaluate()
            if eval_metrics["accuracy"] > best_accuracy:
                best_accuracy = eval_metrics["accuracy"]
                self.save_model("best")
            self.save_model(f"epoch-{epoch}")
            if self.config.use_wandb:
                wandb.log({"epoch": epoch, "train_loss": train_loss / len(self.train_dataloader), **eval_metrics})
        
        self.save_model("final")
        if self.config.use_wandb:
            wandb.finish()

    # Evaluation using accuracy and eval loss
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader):
                r_pref = self.model(batch["preferred_img_inputs"], {"input_ids": batch["preferred_img_inputs"]["input_ids"], "attention_mask": batch["preferred_img_inputs"]["attention_mask"]})
                r_dispref = self.model(batch["dispreferred_img_inputs"], {"input_ids": batch["dispreferred_img_inputs"]["input_ids"], "attention_mask": batch["dispreferred_img_inputs"]["attention_mask"]})
                loss = -F.logsigmoid(r_pref - r_dispref).mean()
                loss_sum += loss.item()
                correct += (r_pref > r_dispref).sum().item()
                total += r_pref.size(0)
        return {"accuracy": correct / total, "eval_loss": loss_sum / len(self.eval_dataloader)}

    # Save model components and config 
    def save_model(self, tag):
        if not self.accelerator.is_local_main_process:
            return
        out_dir = os.path.join(self.config.output_dir, f"reward-model-{tag}")
        os.makedirs(out_dir, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.clip.save_pretrained(os.path.join(out_dir, "clip"))
        torch.save(unwrapped.reward_head.state_dict(), os.path.join(out_dir, "reward_head.pt"))
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(vars(self.config), f, indent=2)

# Configuration object 
class Args:
    input_jsonl = "dataset_path"
    image_dir = "selected_images_path"
    output_dir = "reward_model_path"
    clip_model_name = "openai/clip-vit-base-patch32"
    batch_size = 4
    num_epochs = 5
    learning_rate = 2e-5
    logging_steps = 10
    use_wandb = True
    wandb_project_name = "sd-reward-model"
    wandb_run_name = "clip-reward-model-training"
    seed = 42

if __name__ == "__main__":
    args = Args()
    os.makedirs(args.output_dir, exist_ok=True)
    trainer = RewardModelTrainer(args)
    trainer.train()
