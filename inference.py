import os
import json
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm  

# ==== config ====
adapter_path = "checkpoint_adapter_path"  
json_path = "test_prompt path"   
output_dir = "save_path"        
os.makedirs(output_dir, exist_ok=True)

# ==== base model ====
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,   
    safety_checker=None
).to("cuda")

# ==== LoRA Adapter ====
pipe.load_lora_weights(adapter_path)

# ====  JSON  ====
with open(json_path, "r") as f:
    data = json.load(f)


pipe.enable_attention_slicing()  

for i, item in enumerate(tqdm(data)):
    prompt = item["prompt"] 

    with torch.no_grad():
        image = pipe(prompt).images[0]

    
    image_path = os.path.join(output_dir, f"sample_{i:03}.png")
    image.save(image_path)

print(f" Saved to: {output_dir}")
