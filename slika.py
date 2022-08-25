import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
# devise = "cuda"
device = "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, revision="fp16", use_auth_token=True)
pipe = pipe.to(device)

prompt = "a photo of an astronaut riding a horse on mars"
with autocast(device_type=device, dtype=torch.bfloat16):
    image = pipe(prompt, guidance_scale=7.5)["sample"][0]

image.save("astronaut_rides_horse.png")
