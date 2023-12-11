from diffusers import DiffusionPipeline
import torch

# CompVis/stable-diffusion-v1-4
# stabilityai/stable-diffusion-xl-base-1.0
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda:1")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "An apple riding a green horse"

images = pipe(prompt=prompt).images[0]
images.save("/home/kasra/kasra_dat_file/diffusion_data/generated_image.png")