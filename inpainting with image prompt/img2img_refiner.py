import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image

from ip_adapter import IPAdapter

base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "/home/kasra/AI_projects_/ComfyUI/models/clip_vision/sd_1.5/"
ip_ckpt = "/home/kasra/AI_projects_/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus/models/ip-adapter_sd15.bin"
device = "cuda:1"


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)
# read image prompt
image = Image.open("/home/kasra/kasra_dat_file/diffusion_data/generated_image2")
g_image = Image.open("/home/kasra/kasra_dat_file/diffusion_data/style.jpg")
image_grid([image.resize((256, 256)), g_image.resize((256, 256))], 1, 2)
# load ip-adapter
ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
images = ip_model.generate(pil_image=image, num_samples=1, num_inference_steps=50, seed=42, image=g_image, strength=0.6)
grid = image_grid(images, 1, 1)
grid.save("/home/kasra/kasra_dat_file/diffusion_data/generated_image4.png")
print('before refinement')
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()
init_image = images[0]

prompt = "in detail image of the person in image fed to the model and same background scene"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image, strength=0.5).images[0]
grid = make_image_grid([init_image, image], rows=1, cols=2)
grid.save("/home/kasra/kasra_dat_file/diffusion_data/generated_image5.png")
print('image has been saved')
