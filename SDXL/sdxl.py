from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
import torch

# Initialize CLIP model for text encoding
clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# Encode text prompts
text_inputs = ["a bear drinking beer"]
inputs = clip_tokenizer(text_inputs, padding=True, return_tensors="pt")
text_embeddings = clip_model(**inputs)

# Load the generative model (e.g., Stable Diffusion)
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = StableDiffusionPipeline.from_pretrained(model_id)
pipeline = pipeline.to(device)

# Generate an image
generator = torch.manual_seed(0)  # for reproducibility
latent_image = pipeline(text_embeddings, num_inference_steps=50, generator=generator)["sample"]

# Save the generated image
latent_image.save("generated_image.png")
