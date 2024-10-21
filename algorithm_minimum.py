import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import random

# diffusers
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline, 
    DPMSolverMultistepScheduler,
    StableDiffusionLatentUpscalePipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
)

from diffusers.utils import load_image
from controlnet_aux import PidiNetDetector

def make_inpaint_condition(image, mask):
    image = image / 255.0
    mask = mask / 255.0
    assert image.shape[0:1] == mask.shape[0:1], "image and image_mask must have the same image size"
    image[mask > 0.5] = -1.0  # set as masked pixel
    # image[mask <= 0.5] = 1
    return image

def np_array_to_pil(image):
    return Image.fromarray(np.uint8(image))

class Algorithm:
    def __init__(self) -> None:
        self.controlnets = []
        self.controlnets.append(ControlNetModel.from_pretrained("alimama-creative/EcomXL_controlnet_inpaint", torch_dtype=torch.float16, use_safetensors=True))
        self.controlnets.append(ControlNetModel.from_pretrained("alimama-creative/EcomXL_controlnet_softedge", torch_dtype=torch.float16, use_safetensors=True))
        
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "SG161222/RealVisXL_V5.0", 
            controlnet=self.controlnets, 
            safety_checker=None,
            torch_dtype=torch.float16
        ).to('cuda')
        self.pipe.enable_vae_slicing()
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, algorithm_type='sde-dpmsolver++', use_karras_sigmas=True)

        self.edge_processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')

        self.lora_dir = '/SD/SDXL_minimum_pipeline/'
        self.lora_name = 'kitchen.safetensors'
        self.lora_scale = 0.6
        self.trigger_words = 'kitchen'
        self.pipe.load_lora_weights(self.lora_dir, weight_name=self.lora_name)
        self.pipe.fuse_lora(lora_scale=self.lora_scale)


algo = Algorithm()
image = load_image(
    "https://huggingface.co/alimama-creative/EcomXL_controlnet_inpaint/resolve/main/images/inp_0.png"
)
mask = load_image(
    "https://huggingface.co/alimama-creative/EcomXL_controlnet_inpaint/resolve/main/images/inp_1.png"
)

inpaint_control_image = make_inpaint_condition(np.array(image), 255-np.array(mask))

edge_image = algo.edge_processor(np_array_to_pil(image), safe=False)

inpaint_control_image_tensor = torch.from_numpy(np.expand_dims(inpaint_control_image, 0).transpose(0, 3, 1, 2))
edge_image_tensor = torch.from_numpy(np.expand_dims(np.array(edge_image.convert("RGB")).astype(np.float32) / 255.0, 0).transpose(0, 3, 1, 2))

import pdb;pdb.set_trace()
generator = torch.manual_seed(random.randint(0,31337))
prompt = algo.trigger_words+", a product on a kitchen table, light-colored clean background, indoor lights from top left and front, bright beige tones, modern minimalist style, high detail, realistic, best quality, real picture, intricate details, Interior Photography, fujifilm xt3, raw photo, 8k uhd, film grain, unreal engine 5, ray tracing"
negative_prompt = "blurry, low details, low quality, lowres, worst quality, top view, float things, extra connection, adjunct, appendages, stand, bracket, bad anatomy, text, font, watermark, signature, logo, letters, word, digits, grid, brown, grey, bubble, oversaturated, undersaturated, overexposed, underexposed, grayscale, stripe, spot, wheel, ugly, bad hands, bad anatomy, cropped, baby, body, human, brand, bad face, airbrushed, cartoon, anime, semi-realistic, cgi, render, 3D"

bg_imgs = algo.pipe(
    negative_prompt=negative_prompt,
    prompt=prompt,
    image=[inpaint_control_image_tensor,edge_image_tensor],
    generator=generator,
    num_images_per_prompt=4,
    num_inference_steps=20,
    guidance_scale=7,
    height=1024,
    width=1024,
    controlnet_conditioning_scale=[0.5, 0.6]).images

for idx, bg_img in enumerate(bg_imgs):
    bg_img.save('result_{}.png'.format(idx))
