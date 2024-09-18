import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from accelerate import Accelerator
import os
from torch.utils.data import Dataset
import random
from utils import IMAGESIZE


IMGSIZE=IMAGESIZE

class ModelHandler:
    def __init__(self, model_id, device):
        self.accelerator = Accelerator()
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None
        ).to(device)
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipeline.scheduler.config)

    def generate_images(self, prompt, img_path, num_images, guidance_scale):
        image = Image.open(img_path).convert('RGB').resize((IMGSIZE, IMGSIZE))
        return self.pipeline(prompt, image=image, num_images_per_prompt=num_images, guidance_scale=guidance_scale).images

class DiffuseMix(Dataset):
    def __init__(self, original_dataset, num_images, guidance_scale, fractal_imgs, idx_to_class, prompts, model_handler):
        self.original_dataset = original_dataset
        self.idx_to_class = idx_to_class
        self.combine_counter = 0
        self.fractal_imgs = fractal_imgs
        self.prompts = prompts
        self.model_handler = model_handler
        self.num_augmented_images_per_image = num_images
        self.guidance_scale = guidance_scale

    def start(self):
        self.generate_augmented_images()

    def generate_augmented_images(self):
        augmented_data = []
        base_directory = './result'
        original_resized_dir = os.path.join(base_directory, 'original_resized')
        generated_dir = os.path.join(base_directory, 'generated')
        fractal_dir = os.path.join(base_directory, 'fractal')

        # Ensure these directories exist
        os.makedirs(original_resized_dir, exist_ok=True)
        os.makedirs(generated_dir, exist_ok=True)
        os.makedirs(fractal_dir, exist_ok=True)

        for idx, (img_path, label_idx) in enumerate(self.original_dataset.samples):

            label = self.idx_to_class[label_idx]

            original_img = Image.open(img_path).convert('RGB')
            original_img = original_img.resize((IMGSIZE, IMGSIZE))
            img_filename = os.path.basename(img_path)

            label_dirs = {dtype: os.path.join(base_directory, dtype, str(label)) for dtype in
                          ['original_resized', 'generated', 'fractal']}

            for dir_path in label_dirs.values():
                os.makedirs(dir_path, exist_ok=True)

            original_img.save(os.path.join(label_dirs['original_resized'], img_filename))

            for prompt in self.prompts:
                augmented_images =  self.model_handler.generate_images(prompt, img_path, self.num_augmented_images_per_image,
                                                          self.guidance_scale)

                for i, img in enumerate(augmented_images):
                    img = img.resize((IMGSIZE, IMGSIZE))
                    while 1:
                        generated_img_filename = f"{img_filename}_generated_{prompt}_{i}.jpg"
                        if not self.utils.is_black_image(generated_img_filename):
                            img.save(os.path.join(label_dirs['generated'], generated_img_filename))
                            random_fractal_img = random.choice(self.fractal_imgs)
                            fractal_img_filename = f"{img_filename}_fractal_{prompt}_{i}.jpg"
                            random_fractal_img.save(os.path.join(label_dirs['fractal'], fractal_img_filename))
                            break


    def __len__(self):
        return len(self.augmented_images)

    def __getitem__(self, idx):
        return None