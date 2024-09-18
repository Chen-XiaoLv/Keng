import argparse
from torchvision import datasets
from SD_Generate import  *
from utils import *


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate an augmented dataset from original images and fractal patterns.")
    parser.add_argument('--train_dir', type=str, required=True, help='Path to the directory containing the original training images.')
    parser.add_argument('--fractal_dir', type=str, required=True, help='Path to the directory containing the fractal images.')
    parser.add_argument('--prompts', type=str, required=True, help='Comma-separated list of prompts for image generation.')
    return parser.parse_args()

def main():

    args = parse_arguments()
    prompts = args.prompts.split(',')  # This will give you a list of prompts

    # Initialize the model
    model_id = "timbrooks/instruct-pix2pix"
    model_initialization = ModelHandler(model_id=model_id, device='cuda')

    # Load the original dataset
    train_dataset = datasets.ImageFolder(root=args.train_dir)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    # Load fractal images
    fractal_imgs =load_fractal_images(args.fractal_dir)

    # Create the augmented dataset
    aug = DiffuseMix(
        original_dataset=train_dataset,
        fractal_imgs=fractal_imgs,
        num_images=1,
        guidance_scale=4,
        idx_to_class = idx_to_class,
        prompts=prompts,
        model_handler=model_initialization
    )

    aug.start()
if __name__ == '__main__':
    main()