import numpy as np
import random
from PIL import Image

def combine_images(original_img, augmented_img, blend_width=20):
    width, height = 640, 640
    combine_choice = random.choice(['horizontal', 'vertical',])

    if combine_choice == 'randomMix':
        data = [i for i in range(640 * 640)]
        idx = random.sample(data, int(640 * 640 * 0.5))
        for k in idx:
            i, j = k // 640, k % 640
            original_img[i][j] = augmented_img[i][j]
            return original_img

    elif combine_choice == 'vertical':  # Vertical combination
        mask = np.linspace(0, 1, blend_width).reshape(-1, 1)
        mask = np.tile(mask, (1, width))  # Extend mask horizontally
        mask = np.vstack([np.zeros((height // 2 - blend_width // 2, width)), mask,
                          np.ones((height // 2 - blend_width // 2 + blend_width % 2, width))])
        mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))

    elif combine_choice == 'horizontal':
        mask = np.linspace(0, 1, blend_width).reshape(1, -1)
        mask = np.tile(mask, (height, 1))  # Extend mask vertically
        mask = np.hstack([np.zeros((height, width // 2 - blend_width // 2)), mask,
                          np.ones((height, width // 2 - blend_width // 2 + blend_width % 2))])
        mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))
    elif combine_choice == 'diag1':
        mask = np.ones((width, height))
        mask[:width // 2, :height // 2] = 0
        mask[width // 2:, height // 2:] = 0
        mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))
    else:
        mask = np.zeros((width, height))
        mask[:width // 2, :height // 2] = 1
        mask[width // 2:, height // 2:] = 1
        mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))

    original_array = np.array(original_img, dtype=np.float32) / 255.0
    augmented_array = np.array(augmented_img, dtype=np.float32) / 255.0

    blended_array = (1 - mask) * original_array + mask * augmented_array
    blended_array = np.clip(blended_array * 255, 0, 255).astype(np.uint8)

    return  Image.fromarray(blended_array)