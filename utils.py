from PIL import Image
import numpy as np
import os
from ImageMix import multiply

IMAGESIZE=640

def is_black_image(image):
    histogram = image.convert("L").histogram()
    return histogram[-1] > 0.9 * image.size[0] * image.size[1] and max(histogram[:-1]) < 0.1 * image.size[0] * \
        image.size[1]

def imageAdd(base_img, overlay_img, alpha=0.20):

    overlay_img_resized = overlay_img.resize(base_img.size)
    base_array = np.array(base_img, dtype=np.float32)
    overlay_array = np.array(overlay_img_resized, dtype=np.float32)

    blended_array = (1 - alpha) * base_array + alpha * overlay_array
    blended_array = np.clip(blended_array, 0, 255).astype(np.uint8)
    blended_img = Image.fromarray(blended_array)
    return blended_img

def ImageListAdd(imglist, weight=None):

    n=[1/len(imglist)]*len(imglist) if not weight else weight
    base_array=np.array(imglist[0],dtype=np.float32)*n[0]
    for idx,i in enumerate(imglist[1:]):
        i=np.array(i,dtype=np.float32)
        base_array+=n[idx+1]*i
    blended_array = np.clip(base_array, 0, 255).astype(np.uint8)
    blended_img = Image.fromarray(blended_array)
    return blended_img
def imageMultiply(base,mul,beta=4):
    base,mul=np.array(base,dtype=np.float32)/255,np.array(mul,dtype=np.float32)/255
    out=np.clip((multiply(base,mul,beta))*255+0.5,0,255).astype(np.uint8)
    return Image.fromarray(out)


def load_fractal_images(fractal_img_dir):
    fractal_img_paths = [os.path.join(fractal_img_dir, fname) for fname in os.listdir(fractal_img_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]
    return [Image.open(path).convert('RGB').resize((IMAGESIZE, IMAGESIZE)) for path in fractal_img_paths]
