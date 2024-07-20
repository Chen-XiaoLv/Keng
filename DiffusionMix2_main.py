import cv2 as cv
from ImageAug import *
from ImageMix import *
import os
from tqdm import tqdm
from combine import combine_images
from utils import *

diffusionImage=r"C:\Users\Administrator\Downloads\result2.0\result\generated\class0"
fractralImage=r"C:\Users\Administrator\Downloads\result2.0\result\fractal\class0"
OriImage=r"C:\Users\Administrator\Downloads\result2.0\result\original_resized\class0"
savePath=r"C:\Users\Administrator\Desktop\save"

def augment_input(image):
    aug_list=augmentations_
    op=np.random.choice(aug_list)
    return np.asarray(op(Image.fromarray(np.uint8((image))).copy(),1))

def getFile(path):
    return cv.imread(path)

if __name__ == '__main__':

    mixing=mixings
    os.makedirs(savePath,exist_ok=True)
    loc = 0
    dList = os.listdir(diffusionImage)
    fList = os.listdir(fractralImage)
    oList = os.listdir(OriImage)


    for i in tqdm(oList):

        oriImage = getFile(os.path.join(OriImage, i))
        keys = i.split(".")[0]
        temDlist = []

        while loc < len(dList) and dList[loc].split(".")[0] == keys:
            temDlist.append(dList[loc])
            loc += 1

        for j in range(len(temDlist)):
            for _ in range(2):
                if os.path.exists(savePath + "//" + temDlist[j] + "%s.jpg" % _):
                    continue

                fImage = Image.open(os.path.join(fractralImage, random.choice(fList))).convert("RGB")
                fImage=fImage.resize((640,640))
                fImage=np.asarray(fImage)

                beta=random.uniform(0.15,0.25)

                dImage = getFile(os.path.join(diffusionImage, temDlist[j]))

                # if np.random.random()<0.5:
                #     dImage=augment_input(dImage)

                for k in range(np.random.randint(4)):
                    if np.random.random()<0.5:
                        aug_image_copy=augment_input(dImage)
                    else:
                        aug_image_copy=getFile(os.path.join(diffusionImage, random.choice(temDlist)))
                    mixed_op=np.random.choice(mixing)
                    dImage=mixed_op(dImage/255,aug_image_copy/255,4).copy()

                # IHS
                dImage = combine_images(dImage, oriImage)
                dImage = IHS(dImage, oriImage, 0.1)

                try:
                    dImage = dImage * 0.8+ fImage * 0.2
                except:
                    dImage = dImage

                dImage.clip(0,255)
                cv.imwrite(savePath + "//" +temDlist[j]+"%s.jpg"%_, dImage)









