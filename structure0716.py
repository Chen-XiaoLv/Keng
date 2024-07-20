import cv2 as cv
import numpy as np

from ImageAug import *
from ImageMix import *
import os
from tqdm import tqdm
from combine import combine_images
from utils import *

diffusionImage=r"C:\Users\Administrator\Downloads\result2.0\result\generated\class0"
fractralImage=r"C:\Users\Administrator\Downloads\result2.0\result\fractal\class0"
OriImage=r"C:\Users\Administrator\Downloads\result1.5\result\original_resized\class0"
savePath=r"C:\Users\Administrator\Desktop\outs"

def augment_input(image):
    aug_list=augmentations_
    op=np.random.choice(aug_list)
    return op(image.copy(),1)

def getFile(path):
    return Image.open(path)


def oriAug(img,k=3):
    aug=img.copy()
    for i in range(np.random.randint(k+1)):
        aug=augment_input(aug)
    return aug


# 获取Mask
# combine_images(A,B)

# 影像加法
# imageAdd(a,b,alpha)

# 影像乘法
# 此时需要先将其转化为(0,1)区间
# imageMultiply(a,b,beta)

# cv 与 PIL
# 转成cv: np.array(image,dtype)
# 转成Image: Image.fromarray(data)
class Train:
    def __init__(self):
        os.makedirs(savePath, exist_ok=True)

    def reset(self):
        self.loc = 0
        self.dList = os.listdir(diffusionImage)
        self.fList = os.listdir(fractralImage)
        self.oList = os.listdir(OriImage)

    def structure0716(self):
        self.reset()
        for i in tqdm(self.oList):
            oriImage = getFile(os.path.join(OriImage, i))  # 原始影像
            keys = i.split(".")[0]
            temDlist = []  # 所有diffusion生成图
            temFlist = []

            while self.loc < len(self.dList) and self.dList[self.loc].split(".")[0] == keys:
                temDlist.append(self.dList[self.loc])
                temFlist.append(self.fList[self.loc])
                self.loc += 1

            # 针对所有diffusion图进行合成
            for j in range(len(temDlist)):
                fImage = getFile(fractralImage + "//" + temFlist[j])

                res = oriAug(oriImage)
                # print(getFile(diffusionImage+"//"+temDlist[j]))
                res = combine_images(res, getFile(diffusionImage + "//" + temDlist[j]))

                res = imageAdd(res, fImage)
                res = imageMultiply(res, oriImage)
                res = imageMultiply(res, fImage, 4000)
                # cv.imwrite(savePath+"//"+temDlist[j],res)
                res.save(savePath + "//" + temDlist[j], "jpeg")

    def structure0717(self):
        self.reset()
        for i in tqdm(self.oList):
            oriImage = getFile(os.path.join(OriImage, i))  # 原始影像
            keys = i.split(".")[0]
            temDlist = []  # 所有diffusion生成图
            temFlist = []

            while self.loc < len(self.dList) and self.dList[self.loc].split(".")[0] == keys:
                temDlist.append(self.dList[self.loc])
                temFlist.append(self.fList[self.loc])
                self.loc += 1
                # 针对所有diffusion图进行合成
            for j in range(len(temDlist)):
                aug=oriImage.copy()
                for k in range(3):
                    aug = oriAug(aug)
                fImage = getFile(fractralImage + "//" + temFlist[j])

                dImage=getFile(diffusionImage + "//" + temDlist[j])
                # print(getFile(diffusionImage+"//"+temDlist[j]))

                res=oriImage.copy()
                res = ImageListAdd([res,fImage,aug,dImage])

                res.save(savePath + "//" + temDlist[j], "jpeg")

    def structure0718(self):
        self.reset()
        for i in tqdm(self.oList):
            oriImage = getFile(os.path.join(OriImage, i))  # 原始影像
            keys = i.split(".")[0]
            temDlist = []  # 所有diffusion生成图
            temFlist = []

            while self.loc < len(self.dList) and self.dList[self.loc].split(".")[0] == keys:
                temDlist.append(self.dList[self.loc])
                temFlist.append(self.fList[self.loc])
                self.loc += 1
                # 针对所有diffusion图进行合成
            for j in range(len(temDlist)):
                aug = oriImage.copy()
                for k in range(2):
                    aug = oriAug(aug)
                fimg = getFile(fractralImage + "//" + temFlist[j])
                dimg=getFile(diffusionImage + "//" + temDlist[j])

                # dimg=imageMultiply(dimg,aug)
                # fimg=imageMultiply(fimg,dimg)

                res=ImageListAdd([aug,dimg,fimg],weight=[0.12,0.68,0.2])
                # res=imageAdd(res,dimg)
                # # res=imageMultiply(res,oriImage)
                # res=imageMultiply(res,fimg,4000)

                res.save(savePath + "//" + temDlist[j], "jpeg")



if __name__ == '__main__':
    t=Train()
    t.structure0718()
