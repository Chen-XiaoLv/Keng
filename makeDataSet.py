import os
import shutil
from tqdm import tqdm


path=r"C:\Users\Administrator\Desktop\save1"
lpath=r"C:\Users\Administrator\Desktop\0710数据集\0710数据集\G\train\labels"
lc=r"C:\Users\Administrator\Desktop\DiffusionMix0721"
os.makedirs(lc,exist_ok=True)
def preprocess():
    datapath=r"C:\Users\Administrator\Downloads\G\train\images"
    ldatapath=r"C:\Users\Administrator\Downloads\G\train\labels"
    for idx,val in enumerate(os.listdir(datapath)):
        os.rename(datapath+"//"+val,datapath+"//"+str(idx)+".jpg")
        os.rename(ldatapath+"//"+val[:-4]+".txt",ldatapath+"//"+str(idx)+".txt")


for i in tqdm(os.listdir(path)):
    ls=i.split("_")
    os.makedirs(lc+"\\images",exist_ok=True)
    os.makedirs(lc+"\\labels",exist_ok=True)
    shutil.copyfile(path+"\\"+i,lc+"\\images"+"\\"+i)
    shutil.copyfile(lpath+"\\"+ls[0][:-4]+".txt",lc+"\\labels\\"+i[:-4]+".txt")