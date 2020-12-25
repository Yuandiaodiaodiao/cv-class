from torch import device, load, no_grad, unsqueeze
import torch
from torchvision import transforms
from PIL import Image
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import numpy as np
from model_classes.m_2ch import MODEL_2CH
modelpath="./model"
class TestCore():
    def __init__(self, net_title=""):
        d = device('cpu')
        self.gpu=False
        try:
            save = load(f'{modelpath}/{net_title}', map_location=d)
            self.model = save["model"]
            self.model.eval()
            self.minisave()
        except:
            # print("加载debug模型失败")
            pass


    def testimg(self, img1,img2):
        with no_grad():
            def transfer(img):
                mean=np.mean(img)
                img=cv2.resize(img,(64,64))
                img=(img.astype(np.float32)-mean)/256.0
                return img
            img1=transfer(img1)
            img2=transfer(img2)
            img2ch=np.array([img1,img2])
            img2ch=np.array([img2ch])
            img2ch=img2ch.astype(np.float32)
            tensor=torch.from_numpy(img2ch)
            output = self.model(tensor)
            output=output.item()
            return output

    def minisave(self):
        torch.save(self.model, f"{modelpath}/modelall.pkl")

    def miniload(self):
        d = device('cpu')
        self.model = load(f'{modelpath}/modelall.pkl', map_location=d)
        self.model.eval()

import matplotlib.pyplot as plt
if __name__ == "__main__":
    NetTitle = "2chBest"
    core = TestCore(NetTitle)
    core.miniload()
    from itertools import combinations
    lll=len(list(combinations([1,2,4,5],2)))
    print(lll)
    i=0
    for img1,img2 in combinations([1,2,4,5],2):
        img1o=cv2.imread(f"./test/{img1}.png")
        img2o=cv2.imread(f"./test/{img2}.png")
        img1=cv2.cvtColor(img1o,cv2.COLOR_BGR2GRAY)
        img2=cv2.cvtColor(img2o,cv2.COLOR_BGR2GRAY)
        res = core.testimg(img1,img2)
        plt.subplot(lll,3,1+i*3)
        plt.imshow(cv2.cvtColor(img1o,cv2.COLOR_BGR2RGB))
        plt.subplot(lll,3,2+i*3)
        plt.imshow(cv2.cvtColor(img2o,cv2.COLOR_BGR2RGB))
        plt.subplot(lll,3,3+i*3)
        if abs(res-1)<(res-(-1)):
            plt.text(0.5,0.5,"same")
            print("same")
        else:
            plt.text(0.5,0.5,"not same")
            print('not same')
        print(res)
        i+=1

    plt.show()
