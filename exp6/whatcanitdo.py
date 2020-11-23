import torch
import PIL
from torch.autograd import Variable
from torchvision.transforms import transforms
import time



class Netjudge(object):
    def __init__(self,net):
        t1 = time.time()
        self.net = net
        self.net.eval()
        self.transform = transforms.Compose([
            transforms.Resize(280, PIL.Image.BICUBIC),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化至[-1, 1]，规定均值和标准差
        ])
        # torch.save(self.net.state_dict(),"./model/alex1")
        print("初始化完成 耗时" + str((time.time() - t1)))

    def judge(self, filename=""):
        with torch.no_grad():
            img = PIL.Image.open(filename)
            img1 = self.transform(img)
            img1 = torch.unsqueeze(img1, 0)
            img1 = Variable(img1).cuda()
            # print(img1)
            outputs = self.net(img1)
            _, ans = torch.max(outputs.data, 1)
            return int(ans[0])

if __name__=="__main__":

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    save = torch.load('./model/resnet50Newest')
    model = save["model"]
    model.cuda()
    epoch = save["epoch"]
    print("load from './model/resnet50Newest'")
    judge=Netjudge(model)
    t=judge.judge("imgs/ship.jpg")
    print(f"是 {classes[t]}")