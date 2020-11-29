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
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
    save = torch.load('./model/ResNet18Best')
    model = save["model"]
    model.cuda()
    epoch = save["epoch"]
    print("load from './model/ResNet18Best'")
    judge=Netjudge(model)
    t=judge.judge("imgs/ship.jpg")
    print(f"是 {classes[t]}")