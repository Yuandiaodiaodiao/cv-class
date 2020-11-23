import torchvision
import time
# 网络模型
def timeShow(fn):
    def _wrapper(*args, **kwargs):
        start = time.time()
        ans = fn(*args, **kwargs)
        print("%s cost %s second" % (fn.__name__,time.time() - start))
        return ans
    return _wrapper
from torchvision import datasets, transforms

imageTransform = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.Resize(224),                  #resnet默认图片输入大小224*224
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

import torch

#玄学加速
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn as nn



criterion = nn.CrossEntropyLoss()
from torch.autograd import Variable




def train():
    global epoch
    trainDataset = datasets.CIFAR10('./train', train=True, download=True, transform=imageTransform)
    train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=train_batch_size,pin_memory=True,num_workers=0)
    model.train()

    @timeShow
    def perEpoch(epoch):
        print(f"epoch{epoch}")
        showPercent = round(len(train_loader) * 0.25)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            # gpu加速
            data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            # 前向传播
            output = model(data)
            # 计算损失函数
            loss = criterion(output, target)
            # 反向传播
            loss.backward()
            # 调参
            optimizer.step()

            if batch_idx % showPercent == 0:
                print(f'{epoch} {batch_idx}/{len(train_loader)} {loss.item()}  ')
        # torch.save(model, f'./model/resnet50{epoch}')
        # torch.save(model, f'./model/resnet50Newest')
        print("start save")
        torch.save({"model": model, "optimizer": optimizer.state_dict(), "epoch": epoch}, f'./model/{NetTitle}Newest')
        print("save success")
    # 跑n遍训练集
    for i in range(5):

        perEpoch(epoch)
        epoch += 1


import math


def test():
    def dotest(title, loader):
        with torch.no_grad():
            model.eval()  # 设置为test模式
            correct = 0  # 初始化预测正确的数据个数为0
            total = len(loader)
            # 打log的概率
            showPercent = round(total * 0.1)
            totalImg = 0
            for index, (data, target) in enumerate(loader):
                data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == target).sum().item()
                totalImg += target.size(0)
                if index % showPercent == 0:
                    print(f"{title}进度 {math.floor(index / total * 100)}%")
            print(f"{title}正确率 {correct}/{totalImg} = {round(correct / totalImg * 100)}%")
            return correct / totalImg

    testDataset = datasets.CIFAR10('./train', train=False, download=True, transform=imageTransform)

    test_loader = torch.utils.data.DataLoader(testDataset, batch_size=test_batch_size)

    testTrainDataset = datasets.CIFAR10('./train', train=True, download=True, transform=imageTransform)
    test_train_loader = torch.utils.data.DataLoader(testTrainDataset, batch_size=test_batch_size)

    rate1 = dotest("测试集", test_loader)
    # rate2 = dotest("训练集", test_train_loader)
    rate2 = 0
    return rate1, rate2


if __name__ == "__main__":

    #带预训练的初始模型
    modelNetTorch = torchvision.models.densenet121(pretrained=True)

    #自己添油加醋
    class Net(nn.Module):
        def __init__(self,net):
            super(Net, self).__init__()
            self.net = net
            self.fc = nn.Linear(1000, 10)
        def forward(self, x):
            x = self.net(x)
            x = self.fc(x)
            return x

    #套娃进自己的模型
    modelNet=Net(modelNetTorch)
    #存档名字
    NetTitle="fcdensenet121"
    #爆显存了就缩一缩
    train_batch_size=512
    test_batch_size=128

    try:

        save = torch.load(f'./model/{NetTitle}Newest')
        model = save["model"]
        model.cuda()
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.05, momentum=0.9,
                                    weight_decay=5e-4)
        optimizer.load_state_dict(save["optimizer"])
        epoch = save["epoch"]
        print(f"load from './model/{NetTitle}Newest'")
    except:
        # 全新训练
        model = modelNet.cuda()
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.05, momentum=0.9,
                                    weight_decay=5e-4)
        epoch = 0


    rate1, rate2 = 0, 0
    while rate1 < 0.9 or rate2 < 0.95:
        # 给👴训练到合格为止
        train()
        rate1, rate2 = test()
    pass
