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
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((.5,.5,.5),(.5,.5,.5))
])
testTransform = transforms.Compose([

    transforms.ToTensor(),
    transforms.Normalize((.5,.5,.5),(.5,.5,.5))
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
    for i in range(1):

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
            showPercent = round(total * 0.25)
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



    rate1 = dotest("测试集", test_loader)
    # rate2 = dotest("训练集", test_train_loader)
    rate2 = 0
    return rate1, rate2


if __name__ == "__main__":

    #带预训练的初始模型
    modelNetTorch = torchvision.models.densenet169(pretrained=True)
    modelNetTorch.classifier=nn.Linear(modelNetTorch.classifier.in_features, 10)
    # for k, v in modelNetTorch.named_parameters():
    #     # v.requires_grad = True
    #     # k 是某一层模型名称，v为该层的模型参数
    #     v.requires_grad = False
    #     if "classifier" in k :
    #         v.requires_grad = True
    #     print(k, v.requires_grad)
    from vit_pytorch import ViT

    modelNetVit = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
    )
    #自己添油加醋
    class Net(nn.Module):
        def __init__(self,net):
            super(Net, self).__init__()
            self.net = net
            self.fc = nn.Linear(1000, 10)
        def forward(self, x):
            x = self.net(x)
            # x = self.fc(x)
            return x

    #套娃进自己的模型
    modelNet=Net(modelNetTorch)
    modelNet=modelNetTorch
    #存档名字
    NetTitle="densenet169"
    #爆显存了就缩一缩
    train_batch_size=1024
    test_batch_size=1024
    lr=0.1
    testrate=0

    trainDataset = datasets.CIFAR10('./train', train=True, download=True, transform=imageTransform)
    train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=train_batch_size, pin_memory=True,shuffle=True,
                                               num_workers=0)

    testDataset = datasets.CIFAR10('./train', train=False, download=True, transform=testTransform)

    test_loader = torch.utils.data.DataLoader(testDataset, batch_size=test_batch_size)

    testTrainDataset = datasets.CIFAR10('./train', train=True, download=True, transform=testTransform)
    test_train_loader = torch.utils.data.DataLoader(testTrainDataset, batch_size=test_batch_size)
    def changegrad(net):
        ignored_params = list(map(id, net.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
        # 对不同参数设置不同的学习率
        params_list = [{'params': base_params, 'lr': 0.001}, ]
        params_list.append({'params': net.classifier.parameters(), 'lr': 0.01})
        for k, v in net.named_parameters():
            v.requires_grad = True
        return params_list
    try:

        save = torch.load(f'./model/{NetTitle}Newest')
        model = save["model"]
        params_list=changegrad(model)
        model.cuda()
        optimizer = torch.optim.SGD(params_list, lr=lr, momentum=0.9,
                                    weight_decay=0.00008)
        try:
            optimizer.load_state_dict(save["optimizer"])
        except:
            pass
        epoch = save["epoch"]
        print(f"load from './model/{NetTitle}Newest'")
    except:
        # 全新训练
        params_list=changegrad(modelNet)
        model = modelNet.cuda()
        optimizer = torch.optim.SGD(params_list, lr=lr, momentum=0.9,
                                    weight_decay=0.00008)
        epoch = 0


    rate1, rate2 = test()
    testCorrectMax=rate1
    while rate1 < 0.9 or rate2 < 0.95:
        # 给👴训练到合格为止

        train()
        rate1, rate2 = test()

        if rate1 > testCorrectMax:
            testCorrectMax=rate1
            print("新的正确率")
            torch.save({"model": model, "optimizer": optimizer.state_dict(), "epoch": epoch},
                       f'./model/{NetTitle}Best')

    pass
