import torchvision
import time

writer = None
import os

try:
    os.mkdir("./model")
except:
    pass

import torch
import random
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(555)


# 网络模型
def timeShow(fn):
    def _wrapper(*args, **kwargs):
        start = time.time()
        ans = fn(*args, **kwargs)
        print("%s cost %s second" % (fn.__name__, time.time() - start))
        return ans

    return _wrapper


from torchvision import datasets, transforms

import torch

if torch.cuda.is_available():
    # to prevent opencv from initializing CUDA in workers
    torch.randn(8).cuda()
# 玄学加速
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn as nn
import torch.nn.functional as F

# -y*(x1-x2)+m =-y*x1+y*x2+m  x2=0
criterion = nn.MarginRankingLoss(margin=1)


def cast(t):
    return t.cuda(non_blocking=True) if torch.cuda.is_available() else t


def create_variables(sample):
    inputs = sample['input'].view(-1, 2, 64, 64)
    targets = sample['target'].view(-1)
    return inputs, targets


def train():
    global epoch

    model.train()
    import data_provider

    zeroT = torch.zeros(data_provider.batch_size).cuda()

    @timeShow
    def perEpoch(epoch):
        print(f"epoch{epoch}")
        avgloss = torch.zeros(1).cuda()
        for batch_idx, sample in enumerate(train_loader):
            # gpu加速
            # data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            data, target = create_variables(sample)

            optimizer.zero_grad()
            # 前向传播
            output = model(data)
            output = torch.flatten(output, 0)
            # 计算损失函数
            loss = criterion(output, zeroT, target)
            loss = loss + reg_loss(model)
            # 反向传播
            loss.backward()
            # 调参
            optimizer.step()
            avgloss += loss

        # torch.save(model, f'./model/resnet50{epoch}')
        # torch.save(model, f'./model/resnet50Newest')
        avgloss = avgloss.item()
        avgloss /= len(train_loader)
        scheduler.step(avgloss)

        print(f'{epoch} loss={avgloss}  datas={len(train_loader)} ')

        writer.add_scalar(NetTitle + "/loss", avgloss, global_step=epoch)
        writer.add_scalar(NetTitle + "/lr", optimizer.param_groups[0]['lr'], global_step=epoch)
        writer.flush()
        print("start save")
        try:
            os.rename(f'./model/{NetTitle}Newest', f'./model/{NetTitle}Newest.bak')
        except:
            pass
        torch.save({"model": model, "optimizer": optimizer.state_dict(), "epoch": epoch}, f'./model/{NetTitle}Newest')
        print("save success")

    # 跑n遍训练集
    for i in range(1):
        perEpoch(epoch)
        epoch += 1


import math

import numpy as np
import matplotlib.pyplot as plt


def test():
    def dotest(title, loader):
        with torch.no_grad():
            model.eval()  # 设置为test模式
            correct = torch.zeros(1).cuda()  # 初始化预测正确的数据个数为0
            total = len(loader)
            # 打log的概率
            showPercent = round(total)
            totalImg = 0
            for index, sample in enumerate(loader):
                # arrayImg = data.numpy()  # transfer tensor to array
                # arrayShow = np.squeeze(arrayImg[0], 0)  # extract the image being showed
                # plt.imshow(arrayShow)  # show image
                # plt.show()
                # exit(0)
                # data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                data, target = create_variables(sample)

                output = model(data)
                output = torch.flatten(output, 0)
                distance = torch.sub(target, output)
                distance = torch.abs(distance)
                judge = (distance < 1)
                correct += judge.sum()
                totalImg += target.size(0)
                if index % showPercent == 0:
                    pass
                    # print(f"{title}进度 {math.floor(index / total * 100)}%")
            correct = correct.item()
            print(f"{title}正确率 {correct}/{totalImg} = {round(correct / totalImg * 1000) / 10}%")
            return correct / totalImg

    rate1 = dotest("测试集", test_loader)
    rate2 = dotest("训练集", test_train_loader)
    return rate1, rate2


if __name__ == "__main__":
    NetTitle = "2ch"

    try:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(comment=NetTitle)
    except:
        pass

    from model_classes.m_2ch import MODEL_2CH
    from model_classes.m_2chstream import MODEL_2CH2STREAM

    modelNet = MODEL_2CH()
    # 存档名字
    # 爆显存了就缩一缩

    # 训练数据
    import data_provider

    train_loader = data_provider.get_train_iter()
    # 测试数据
    test_loader = data_provider.get_test_iter()
    # 训练集测试
    test_train_loader = data_provider.get_testtrain_iter()
    lr=0.05
    if os.path.exists(f'./model/{NetTitle}Newest'):
        save = torch.load(f'./model/{NetTitle}Newest')
        # save = torch.load(f'./model/{NetTitle}Best')
        model = save["model"]
        model.cuda()
        optimizer = torch.optim.ASGD(model.parameters(), lr=lr, t0=1e+4, alpha=0.1,
                                     weight_decay=0.0005)
        try:
            optimizer.load_state_dict(save["optimizer"])
            print('optimizer load success')
        except:
            pass
        epoch = save["epoch"]
        print(f"load from './model/{NetTitle}Newest'")
    else:
        print("全新训练")
        optimizer = torch.optim.ASGD(modelNet.parameters(), lr=lr, t0=1e+4, alpha=0.1,
                                     weight_decay=0.0005)
        epoch = 0
        writer.add_graph(modelNet, (torch.rand([1, 2, 64, 64])))
        model = modelNet.cuda()

    from regular import Regularization
    reg_loss = Regularization(model, 0.0005, p=2).to(device)


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.85,
                                                           patience=2, verbose=True,
                                                           threshold=0.04, threshold_mode='rel', cooldown=0,
                                                           min_lr=0, eps=1e-08)


    rate1, rate2 = test()
    testCorrectMax = rate1
    conf = True
    conf_times = 0
    while (conf or conf_times < 10 ) and epoch<=100:
        # 给👴训练到合格为止

        train()
        rate1, rate2 = test()
        writer.add_scalar(NetTitle + "/训练集", rate2, global_step=epoch)
        writer.add_scalar(NetTitle + "/测试集", rate1, global_step=epoch)

        if rate1 > testCorrectMax:
            testCorrectMax = rate1
            print("新的正确率")

            torch.save({"model": model, "optimizer": optimizer.state_dict(), "epoch": epoch},
                       f'./model/{NetTitle}Best')
        conf = rate1 < 1 or rate2 < 1
        if not conf:
            conf_times += 1
        else:
            conf_times = 0
    torch.save({"model": model, "optimizer": optimizer.state_dict(), "epoch": epoch},
               f'./model/{NetTitle}BestEnd')
    pass
