import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import time
import d2lzh_pytorch as d2l
from torch.utils.tensorboard import SummaryWriter
from tqdm import *
import matplotlib.pyplot as plt

show_mode = 0 #0的时候是plt绘制，1的时候是读取txt日志绘制
learning_rate = 0.001
root = os.getcwd() + '/data/' #数据集的地址
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH = 'F:/ele/best.pt'
mean_loss_list = []
train_acc_list = []
test_acc_list = []
#定义读取文件的格式
def default_loader(path):
    return Image.open(path).convert('L')

class MyDataset(Dataset):
    #初始化一些需要传入的参数和数据集的调用
    def __init__(self, txt, transform=None,
                 target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        imgs = []
        fh = open(txt, 'r')
        #按照传入的路径和txt文本参数，以只读的形式打开这个文本
        for line in fh:
            #迭代该列表，按行循环txt文本
            line = line.strip('\n')
            line = line.rstrip('\n')
            #删除本行string字符串末尾的指定字符，
            words = line.split()
            #用split的方式将该行分割成列表
            #split的默认参数是空格，所以不传递任何参数时分割空格
            imgs.append((words[0], int(words[1])))
            #把txt的内容读入imgs列表保存
            #word[0]是图片信息，words[1]是label
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    #对数据进行预处理并返回想要的信息
    #这个方法是必须要有的，用于按照索引读取每个元素的具体内容
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn) #按照标签里面的地址读取图片的RGB像素
        if self.transform is not None:
            img = self.transform(img)
        return img, label
        #return哪些内容，那么我们在训练时循环读取每个batch时，就能获取哪些内容
    #初始化一些需要传入的参数和数据集的调用
    #这个函数必须写，返回数据集长度，也就是多少张图片，和Loader长度区分
    def __len__(self):
        return len(self.imgs)

train_transforms = transforms.Compose(
    [transforms.RandomResizedCrop((28, 28)),
     transforms.ToTensor()]
)
test_transforms = transforms.Compose(
    [transforms.RandomResizedCrop((28, 28)),
     transforms.ToTensor()]
)

train_data = MyDataset(txt=root+'train.txt', transform=train_transforms)
test_data = MyDataset(txt=root+'test.txt', transform=test_transforms)
train_loader = DataLoader(dataset=train_data, batch_size=10, shuffle=True, num_workers=1)
test_loader = DataLoader(dataset=test_data, batch_size=10, shuffle=False, num_workers=1)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

net = LeNet()
lr, num_epoch = 0.001, 20
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
def test(net, test_iter, device):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for imgs, targets in test_iter:
            net.eval()
            y_hat = net(imgs.to(device)).argmax(dim=1)
            acc_sum += (y_hat == targets.to(device)).float().sum().cpu().item()
            net.train()
            n += targets.shape[0]
        return acc_sum/n

def train(net, train_iter, test_iter, start_epoch, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on:", device)
    loss = torch.nn.CrossEntropyLoss()#定义损失函数
    batch_count = 0 #第几个batch，如果7000张图片的batch_size是10，那么共有700个batch
    nb = len(train_iter)#训练数据一共有多少
    for epoch in range(start_epoch, num_epochs):
        #这里之所以会有start_epoch是为了后面直接加载上次未训练完的信息
        train_l_sum = 0.0#训练损失值
        train_acc_sum = 0.0#训练精度
        n, start = 0, time.time()
        pbar = tqdm(enumerate(train_iter), total=nb)
        #tqmd可以更直观地观察训练集加载的过程
        for i, (imgs, targets) in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)
            y_hat = net(imgs)#把像素信息传入网络得出预测结果
            l = loss(y_hat, targets)#计算预测结果和标签的损失值
            optimizer.zero_grad()#梯度清零
            l.backward()#反向传播
            optimizer.step()#优化器作用
            train_l_sum += l.cpu().item()
            #这里使用y_hat.argmax(dim=1)是因为该网络返回的是一个包含10个结果的向量
            # 这10个结果分别是所属类别的概率
            train_acc_sum += (y_hat.argmax(dim=1) == targets).sum().cpu().item()
            #10个类别里面取出最大值的索引作为结果
            n += targets.shape[0]
            batch_count += 1
            s = '%g/%g  %g' % (epoch, num_epochs - 1, len(targets))
            pbar.set_description(s)  # 这个就是进度条显示

        mean_loss = train_l_sum/batch_count
        train_acc = train_acc_sum/n
        test_acc = test(net, test_iter, device)
        #下面这三个列表作为全局变量用于后面的绘图
        mean_loss_list.append(mean_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('loss %.4f, train_acc %.3f, test_acc %.3f' % (mean_loss, train_acc, test_acc))
        #在所有的epoch训练完之后创建节点列表保存到.pt文件里面
        #这样创建的好处是可以把当前未训练完的epoch也保存进去
        chkpt = {'epoch': epoch,
                 'model': net.state_dict(),
                 'optimizer': optimizer.state_dict()}
        torch.save(chkpt, PATH)
        del chkpt

def load_weight():
    chkpt = torch.load(PATH, map_location=device)
    chkpt['model'] = {k: v for k, v in chkpt['model'].items() if net.state_dict()[k].numel() == v.numel()}
    net.load_state_dict(chkpt['model'], strict=False)
    net.to(device)
    optimizer.load_state_dict(chkpt['optimizer'])
    global start_epoch
    start_epoch = chkpt['epoch'] + 1
    del chkpt

def plot_use_plt(mean_loss_list, train_acc_list, test_acc_list, num_epoch):
    x1 = range(0, num_epoch)
    x2 = range(0, num_epoch)
    x3 = range(0, num_epoch)
    plt.subplot(1, 3, 1)
    plt.plot(x1, mean_loss_list, 'o-')
    plt.title('Train_loss vs.epochs')
    plt.ylabel('Train loss')
    plt.subplot(1, 3, 2)
    plt.plot(x2, train_acc_list, '.-')
    plt.title('Train_acc vs.epochs')
    plt.ylabel('Train acc')
    plt.subplot(1, 3, 3)
    plt.plot(x3, test_acc_list, '.-')
    plt.title('Test_acc vs.epochs')
    plt.ylabel('Test acc')
    plt.savefig("F:/ele/show.jpg")#这一句话一定要放在plt.show()前面
    plt.show()


if __name__ == '__main__':
    #模型迁移训练设置
    global start_epoch
    start_epoch = 0
    #load_weight()
    train(net, train_loader, test_loader, start_epoch=start_epoch, optimizer=optimizer, device=device, num_epochs=num_epoch)
    plot_use_plt(mean_loss_list, train_acc_list, test_acc_list, num_epoch)
