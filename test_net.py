import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch
import torchvision
import torchvision.transforms as transforms

batch_size=32

#定义网络
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv2d1=nn.Conv2d(3,8,5,1,2)
        self.conv2d2=nn.Conv2d(8,16,3,1,1)
        self.conv2d3=nn.Conv2d(16,32,3,1,1)
        self.conv2d4=nn.Conv2d(32,64,3,1,1)
        self.conv2d5=nn.Conv2d(64,128,3,1,1)
        self.conv2d6=nn.Conv2d(128,128,3,1,1)
        self.fc=nn.Linear(128*16*16,10)
        self.dropout=nn.Dropout(0.5)
    def forward(self,x):
        x=F.relu(self.conv2d1(x))
        x=F.max_pool2d(F.relu(self.conv2d2(x)),(2,2))
        x=F.relu(self.conv2d3(x))
        x=F.relu(self.conv2d4(x))
        x=F.relu(self.conv2d5(x))
        x=F.relu(self.conv2d6(x))
        x=x.view(-1,128*16*16)
        x=self.fc(x)
        return x

def weight_init(m):
    classname=m.__class__.__name__
    if classname.find('conv')!=-1:
        m.weight.data.normal (0.0,0.02)
    elif classname.find('BatchNorm')!=-1:
        m.weight.data.normal (1.0,0.02)
        m.bias.data.fill_(0)
#实例化网络
net=Net()
net.apply(weight_init)
print(net)
#定义损失函数和优化器
lossfun=nn.CrossEntropyLoss()
opter=opt.Adam(net.parameters(),lr=0.001,weight_decay=0.0001)
print(str(opter))
#读取数据集
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_set=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
test_set=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
trainloader=torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
testloader=torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False)
classes=('飞机','轿车','鸟','猫','鹿','狗','蛙','马','船','货车')
#定义训练函数
def train(epochs):
    for epoch in range(epochs):
        running_loss=0.0
        for i, data in enumerate(trainloader,0):
            inputs, lables=data
            opter.zero_grad()
            outputs=net(inputs)
            loss=lossfun(outputs,lables)
            loss.backward()
            opter.step()
            running_loss+=loss.item()
            if i % 2000==1999:#每2000个batch打印一次
                print('[%d, %5d] loss: %.3f'%(epoch+1,i+1,running_loss/2000))
                running_loss=0.0
    print('finish')
#训练并保存模型
if __name__=='__main__':
    cmd=str(input('start new train?[y/n]'))
    epochs=int(input('epochs:'))
    if cmd=='n' or cmd=='N':
        net=torch.load('modle.pth')
    train(epochs)
    torch.save(net,'modle.pth')
