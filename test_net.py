import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch
import torchvision
import torchvision.transforms as transforms

#定义网络
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv2d1=nn.Conv2d(3,8,5,1,2)
        self.conv2d2=nn.Conv2d(8,16,3,1,1)
        self.fc1=nn.Linear(16*16*16,200)
        self.fc2=nn.Linear(200,10)
        self.dropout=nn.Dropout(0.5)
    def forward(self,x):
        x=F.relu(self.conv2d1(x))
        x=F.max_pool2d(F.relu(self.conv2d2(x)),(2,2))
        x=x.view(-1,16*16*16)
        x=F.relu(self.fc1(x))
        x=self.dropout(x)
        x=self.fc2(x)
        return x
#实例化网络
net=Net()
print(net)
#定义损失函数和优化器
lossfun=nn.CrossEntropyLoss()
opter=opt.Adam(net.parameters(),lr=0.002)
#读取数据集
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_set=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
test_set=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
trainloader=torch.utils.data.DataLoader(train_set,batch_size=4,shuffle=True)
testloader=torch.utils.data.DataLoader(test_set,batch_size=4,shuffle=False)
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
    train(10)
    torch.save(net,'modle.pth')