import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision
import torch
from test_net import Net

net=torch.load('modle.pth')
#读取测试集
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
test_set=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
testloader=torch.utils.data.DataLoader(test_set,batch_size=4,shuffle=False)
classes=('飞机','轿车','鸟','猫','鹿','狗','蛙','马','船','货车')

def imshow(img):
    img=img/2+0.5  #逆归一化
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

def evaluate():
    correct=0
    total=0
    with torch.no_grad():
        for data in testloader:
            imgs,labels=data
            outputs=net(imgs)
            _,predict=torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=(predict==labels).sum().item()
    return 100*correct/total

dataiter=iter(testloader)
imgs,labels=next(dataiter)
imshow(torchvision.utils.make_grid(imgs))
print(''.join('%5s'%classes[labels[j]] for j in range(4)))

print(evaluate())