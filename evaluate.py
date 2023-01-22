import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision
import torch
from test_net import Net,testloader,classes,batch_size

net=torch.load('modle.pth')

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

def test():
    dataiter=iter(testloader)
    imgs,labels=next(dataiter)
    print(''.join('%5s'%classes[labels[j]] for j in range(batch_size)))
    outs=net(imgs)
    _,preds=torch.max(outs.data,1)
    print(''.join('%5s'%classes[preds[j]] for j in range(batch_size)))
    imshow(torchvision.utils.make_grid(imgs))

print(evaluate())
test()
