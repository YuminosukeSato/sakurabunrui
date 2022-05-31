import math
from platform import python_branch
from tkinter import NE
from turtle import forward

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


class Swish(nn.Module):  # Swish activation                                      
    def forward(self, x):
        return x * torch.sigmoid(x)
class SEblock(nn.Module): # Squeeze Excitation                                  
    def __init__(self, ch_in, ch_sq):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch_in, ch_sq, 1),
            Swish(),
            nn.Conv2d(ch_sq, ch_in, 1),
            )
        self.se.apply(weights_init)

    def forward(self, x):
        return x * torch.sigmoid(self.se(x))

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
class ConvBN(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size,stride=1, padding=0, groups=1):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size,
                        stride, padding, groups=groups, bias=False),
                nn.BatchNorm2d(ch_out),
        )
        self.layers.apply(weights_init)

    def forward(self, x):
        return self.layers(x)
class DropConnect(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if self.training:
            keep_rate = 1.0 - self.drop_rate
            r = torch.rand([x.size(0),1,1,1], dtype=x.dtype).to(x.device)
            r += keep_rate
            mask = r.floor()
            return x.div(keep_rate) * mask
        else:
            return x 
class BMConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out,
                 expand_ratio, stride, kernel_size,
                 reduction_ratio=4, drop_connect_rate=0.2):
        super().__init__()
        self.use_residual = (ch_in==ch_out) & (stride==1)
        ch_med = int(ch_in * expand_ratio)
        ch_sq = max(1, ch_in//reduction_ratio)

        # define network                                                        
        if expand_ratio != 1.0:
            layers = [ConvBN(ch_in, ch_med, 1),Swish()]
        else:
            layers = []

        layers.extend([
            ConvBN(ch_med, ch_med, kernel_size, stride=stride,
                   padding=(kernel_size-1)//2, groups=ch_med), # depth-wise    
            Swish(),
            SEblock(ch_med, ch_sq), # Squeeze Excitation                        
            ConvBN(ch_med, ch_out, 1), # pixel-wise                             
        ])

        if self.use_residual:
            self.drop_connect = DropConnect(drop_connect_rate)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.drop_connect(self.layers(x))
        else:
            return self.layers(x)  
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
class EfficientNet(nn.Module):
    def __init__(self, width_mult=1.0, depth_mult=1.0,resolution=False, dropout_rate=0.2, input_ch=3, num_classes=1000):
        super().__init__()

        # expand_ratio, channel, repeats, stride, kernel_size                   
        settings = [
            [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112                   
            [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56                   
            [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28                   
            [6,  80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14                   
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14                   
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7                   
            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7]                  
        ]

        ch_out = int(math.ceil(32*width_mult))
        features = [nn.AdaptiveAvgPool2d(resolution)] if resolution else []
        features.extend([ConvBN(input_ch, ch_out, 3, stride=2), Swish()])

        ch_in = ch_out
        for t, c, n, s, k in settings:
            ch_out  = int(math.ceil(c*width_mult))
            repeats = int(math.ceil(n*depth_mult))
            for i in range(repeats):
                stride = s if i==0 else 1
                features.extend([BMConvBlock(ch_in, ch_out, t, stride, k)])
                ch_in = ch_out

        ch_last = int(math.ceil(1280*width_mult))
        features.extend([ConvBN(ch_in, ch_last, 1), Swish()])

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(ch_last, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
def _efficientnet(w_mult, d_mult, resolution, drop_rate,input_ch, num_classes=22):
    model = EfficientNet(w_mult, d_mult,resolution, drop_rate,input_ch, num_classes)
    return model
def efficientnet_b7(input_ch=3, num_classes=22):
    #(w_mult, d_mult, resolution, droprate) = (2.0, 3.1, 600, 0.5)              
    return _efficientnet(2.0, 3.1, None, 0.5, input_ch, num_classes)
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, (12,9)) 
        self.conv2 = nn.Conv2d(20, 50, (9,6))  
        self.pool1 = nn.MaxPool2d(2, 2) 
        self.pool2 = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(6 * 4 * 50, 50)
        self.fc2 = nn.Linear(50, 22)
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=0)
    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(self.pool1(x))
        x = self.conv2(x)
        x = self.tanh(self.pool2(x))
        x = x.view(-1, 6 * 4 * 50)
        x = self.tanh(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x
"""
if __name__ == '__main__':
    train_img = np.load('X_train.npy')
    train_label = np.load('C:\\Users\\81804\\Desktop\\aidemy\\y_train.npy')
    test_img = np.load('X_test.npy')
    test_label = np.load('y_test.npy')
    print(test_label.shape)
    print(test_img.shape)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(dev)
    net = efficientnet_b7()
    #.to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train_img = torch.Tensor(train_img)
    train_label = torch.Tensor(train_label)
    test_img = torch.Tensor(test_img)
    test_label = torch.Tensor(test_label)
    print(test_label.shape)
    print(test_img.shape)
    train_dataset = TensorDataset(train_img, train_label)
    test_dataset = TensorDataset(test_img, test_label)
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=3,shuffle=False)
    for epoch in range(100):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()
            #print(inputs.shape)
            # forward + backward + optimize
            inputs = inputs.view(3,3,244,244)
            outputs = net(inputs).to(dev)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 106035 == 106034:
                print('[{:d}, {:5d}] loss: {:.20f}'
                        .format(epoch + 1, i + 1, running_loss / 106035))
                running_loss = 0.0

    print('Finished Training')
    model_path = 'model.pth'
    torch.save(net.state_dict(), model_path)
    correct = 0
    total = 0

    with torch.no_grad():
        for (images, labels) in test_loader:
            images = images.view(3,3,244,244)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy: {:.2f} %'.format(100 * float(correct/total)))

    