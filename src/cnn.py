import torch 
import numpy as np 

def corr2d(X, K, padding=0, stride=1):
    if padding > 0:
        _X = torch.zeros((X.shape[-2]+2*padding, X.shape[-1]+2*padding))
        __X[padding:padding+X.shape[-2], padding:padding+X.shape[-1]] = X
    else:
        _X = X
    H, W = _X.shape[-2:]
 
    h, w = K.shape

    Y = torch.zeros(_X.shape[0], (H-h)//stride+1, (W-w)//stride+1)

    for i in range(Y.shape[1]):
        for j in range(Y.shape[2]):
            Y[:, i, j] = (_X[:, i:i+h, j:j+w]*K).view((_X.shape[0], -1)).sum(-1)
    return Y

class Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2d, self).__init__()
        if type(kernel_size) == type(0):
            kernel_size = (out_channels, in_channels, kernel_size, kernel_size)
        elif type(kernel_size) == type(()):
            kernel_size = (out_channels, in_channels, kernel_size[0], kernel_size[1])
        self.weights = torch.nn.Parameter(torch.randn(kernel_size))
        self.bais = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
    
    def forward(self, x):
        y_list = list()
        for i in range(self.out_channels):
            _y_list = list()
            for j in range(self.in_channels):
                _y_list.append(corr2d(x[:, j, :, :], self.weights[i, j], padding=self.padding, stride=self.stride)+self.bais[i, j])
            _y = torch.stack(_y_list).sum(0)
        y_list.append(_y)
        output = torch.stack(y_list, 1)
        return output


if __name__ == "__main__":
    conv = Conv2d(3, 2, 2,stride=2)
    X = torch.ones((4, 3, 7, 5))
    y = conv(X)
    print(y.shape)





