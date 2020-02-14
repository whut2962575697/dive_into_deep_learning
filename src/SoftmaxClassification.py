import torch
import torchvision
import torchvision.transforms as transforms
import numpy as ny 
from tqdm import tqdm

from LinearRegression import data_iter, sgd, init_weights_and_bais, Linear



def generate_dataset(save_path, download=False):
    train_dataset = torchvision.datasets.FashionMNIST(root=save_path, train=True, download=download, transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.FashionMNIST(root=save_path, train=False, download=download, transform=transforms.ToTensor())
    train_features = torch.cat([feature for [feature, _] in train_dataset], 0)
    train_labels = torch.Tensor([label for [_, label] in train_dataset]).long()

    test_features = torch.cat([feature for [feature, _] in test_dataset], 0)
    test_labels = torch.Tensor([label for [_, label] in test_dataset]).long()
    return [train_features, train_labels], [test_features, test_labels]

def softmax(x):
    x = torch.exp(x)
    x_sum = x.sum(dim=1, keepdim=True)
    return x / x_sum

def to_onehot(y, class_num):
    y = torch.zeros((y.shape[0], class_num)).scatter_(1, y.unsqueeze(1), 1)
    return y



def cross_entropy_softmax(y_pre, y_true):
    y_pre = softmax(y_pre)
    y_true = to_onehot(y_true, y_pre.shape[1])
    loss = - y_true*torch.log(y_pre)
    return loss.sum()

def accuracy(y_pre, y_true):
    return (torch.argmax(y_pre, dim=1) == y_true).float().sum().item()

def train(train_dataset, test_dataset, lr, weights_shape, bais_len, batch_size, it_nums):
    weights, bais = init_weights_and_bais(weights_shape, bais_len)
    
    for i in range(it_nums):
        train_loss = 0.
        train_dataloder = data_iter(train_dataset, batch_size, True)
        test_dataloder = data_iter(test_dataset, batch_size, False)
        for train_features, train_labels in tqdm(train_dataloder, leave=False):
            train_features = train_features.view(train_features.shape[0], -1)
            train_outputs = Linear(train_features, weights, bais)
            loss = cross_entropy_softmax(train_outputs, train_labels)
            train_loss = train_loss + loss.data.item()
            loss.backward()
            sgd([weights, bais], lr, batch_size)
            weights.grad.data.zero_()
            bais.grad.data.zero_()
        train_loss = train_loss / len(train_dataset[0])

        print('[iter{0}] train loss:{1}'.format(i+1, train_loss))
        test_loss = 0.
        test_acc = 0.
        with torch.no_grad():
            for test_features, test_labels in tqdm(test_dataloder, leave=False):
                test_features = test_features.view(test_features.shape[0], -1)
                
                test_outputs = Linear(test_features, weights, bais)
                loss = cross_entropy_softmax(test_outputs, test_labels)
                acc = accuracy(test_outputs, test_labels)
                test_loss = test_loss + loss.data.item()
                test_acc = test_acc + acc
                 
        test_loss = test_loss / len(test_dataset[0])
        test_acc = test_acc / len(test_dataset[0])

        print('[iter{0}] test loss:{1} test acc:{2}'.format(i+1, test_loss, test_acc))
    




if __name__ == "__main__":
    lr = 0.01
    batch_size = 64
    weights_shape = [28*28, 10]
    bais_len = 10
    it_nums = 20
    dataset_save_path = r'G:\xin.src\dive_into_deep_learning\dataset'
    download_dataset = True
    train_dataset, test_dataset = generate_dataset(dataset_save_path, download_dataset)
    train(train_dataset, test_dataset, lr, weights_shape, bais_len, batch_size, it_nums)