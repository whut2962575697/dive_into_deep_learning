import torch
from tqdm import tqdm

from MultilayerPerceptron import generate_dataset, data_iter, cross_entropy_softmax, init_weights_and_bais, accuracy

def l2_normal(params, lamd, decay_bais=False):
    penalty = torch.tensor(0.)
    for param in params:
        if len(param.shape) == 1:
            if not decay_bais:
                continue
            else:
                penalty = penalty + 0.5*(param**2).sum()
        else:
            penalty = penalty + 0.5*(param**2).sum()
    return penalty

def dropout(x, drop_prob):
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    if keep_prob == 0:
        return torch.zeros_like(x)
    else:
        mask = (torch.rand(x.shape) < keep_prob).float()
        return mask*x / keep_prob




def sgd(params, lr, batch_size, decay_lamd=0.,  decay_bais=False):
    for param in params:
        if len(param.shape) == 1:
            if decay_bais:
                param.data = (1-lr*decay_lamd/batch_size)*param.data - lr*param.grad / batch_size
            else:
                param.data = param.data - lr*param.grad / batch_size
        else:
            param.data = (1-lr*decay_lamd/batch_size)*param.data - lr*param.grad / batch_size


def mlp(x, weights_list, bais_list, activation='relu', is_training=True, dropout_prob=0.):
    assert  len(weights_list) == len(bais_list)

    if activation == 'relu':
        activation_f = torch.nn.ReLU()
    elif activation == 'tanh':
        activation_f = torch.nn.Tanh()
    elif activation == 'sigmoid':
        activation_f = torch.nn.Sigmoid()
    else:
        activation_f = torch.nn.ReLU()
    for weights, bais in zip(weights_list[:-1], bais_list[:-1]):

        x = torch.mm(x, weights) + bais
        x = activation_f(x)
        if is_training:
            x = dropout(x, dropout_prob)
        
    x = torch.mm(x, weights_list[-1]) + bais_list[-1]
    return x

def train(train_dataset, test_dataset, lr, in_dim, hidden_dims, out_dim, batch_size, it_nums, decay_lamd, dropout_prob=0.):
    weights_list = list()
    bais_list = list()
    last_dim = in_dim
    for dim in hidden_dims:
        weights, bais = init_weights_and_bais((last_dim, dim), dim)
        weights_list.append(weights)
        bais_list.append(bais)
        last_dim = dim
    weights,bais = init_weights_and_bais((last_dim, out_dim), out_dim)
    weights_list.append(weights)
    bais_list.append(bais)
    
    
    for i in range(it_nums):
        train_loss = 0.
        train_dataloder = data_iter(train_dataset, batch_size, True)
        test_dataloder = data_iter(test_dataset, batch_size, False)
        for train_features, train_labels in tqdm(train_dataloder, leave=False):
            train_features = train_features.view(train_features.shape[0], -1)
            params = list()
            for weights_param, bais_param in zip(weights_list, bais_list):
                params.append(weights_param)
                params.append(bais_param)
            train_outputs = mlp(train_features, weights_list, bais_list, is_training=True, dropout_prob=dropout_prob)
            
            loss = cross_entropy_softmax(train_outputs, train_labels)
            if decay_lamd > 0.:
                l2 = l2_normal(params, decay_lamd)
                loss = loss + l2
            train_loss = train_loss + loss.data.item()
            loss.backward()
            
            sgd(params, lr, batch_size, decay_lamd=decay_lamd)
            
            for param in params:
                param.grad.data.zero_()
            
        train_loss = train_loss / len(train_dataset[0])

        print('[iter{0}] train loss:{1}'.format(i+1, train_loss))
        test_loss = 0.
        test_acc = 0.
        with torch.no_grad():
            for test_features, test_labels in tqdm(test_dataloder, leave=False):
                test_features = test_features.view(test_features.shape[0], -1)
                
                test_outputs = mlp(test_features, weights_list, bais_list, is_training=False)
                loss = cross_entropy_softmax(test_outputs, test_labels)
                acc = accuracy(test_outputs, test_labels)
                test_loss = test_loss + loss.data.item()
                test_acc = test_acc + acc
                 
        test_loss = test_loss / len(test_dataset[0])
        test_acc = test_acc / len(test_dataset[0])

        print('[iter{0}] test loss:{1} test acc:{2}'.format(i+1, test_loss, test_acc))



if __name__ == "__main__":
    lr = 0.03
    batch_size = 64
    in_dim = 28*28
    hidden_dims = [256]
    out_dim = 10
    it_nums = 30
    dataset_save_path = r'G:\xin.src\dataset\dataset'
    download_dataset = True
    train_dataset, test_dataset = generate_dataset(dataset_save_path, download_dataset)
    train(train_dataset, test_dataset, lr, in_dim, hidden_dims, out_dim, batch_size, it_nums, decay_lamd=0.0, dropout_prob=0.1)     
            

        


