from typing import List, Dict
import sys
import os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from chapter5.two_layer_net import TwoLayerNet
import numpy as np

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

network: TwoLayerNet = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num: int = 10000
train_size: int = x_train.shape[0]
batch_size: int = 100
learning_rate: float = 0.1

train_loss_list: List[float] = []
train_acc_list: List[float] = []
test_acc_list: List[float] = []

iter_per_epoch: int = max(int(train_size / batch_size), 1)

for i in range(iters_num):
    batch_mask: np.ndarray = np.random.choice(train_size, batch_size)
    x_batch: np.ndarray = x_train[batch_mask]
    t_batch: np.ndarray = t_train[batch_mask]

    grad: Dict[str, np.ndarray] = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(t_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)


