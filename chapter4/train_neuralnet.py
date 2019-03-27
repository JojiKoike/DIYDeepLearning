import sys
import os
sys.path.append(os.pardir)
import numpy as np
from typing import List
from dataset.mnist import load_mnist
from chapter4.two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_loss_list: List[float] = []
train_acc_list: List[float] = []
test_acc_list: List[float] = []


# Hyper Parameter
iters_num: int = 10000
train_size: int = x_train.shape[0]
batch_size: int = 100
learning_rate: float = 0.1
iter_per_epoch: int = max(int(train_size / batch_size), 1)

network: TwoLayerNet = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # Get Mini Batch
    batch_mask: np.ndarray = np.random.choice(train_size, batch_size)
    x_batch: np.ndarray = x_train[batch_mask]
    t_batch: np.ndarray = t_train[batch_mask]
    print(batch_mask)

    # Calc. Gradient
    grad: np.ndarray = network.numerical_gradient(x_batch, t_batch)

    # Update Params
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # Record Learning Process
    loss: float = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc: float = network.accuracy(x_train, t_train)
        test_acc: float = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

