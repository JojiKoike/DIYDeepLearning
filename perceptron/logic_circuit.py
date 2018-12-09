import numpy as np


def and_gate(x1, x2):
    w1, w2, b = 0.5, 0.5, -0.7
    X = np.array([x1, x2])
    W = np.array([w1, w2])
    tmp = np.dot(X, W) + b
    return 1 if tmp > 0 else 0


def nand_gate(x1, x2):
    w1, w2, b = -0.5, -0.5, 0.7
    X = np.array([x1, x2])
    W = np.array([w1, w2])
    tmp = np.dot(X, W) + b
    return 1 if tmp > 0 else 0


def or_gate(x1, x2):
    w1, w2, b = 0.5, 0.5, -0.4
    X = np.array([x1, x2])
    W = np.array([w1, w2])
    tmp = np.dot(X, W) + b
    return 1 if tmp > 0 else 0

for i in range(2):
    for j in range(2):
        print("{0} and {1}  = {2}".format(i, j, and_gate(i, j)))
        print("{0} nand {1} = {2}".format(i, j, nand_gate(i, j)))
        print("{0} or {1} = {2}".format(i, j, or_gate(i, j)))






