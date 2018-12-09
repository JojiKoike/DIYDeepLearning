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

def xor_gate(x1, x2):
    s1 = nand_gate(x1, x2)
    s2 = or_gate(x1, x2)
    tmp = and_gate(s1, s2)
    return 1 if tmp > 0 else 0

if __name__ == '__main__':
    for i in range(2):
        for j in range(2):
            print("{0} and {1}  = {2}".format(i, j, and_gate(i, j)))
            print("{0} nand {1} = {2}".format(i, j, nand_gate(i, j)))
            print("{0} or {1} = {2}".format(i, j, or_gate(i, j)))
            print("{0} xor {1} = {2}".format(i, j, xor_gate(i, j)))






