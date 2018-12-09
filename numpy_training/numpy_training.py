import numpy as np

# Calc
x1 = np.array([1.0, 2.0, 3.0])
x2 = np.array([4.0, 5.0, 6.0])
print(x1)
print(type(x1))
print(x1 + x2)
print(x1 * x2)
print(x1 / x2)

# n dimension array
A = np.array([[1, 2], [3, 4]])
print(A.shape)
print(A.dtype)
B = np.array([[3, 0],[0, 6]])
print(A + B)
print(A - B)
print(A * 10)
print(A * B)
x = np.dot(A, B)
X = x.flatten()
print(X[X > 10])
