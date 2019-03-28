import sys
import os
sys.path.append(os.pardir)

from chapter5.layer_naive import MulLayer

apple: float = 100.0
apple_num: float = 2.0
tax: float = 1.1

# Layer
mul_apple_layer: MulLayer = MulLayer()
mul_tax_layer: MulLayer = MulLayer()

# Forward
apple_price: float = mul_apple_layer.forward(apple, apple_num)
price: float = mul_tax_layer.forward(apple_price, tax)

print("price: {0}".format(int(price)))

# Backward
d_price: float = 1.0
d_apple_price: float = 0.0
d_tax: float = 0.0
d_apple_price, d_tax = mul_tax_layer.backward(d_price)
print("d_tax: {0}".format(int(d_tax)))
d_apple: float = 0.0
d_apple_num: float = 0.0
d_apple, d_apple_num = mul_apple_layer.backward(d_apple_price)
print("d_apple: {0}, d_apple_num: {1}".format(d_apple, int(d_apple_num)))
