import sys
import os
sys.path.append(os.pardir)
from chapter5.layer_naive import AddLayer, MulLayer


apple: float = 100.0
apple_num: float = 2.0
orange: float = 150
orange_num: float = 3.0
tax: float = 1.1

# Layer
mul_apple_layer: MulLayer = MulLayer()
mul_orange_layer: MulLayer = MulLayer()
add_apple_orange_layer: AddLayer = AddLayer()
mul_tax_layer: MulLayer = MulLayer()

# Forward
apple_price: float = mul_apple_layer.forward(apple, apple_num)
orange_price: float = mul_orange_layer.forward(orange, orange_num)
all_price: float = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)
print("price: {0}".format(int(price)))

# Backward
d_price: float = 1.0
d_all_price: float = 0.0
d_tax: float = 0.0
d_all_price, d_tax = mul_tax_layer.backward(d_price)
print("d_all_price: {0}, d_tax: {1}".format(d_all_price, d_tax))
d_apple_price: float = 0.0
d_orange_price: float = 0.0
d_apple_price, d_orange_price = add_apple_orange_layer.backward(d_all_price)
print("d_apple_price: {0}, d_orange_price: {1}".format(d_apple_price, d_orange_price))
d_apple: float = 0.0
d_apple_num: float = 0.0
d_apple, d_apple_num = mul_apple_layer.backward(d_apple_price)
print("d_apple: {0}, d_apple_num: {1}".format(d_apple, d_apple_num))
d_orange: float = 0.0
d_orange_num: float = 0.0
d_orange, d_orange_num = mul_orange_layer.backward(d_orange_price)
print("d_orange: {0}, d_orange_num: {1}".format(d_orange, d_orange_num))
