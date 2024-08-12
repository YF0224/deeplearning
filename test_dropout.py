import numpy as np

def train(rate, x, w1, b1, w2, b2):
    layer1 = np.maximum(0, np.dot(x, w1) + b1)
    mask1 = np.random.binomial(1, 1 - rate, layer1.shape)
    layer1 = layer1 * mask1

    layer2 = np.maximum(0, np.dot(layer1, w2) + b2)
    mask2 = np.random.binomial(1, 1 - rate, layer2.shape)
    layer2 = layer2 * mask2
    return layer2

def another_train(rate, x, w1, b1, w2, b2):
    layer1 = np.maximum(0, np.dot(x, w1) + b1)
    mask1 = np.random.binomial(1, 1 - rate, layer1.shape)
    layer1 = layer1 * mask1
    layer1 = layer1 / (1 - rate)

    layer2 = np.maximum(0, np.dot(layer1, w2) + b2)
    mask2 = np.random.binomial(1, 1 - rate, layer2.shape)
    layer2 = layer2 * mask2
    layer2 = layer2 / (1 - rate)
    return layer2

def test(rate, x, w1, b1, w2, b2):
    layer1 = np.maximum(0, np.dot(x, w1) + b1)
    layer1 = layer1 * (1 - rate)

    layer2 = np.maximum(0, np.dot(layer1, w2) + b2)
    layer2 = layer2 * (1 - rate)
    return layer2

def another_test(x, w1, b1, w2, b2):
    layer1 = np.maximum(0, np.dot(x, w1) + b1)
    layer2 = np.maximum(0, np.dot(layer1, w2) + b2)
    return layer2