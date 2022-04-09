from .layers import Softmax
import numpy as np


class CrossEntropyLoss(object):
    def __init__(self):
        self.classifier = Softmax()

    def gradient(self):
        return self.grad

    def __call__(self, a, y, requires_acc=True):
        a = self.classifier.forward(a)
        self.grad = a - y
        loss = -1 * np.einsum('ij, ij->', y, np.log(a), optimize=True) / y.shape[0]
        if requires_acc:
            acc = np.argmax(a, axis=-1) == np.argmax(y, axis=-1)
            return acc.mean(), loss
        return loss