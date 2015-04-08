__author__ = 'alex'
import neurolab as nl
import numpy as np
import time

class Neural_Net:
    def __init__(self, x_data, y_data, hidden, elements):
        self.limits = self.create_limit_array(elements)
        self.classes = len(np.unique(y_data))
        self.network = nl.net.newelm(self.limits, [2, 1])
        print('Starting Train')
        t0 = time.time()
        self.error = self.network.train(x_data, y_data, epochs = 20, show = 1, goal = 0.01)
        t1 = time.time()
        print('Finished Training network: %.2fs' % (t1 - t0))

    def create_limit_array(self, elem):
        out_shape = elem * 32
        single = [[0, 255]]
        limits = np.array(single * out_shape)
        return limits

    def predict(self, data, speed):
        return speed[0](data)

    def get_neural_speed(self):
        return [self.network.sim]


