__author__ = 'alex'

from theBrain import *
import numpy as np


class prediction:
    def __init__(self, path=''):
        self.path = path
        self.x_data = []
        self.y_data = []
        self.brain = None


    def get_add_speed(self):
        return [self.x_data.append, self.y_data.append]

    def get_pred_speed(self):
        return [self.brain.predictSVM]

    def create_brain(self):
        if len(self.y_data) is 0:
            self.y_data = None
        self.brain = Brain(self.x_data, self.y_data, True)
        self.brain.initClustering(2)
        self.brain.initSVM(MODEL.SVM.SVC, options='auto', paramRange=[np.arange(1, 10, 1)])
        self.brain.saveBrain(self.path, True)

    def load_brain(self):
        self.brain = Brain.loadBrain(self.path)

    def add_data(self, speed, x_data, y_data=None):
        speed[0](x_data)
        if y_data is not None:
            speed[1](y_data)

    def get_data_length(self):
        return len(self.x_data)

    def predict(self, data, speed):
        return speed[0](data)

