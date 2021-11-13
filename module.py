import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Model

class QDN(Model):
    def __init__(self):
        super(QDN, self).__init__()

        self.conv1 = Conv2D(32, 8, strides = (4, 4), padding = 'same', activation='relu')
        self.pool1 = MaxPool2D(pool_size=(2, 2), strides = (2, 2), padding = "same")

        self.conv2 = Conv2D(64, 4, strides = (2, 2), padding = 'same', activation='relu')
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides = (2, 2), padding = "same")

        self.conv3 = Conv2D(64, 3, strides = (1, 1), padding = 'same', activation='relu')
        self.pool3 = MaxPool2D(pool_size=(2, 2), strides = (2, 2), padding = "same")

        self.flatten = Flatten()
        self.d1 = Dense(512, activation='relu')
        self.d2 = Dense(2)

    def call(self, x):
        '''
        x : (80x80x4)
        h_conv1 : (20x20x32)
        h_conv2 : (5x5x64)
        h_conv3 : (3x3x64) 
        '''

        # hidden layers
        h_conv1 = self.conv1(x)
        h_pool1 = self.pool1(h_conv1)

        h_conv2 = self.conv2(h_pool1)
        h_pool2 = self.pool2(h_conv2)

        h_conv3 = self.conv3(h_pool2)
        h_pool3 = self.pool3(h_conv3)

        #h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
        h_conv3_flat  = self.flatten(h_pool3)

        h_fc1 = self.d1(h_conv3_flat)
        h_fc2 = self.d2(h_fc1)

        return h_fc2