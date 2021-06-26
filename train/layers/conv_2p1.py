import tensorflow as tf
from tensorflow.keras.regularizers import l2


"""
An implementation of the paper:
A Closer Look at Spatiotemporal Convolutions for Action Recognition
Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, Manohar Paluri
CVPR 2018
"""
class Conv2p1Layer(tf.keras.layers.Layer):

    def __init__(self, sequence, channels, l2_reg_weight=1e-5):
        super(Conv2p1Layer, self).__init__()
        self.sequence = sequence
        self.channels = channels
        self.l2_reg_weight = l2_reg_weight
        self.conv2d = tf.keras.layers.Conv2D(self.channels, 3, padding='same',
                                kernel_regularizer=l2(self.l2_reg_weight),
                                use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.conv1d = tf.keras.layers.Conv1D(self.channels, 3, padding='same',
                                # dilation_rate=3,
                                kernel_regularizer=l2(self.l2_reg_weight),
                                use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

    def call(self, x):
        """
        Input shape: (N * T, H, W, C)
        """
        y = x
        y = self.conv2d(y)
        y = self.bn1(y)
        y = self.relu1(y)
        h = tf.shape(y)[1]
        w = tf.shape(y)[2]
        c = tf.shape(y)[3]
        y = tf.reshape(y, [-1, self.sequence, h * w, c])
        y = tf.transpose(y, [0, 2, 1, 3])
        y = tf.reshape(y, [-1, self.sequence, c])
        y = self.conv1d(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = tf.reshape(y, [-1, h * w, self.sequence, c])
        y = tf.transpose(y, [0, 2, 1, 3])
        y = tf.reshape(y, [-1, h, w, self.channels])
        return y

    def get_config(self):
        return {
            'sequence': self.sequence,
            'channels': self.channels,
            'l2_reg_weight': self.l2_reg_weight}
