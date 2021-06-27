import tensorflow as tf
from tensorflow.keras.regularizers import l2


class Conv3D(tf.keras.layers.Layer):

    def __init__(self, sequence, channels, l2_reg_weight=1e-5):
        super(Conv3D, self).__init__()
        self.sequence = sequence
        self.channels = channels
        self.l2_reg_weight = l2_reg_weight
        self.conv3d = tf.keras.layers.Conv3D(channels, 3, padding='same',
                                dilation_rate=(3, 1, 1),
                                kernel_regularizer=l2(l2_reg_weight),
                                kernel_initializer='he_normal')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        """
        Input shape: (N * T, H, W, C)
        """
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        c = tf.shape(x)[3]
        x = tf.reshape(x, [-1, self.sequence, h, w, c])
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = tf.reshape(x, [-1, h, w, self.channels])
        return x

    def get_config(self):
        return {
            'sequence': self.sequence,
            'channels': self.channels,
            'l2_reg_weight': self.l2_reg_weight}
