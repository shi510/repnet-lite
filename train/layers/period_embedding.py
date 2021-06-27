from train.layers.transformer import TransformerLayer

import tensorflow as tf

class PeriodEmbedding(tf.keras.layers.Layer):

    def __init__(self, num_frames, d_model=512, n_heads=4, dff=512, conv_channels=32):
        super(PeriodEmbedding, self).__init__()
        self.num_frames = num_frames
        self.d_model = d_model
        self.n_heads = n_heads
        self.dff = dff
        self.conv_channels = conv_channels
        self.conv2d = tf.keras.layers.Conv2D(self.conv_channels, 3, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.fc1 = tf.keras.layers.Dense(self.d_model)
        self.transformer = TransformerLayer(d_model, n_heads, dff, num_frames)

    def call(self, x):
        """
            Input Shape: (batch_size, num_frames, num_frames, 1)
            Conv2D -> FC -> Transformer
        """
        x = self.conv2d(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = tf.reshape(x, [-1, self.num_frames, self.num_frames * self.conv_channels])
        x = self.fc1(x) # (batch_size, num_frames, d_model)
        x = self.transformer(x)
        return x

    def get_config(self):
        return {
            'num_frames': self.num_frames,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'dff': self.dff,
            'conv_channels': self.conv_channels}
