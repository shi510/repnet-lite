from train.utils import batched_pairwise_l2_distance
from train.utils import pairwise_l2_distance

import tensorflow as tf

class SelfSimilarity(tf.keras.layers.Layer):

    def __init__(self, temperature=1/13.544):
        super(SelfSimilarity, self).__init__()
        self.temperature = temperature

    def call(self, x, is_batch_mode):
        if is_batch_mode:
            x = -1.0 * batched_pairwise_l2_distance(x, x)
        else:
            x = -1.0 * pairwise_l2_distance(x, x)
        x *= self.temperature
        x = tf.nn.softmax(x, axis=-1)
        x = tf.expand_dims(x, -1)
        if not is_batch_mode:
            x = tf.expand_dims(x, 0)
        return x

    def get_config(self):
        return {'temperature': self.temperature}
