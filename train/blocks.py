import train.model_zoo as model_zoo
from train.layers.transformer import TransformerLayer
from train.utils import pairwise_l2_distance

import tensorflow as tf


def backbone(name, input_tensor):
    return model_zoo.get_model(name, input_tensor)


def reduce_temporal_feature(x,
                     seuqnce,
                     channels=512,
                     kernel_size=3,
                     dilation=3,
                     l2_reg_weight=1e-6):
    """
    Conv3D -> GlobalMaxpooling2D
    """
    y = x1 = tf.keras.Input(x.shape[1:], batch_size=x.shape[0])
    h = tf.shape(y)[1]
    w = tf.shape(y)[2]
    c = tf.shape(y)[3]
    y = tf.reshape(y, [-1, seuqnce, h, w, c])
    y = tf.keras.layers.Conv3D(channels, kernel_size, padding='same',
                               dilation_rate=(dilation, 1, 1),
                               kernel_regularizer=tf.keras.regularizers.l2(l2_reg_weight),
                               kernel_initializer='he_normal')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.reduce_max(y, [2, 3])
    return tf.keras.Model(x1, y, name='reduce_temporal_feature')(x)


def self_similarity(x, temperature=13.544):
    """Calculates self-similarity between batch of sequence of embeddings."""
    y = x1 = tf.keras.Input(x.shape[1:], batch_size=x.shape[0])
    def _get_sims(x):
        """Calculates self-similarity between sequence of embeddings."""
        dist = pairwise_l2_distance(x, x)
        sims = -1.0 * dist
        return sims
    y = tf.map_fn(_get_sims, y)
    y /= temperature
    y = tf.nn.softmax(y, axis=-1)
    y = tf.expand_dims(y, -1)
    return tf.keras.Model(x1, y, name='self_similarity')(x)


def period_embedding(x, batch_size, num_frames, d_model=512, n_heads=4, dff=512, conv_channels=32):
    """
        Conv2D -> FC -> Transformer -> FC -> FC
    """
    # (batch_size, num_frames, num_frames, 1)
    y = x1 = tf.keras.Input(x.shape[1:], batch_size=x.shape[0])
    y = tf.keras.layers.Conv2D(conv_channels, 3, padding='same')(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.reshape(y, [batch_size, num_frames, num_frames * 32])
    y = tf.keras.layers.Dense(d_model)(y) # (batch_size, num_frames, d_model)
    y = TransformerLayer(d_model, n_heads, dff, num_frames)(y)
    return tf.keras.Model(x1, y, name='period_embedding')(x)

def periodicity_classifier(x, fc_channels=512, dropout=0.25):
    """
        FC -> FC -> FC
    """
    y = x1 = tf.keras.Input(x.shape[1:], batch_size=x.shape[0])
    y = tf.keras.layers.Dropout(dropout)(y)
    y = tf.keras.layers.Dense(fc_channels,
            kernel_regularizer=tf.keras.regularizers.l2(1e-6))(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Dropout(dropout)(y)
    y = tf.keras.layers.Dense(fc_channels,
            kernel_regularizer=tf.keras.regularizers.l2(1e-6))(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Dense(1)(y)
    return tf.keras.Model(x1, y, name='periodicity_classifier')(x)

def period_length_classifier(x, length, fc_channels=512, dropout=0.25):
    """
        FC -> FC -> FC
    """
    y = x1 = tf.keras.Input(x.shape[1:], batch_size=x.shape[0])
    y = tf.keras.layers.Dropout(dropout)(y)
    y = tf.keras.layers.Dense(fc_channels,
            kernel_regularizer=tf.keras.regularizers.l2(1e-6))(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Dropout(dropout)(y)
    y = tf.keras.layers.Dense(fc_channels,
            kernel_regularizer=tf.keras.regularizers.l2(1e-6))(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Dense(length)(y)
    return tf.keras.Model(x1, y, name='period_length_classifier')(x)
