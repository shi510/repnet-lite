from train.layers.conv_2p1 import Conv2p1Layer
from train.layers.conv3d import Conv3D
from train.layers.self_similarity import SelfSimilarity
from train.layers.period_embedding import PeriodEmbedding

import tensorflow as tf

class FC_BN_RELU(tf.keras.layers.Layer):

    def __init__(self, out_channels, dropout_rate=0.25):
        super(FC_BN_RELU, self).__init__()
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.fc = tf.keras.layers.Dense(out_channels)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def get_config(self):
        return {'out_channels': self.out_channels,
            'dropout_rate': self.dropout_rate}


class PeriodEstimator(tf.keras.Model):

    def __init__(self, backbone, num_frames, embedding_dim=512, **kwargs):
        super(PeriodEstimator, self).__init__(**kwargs)
        self.backbone = backbone
        self.num_frames = num_frames
        self.embedding_dim = embedding_dim
        # self.conv_2p1 = Conv2p1Layer(num_frames, embedding_dim)
        self.conv3d = Conv3D(num_frames, embedding_dim)
        self.gmaxpool2d = tf.keras.layers.GlobalMaxPool2D()
        self.period_embedding = PeriodEmbedding(num_frames)
        self.self_similarity = SelfSimilarity()
        self.periodcity_fc1 = FC_BN_RELU(512)
        self.periodcity_fc2 = FC_BN_RELU(512)
        self.periodcity_fc3 = tf.keras.layers.Dense(1)
        self.period_length_fc1 = FC_BN_RELU(512)
        self.period_length_fc2 = FC_BN_RELU(512)
        self.period_length_fc3 = tf.keras.layers.Dense(num_frames // 2)
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

    def compile(self, optimizer, **kwargs):
        super(PeriodEstimator, self).compile(**kwargs)
        self.optimizer = optimizer

    def spatial_model(self, x):
        return self.backbone(x)

    def temporal_model(self, x, is_batch_mode=True):
        # x = self.conv_2p1(x)
        x = self.conv3d(x)
        x = self.gmaxpool2d(x)
        if is_batch_mode:
            x = tf.reshape(x, [-1, self.num_frames, self.embedding_dim])
        x = self.self_similarity(x, is_batch_mode)
        x = self.period_embedding(x)
        return x

    def predictors(self, x):
        periodcity = x
        periodcity = self.periodcity_fc1(periodcity)
        periodcity = self.periodcity_fc2(periodcity)
        periodcity = self.periodcity_fc3(periodcity)
        period_length = x
        period_length = self.period_length_fc1(period_length)
        period_length = self.period_length_fc2(period_length)
        period_length = self.period_length_fc3(period_length)
        return periodcity, period_length

    def call(self, x):
        x = self.spatial_model(x)
        x = self.temporal_model(x)
        periodcity, period_length = self.predictors(x)
        return periodcity, period_length

    def train_step(self, data):
        x, y_true1, y_true2 = data
        with tf.GradientTape() as tape:
            periodcity, period_length = self(x)
            loss1 = tf.nn.softmax_cross_entropy_with_logits(y_true1, periodcity)
            loss2 = tf.nn.softmax_cross_entropy_with_logits(y_true2, period_length)
            loss1 = tf.math.reduce_mean(loss1)
            loss2 = tf.math.reduce_mean(loss2)
            loss = loss1 + loss2
        trainable_vars = self.trainable_weights
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]

    def split_model(self):
        x1 = tf.keras.Input(shape=(112, 112, 3), batch_size=1)
        y = self.spatial_model(x1)
        spatial_model = tf.keras.Model(inputs=[x1], outputs=[y], name='{}_spatial'.format(self.name))
        x2 = tf.keras.Input(shape=(7, 7, 1024), batch_size=self.num_frames)
        y = self.temporal_model(x2, is_batch_mode=False)
        y = self.predictors(y)
        temporal_model = tf.keras.Model(inputs=[x2], outputs=y, name='{}_temporal'.format(self.name))
        return spatial_model, temporal_model

    def get_model(self):
        x = tf.keras.Input(shape=(112, 112, 3), batch_size=64)
        y = self.call(x)
        return tf.keras.Model(inputs=[x], outputs=y, name=self.name)
