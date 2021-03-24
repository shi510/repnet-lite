import train.blocks

import tensorflow as tf

def build_model(input_shape, batch_size, num_frames, net_arch='ResNet50'):
    y = x = tf.keras.Input(input_shape, batch_size=batch_size*num_frames)
    y = train.blocks.backbone(net_arch, y)(y)
    y = train.blocks.reduce_temporal_feature(y, num_frames)
    y = train.blocks.self_similarity(y)
    y = train.blocks.period_embedding(y, batch_size, num_frames)
    period_length_predictor = tf.keras.layers.Dense(
        num_frames // 2, name='period_length_predictor')(y)
    period_score_predictor = tf.keras.layers.Dense(
        1, name='period_score_predictor')(y)
    return tf.keras.Model(
        inputs=[x],
        outputs=[period_length_predictor, period_score_predictor],
        name='period_estimator')


if __name__ == '__main__':
    batch_size = 5
    num_frames = 64
    img_shape = [112, 112, 3]
    net = build_model(img_shape, batch_size, num_frames)
    dummy_input = tf.random.uniform([batch_size*num_frames]+img_shape)
    net.summary()
    outputs = net(dummy_input)
    for out in outputs:
        print(out.shape)
