import train.blocks

import tensorflow as tf

def build_model(input_shape, batch_size, num_frames, net_arch='ResNet50'):
    y = x = tf.keras.Input(input_shape, batch_size=batch_size*num_frames)
    y = train.blocks.backbone(net_arch, y)(y) # (batch*frames, h, w, c1)
    y = train.blocks.reduce_temporal_feature(y, num_frames) # (batch, frames, c2)
    y = train.blocks.self_similarity(y) # (batch, frames, frames, 1)
    y = train.blocks.period_embedding(y, batch_size, num_frames) # (batch, frames, embeddings)
    plength = train.blocks.period_length_classifier(y, num_frames // 2) # (batch, frames, frames//2)
    periodicity = train.blocks.periodicity_classifier(y) # (batch, frames, 1)
    return tf.keras.Model(
        inputs=[x],
        outputs=[plength, periodicity],
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
