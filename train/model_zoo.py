import tensorflow as tf

def ResNet50V2(x):
    model = tf.keras.applications.ResNet50V2(
        input_tensor=x,
        classifier_activation=None,
        include_top=False,
        weights='imagenet')
    model = tf.keras.models.Model(
            inputs=model.input,
            outputs=model.get_layer('conv4_block3_out').output,
            name=model.name)
    return model

model_list = {
    "ResNet50": ResNet50V2
}

def get_model(name, input_tensor):
    net = model_list[name](input_tensor)
    return net
