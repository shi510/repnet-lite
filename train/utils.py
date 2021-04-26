import tensorflow as tf

def pairwise_l2_distance(A, B):
    """
    (a-b)^2 = a^2 -2ab + b^2
    A shape = (N, D)
    B shaep = (C, D)
    result shape = (N, C)
    """
    row_norms_A = tf.math.reduce_sum(tf.square(A), axis=1)
    row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

    row_norms_B = tf.math.reduce_sum(tf.square(B), axis=1)
    row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

    dist = row_norms_A - 2. * tf.matmul(A, B, transpose_b=True) + row_norms_B
    return tf.math.maximum(dist, 0.)

def batched_pairwise_l2_distance(A, B):
    """
    (a-b)^2 = a^2 -2ab + b^2
    A shape = (N, D)
    B shaep = (C, D)
    result shape = (N, C)
    """
    batch = tf.shape(A)[0]
    row_norms_A = tf.math.reduce_sum(tf.square(A), axis=-1)
    row_norms_A = tf.reshape(row_norms_A, [batch, -1, 1])  # Column vector.

    row_norms_B = tf.math.reduce_sum(tf.square(B), axis=-1)
    row_norms_B = tf.reshape(row_norms_B, [batch, 1, -1])  # Row vector.

    dist = row_norms_A - 2. * tf.matmul(A, B, transpose_b=True) + row_norms_B
    return tf.math.maximum(dist, 0.)
