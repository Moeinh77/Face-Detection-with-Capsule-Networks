import tensorflow as tf


def batch(data, labels, batch_size):
    index = 0

    while True:
        if index + batch_size >= len(labels):
            yield data[index:], labels[index:]
            return
        else:
            yield data[index:index+batch_size], labels[index:index+batch_size]
            index += batch_size


def norm(s, axis=-1, epsilon=1e-7, keepdims=False, name=None):
    '''Numerically stable vector norm'''
    # TODO expand documentaiton
    
    with tf.name_scope(name, default_name='safe_norm'):

        # Compute ||s|| =approx= sqrt(||s||^2 + epsilon) in order to avoid
        # numerical issues such as division by zero or vanishing gradients if 
        # s = 0

        squared_norm = tf.reduce_sum(
            tf.square(s),
            axis=axis,
            keepdims=keepdims,
            name='squared_norm'
        )

        norm = tf.sqrt(squared_norm + epsilon, name='norm')

    return norm


def squash(s, axis=-1, epsilon=1e-7, name=None):
    """Implements the squash nonlinearity."""
    # TODO expand documentation

    with tf.variable_scope(name, default_name='squash'):
        
        squared_norm = tf.reduce_sum(
            tf.square(s),
            axis=axis,
            keepdims=True,
            name='squared_norm'
        )

        # ||s|| =approx= sqrt(||s||^2 + epsilon)
        # Note: We do this inline rather than using utils.norm() since we
        # need squared_norm for further calculations
        norm = tf.sqrt(squared_norm + epsilon, name='norm')

        squash_factor = tf.divide(
            squared_norm,
            (1 + squared_norm),
            name='squash_factor'
        )

        unit_vector = tf.divide(s, norm, name='unit_vector')

        squashed =  squash_factor * unit_vector

    return squashed

