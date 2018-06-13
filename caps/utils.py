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


def intersection_over_union(box_a, box_b, epsilon=1e-7, name=None):
    """Computes the IoU metric for two tensors of boxes
    
    Boxes with neg. width/height are treated as area=0!!

    Args:
        box_a: shape(batch_size, 4) with x, y, width, height
        box_b: shape(batch_size, 4) with x, y, width, height
    """

    with tf.variable_scope(name, default_name='IoU'):
        
        # Convert from box to point representation
        start_a, extent_a = tf.split(box_a, 2, axis=1)
        start_b, extent_b = tf.split(box_b, 2, axis=1)

        # Prevent negative box areas 
        extent_a = tf.maximum(extent_a, 0, name='extent_a')
        extent_b = tf.maximum(extent_b, 0, name='extent_b')

        end_a = start_a + extent_a
        end_b = start_b + extent_b

        # Calculate intersection (in point representation)
        start_cut = tf.maximum(start_a, start_b, name='start_cut')
        end_cut = tf.minimum(end_a, end_b, name='end_cut')

        # Calculate area of intersection
        extent_cut = tf.maximum(end_cut - start_cut, 0, name='extent_cut')
        area_cut = tf.reduce_prod(extent_cut, axis=1, name='area_cut')
        
        area_a = tf.reduce_prod(extent_a, axis=1, name='area_a')
        area_b = tf.reduce_prod(extent_b, axis=1, name='area_b')

        # Calculate area of union
        area_union = area_a + area_b - area_cut

        # Caclulate IoU
        # Note: area_union could be zero if both input boxes have zero extent,
        # hence we add an epsilon to the denominator for stability
        return area_cut / (area_union + epsilon)
