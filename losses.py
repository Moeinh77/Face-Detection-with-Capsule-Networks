from functools import reduce

import tensorflow as tf

import layers
# TODO move layers.norm to utils.norm

def margin_loss(
    labels_one_hot, 
    probabilities, 
    m_plus=0.9, 
    m_minus=0.1, 
    lambda_=0.5, 
    name=None):
    """Implements the margin loss as described in [Hinton, 2017]."""
    # TODO expand documentaiton

    with tf.variable_scope(name, default_name="margin_loss"):
        tf.assert_rank(
            labels_one_hot, 
            rank=2, 
            message="""Expected `labels_one_hot` being a tensor of shape 
                    (batch_size, capsules)"""
        )

        tf.assert_rank(
            probabilities,
            rank=2,
            message="""Expected `probabilities` being a tensor of shape
                    (batch_size, capsules)"""
        )
    
        # The margin loss punishes the deviation of the length of a capsule's 
        # activation vector from an upper margin m_plus (for the capsule 
        # corresponding to the expected output) and from a lower margin m_minus
        # (for the other capsules, respectively). Deviations from the lower
        # margin are down-scaled bby a factor `lambda_` to facilitate training.

        # Compare length of the capsules' activations against the upper margin
        present_error = tf.square(
            tf.maximum(0., m_plus - probabilities), 
            name="present_error"
        )

        # Compare length of the capsules' activations against the lower margin
        absent_error = tf.square(
            tf.maximum(0., probabilities - m_minus),
            name="absent_error"
        )

        # Select the respective error for each capsule
        digit_losses = tf.add(
            labels_one_hot * present_error, 
            lambda_ * (1.0 - labels_one_hot) * absent_error,
            name="digit_losses"
        )

        # Aggregate losses over all capsules within a sample
        losses = tf.reduce_sum(digit_losses, axis=1, name="losses")

        # Average losses over all samples in the batch
        loss = tf.reduce_mean(losses, name="loss")
        return loss


def reconstruction_loss(inputs, reconstructions, name=None):
    with tf.variable_scope(name, default_name='reconstruction_loss'):
        # TODO check input dimensionality
        
        # Flatten inputs
        dim = reduce(lambda x,y: x*y, inputs.shape[1:])
        inputs_flat = tf.reshape(inputs, [-1, dim], name="inputs_flat")

        # The reconstruction loss is the MSE between original and reconstruction
        squared_errors = tf.square(
            inputs_flat - reconstructions, 
            name="squared_errors"
        )

        loss = tf.reduce_mean(squared_errors, name="loss")
        return loss
