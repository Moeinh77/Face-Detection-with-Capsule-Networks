import tensorflow as tf

from caps import layers
from yolophem import utils

def simplistic(
    image_size, 
    network_config, 
    feature_size, 
    num_cells, 
    num_predictors):
    
    ## INPUTS 
    X = tf.placeholder(
        shape=[None, image_size, image_size, 3], 
        dtype=tf.float32, 
        name='X'
    )

    y = tf.placeholder(shape=[None, 5], dtype=tf.int64, name='y')


    ## Network
    network_output = _generate_network(X, network_config)
    _, height, width, filters, dims = network_output.shape.as_list()


    network_output_flat = tf.reshape(
        network_output, 
        [-1, height*width*filters*dims],
        name='network_output_flat'
    )
    
    feature_vector = tf.layers.dense(
        network_output_flat,
        feature_size,
        activation=tf.nn.relu,
        name='feature_vector'
    )

    # Predictions
    predictions_flat = tf.layers.dense(
        feature_vector,
        num_cells * num_cells * num_predictors * 4,
        name='predictions_flat'
    )

    predictions = tf.reshape(
        predictions_flat,
        [-1, num_cells, num_cells, num_predictors, 4],
        name='predictions'
    )

    # Confidences
    confidences_flat = tf.layers.dense(
        feature_vector,
        num_cells * num_cells * num_predictors,
        name='confidences_flat'
    )

    confidences = tf.reshape(
        confidences_flat,
        [-1, num_cells, num_cells, num_predictors],
        name='confidences'
    )

    ## LOSS

    loss = yolophem_loss(predictions, confidences, y, image_size, name='loss')


    ## MODEL PREDICTIONS
    centered_outputs, sample_idx = utils.globalize(predictions, image_size)
    outputs = utils.uncenter(centered_outputs)

    return [X, y], [outputs, sample_idx], loss


def _generate_network(inputs, config):
    
    # Convolutional layers
    for idx, conf in enumerate(config['conv']):
        inputs = tf.layers.conv2d(
            inputs, 
            **conf, 
            name='conv' + str(idx + 1)
        )

    # Primary capsules
    inputs = layers.primaryCaps(
        inputs,
        **config['primaryCaps'],
        name='primaryCaps'
    )

    # Convolutional capsules
    for idx, conf in enumerate(config['convCaps']):
        inputs = layers.convCaps(
            inputs,
            **conf,
            name='convCaps' + str(idx + 1)
        )

    return inputs


def yolophem_loss(
    predictions, 
    confidences, 
    labels_long, 
    img_size,
    lambda_box=5,
    lambda_noobj=.5,
    name=None):
    with tf.variable_scope(name, default_name='yolophem_loss'):
        batch_size = tf.cast(tf.shape(predictions)[0], dtype=tf.int64)
        _, num_cells, _, num_predictors, _ = predictions.shape.as_list()

        # Extract sample indices and labels in YOLO format 
        # (i.e. centered coordinates)
        sample_idx, labels = tf.split(labels_long, [1,4], axis=1)
        labels_centered = utils.center(labels, name='labels_centered')

        labels_localized, cell_idx = utils.localize(
            labels_centered,
            img_size,
            num_cells,
            name='labels_localized'
        )

        # Select predictions from cells corresponding to the localized labels
        predictions_selected = tf.gather_nd(
            predictions,
            tf.concat([sample_idx, cell_idx], axis=-1),
            name='predictions_selected'
        )

        # Replicate labels for each predictor
        labels_localized_tiled = tf.tile(
            tf.reshape(labels_localized, [-1, 1, 4]),
            multiples=[1, num_predictors, 1],
            name='labels_localized_tiled'
        )

        # Compute IoUs between predictions (across all predictors) and labels
        prediction_ious = utils.intersection_over_union(
            predictions_selected,
            labels_localized_tiled,
            name='prediction_ious'
        )

        # Compute indices of the responsible predictors for each label
        predictor_idx = tf.argmax(
            prediction_ious, 
            axis=-1, 
            name='predictor_idx'
        )

        # Piece indices together
        targets_idx = tf.concat(
            [sample_idx, cell_idx, tf.reshape(predictor_idx, shape=[-1, 1])],
            axis=-1,
            name='targets_idx'
        )

        # Select labels in positions of responsible predictors to create the 
        # final target tensor
        targets_shape = tf.stack(
            [batch_size, num_cells, num_cells, num_predictors, 4],
            name='targets_shape'
        )

        targets = tf.scatter_nd(
            targets_idx, 
            labels_localized,
            shape=targets_shape,
            name='targets'
        )
        
        # Create a binary mask indicating target positions
        mask_shape = tf.stack(
            [batch_size, num_cells, num_cells, num_predictors, 1],
            name='mask_shape'
        )

        predictions_mask = tf.scatter_nd(
            targets_idx,
            tf.ones_like(sample_idx, dtype=tf.float32),
            shape=mask_shape,
            name='predictions_mask'
        )

        # Loss incurred by box predictions
        box_loss = tf.reduce_sum(
            predictions_mask * ((predictions - targets) ** 2),
            name='box_loss'
        )

        # Loss incurred by confidence corresponding to 'responsible' predictions
        # (should equal IoU)
        confidence_mask = tf.reshape(
            predictions_mask, 
            [-1, num_cells, num_cells, num_predictors],
            name='confidence_mask'
        )

        target_ious = utils.intersection_over_union(
            predictions,
            targets,
            name='target_ious'
        )

        responsible_confidence_loss = tf.reduce_sum(
            confidence_mask * ((confidences - target_ious) ** 2),
            name='responsible_confidence_loss'
        )

        # Loss incurred by confidence corresponding to non-'responsible'
        # predictions (should equal the confidence)
        non_responsible_confidence_loss = tf.reduce_sum(
            (1 - confidence_mask) * (confidences ** 2),
            name='non_responsible_confidence_loss'
        )

        loss = lambda_box * box_loss + \
                responsible_confidence_loss + \
                lambda_noobj * non_responsible_confidence_loss

    return loss
