import tensorflow as tf

from caps import layers
from caps.utils import norm
from yolophem import utils


config_v1 = {
    'image_size': 448,
    'num_cells': 7,
    'num_predictors': 3,

    # 448x448 -> 221x221
    'conv': [
        {
            'filters': 256,
            'kernel_size': 7,
            'strides': 2,
            'padding': 'VALID',
            'activation': tf.nn.relu,
        }
    ],

    # 221x221 -> 221x221
    'primaryCaps': {
        'filters': 32,
        'dims': 8,
        'kernel_size': 1,
        'strides': 1,
    },

    'convCaps': [
        
        # 221x221 -> 109x109
        {
            'filters': 16,
            'dims': 16,
            'kernel_size': 5,
            'strides': 2
        },
        
        # 109x109 -> 53x53
        {
            'filters': 16,
            'dims': 16,
            'kernel_size': 5,
            'strides': 2
        },
        
        # 53x53 -> 25x25
        {
            'filters': 8,
            'dims': 24,
            'kernel_size': 5,
            'strides': 2
        },
        
        # 25x25 -> 11x11
        {
            'filters': 8,
            'dims': 24,
            'kernel_size': 5,
            'strides': 2
        },
        
        # 11x11 -> 9x9
        {
            'filters': 3,
            'dims': 32,
            'kernel_size': 3,
            'strides': 1
        },
        
        # 9x9 -> 7x7
        {
            'filters': 3,
            'dims': 32,
            'kernel_size': 3,
            'strides': 1
        },
    ]
}

def adapted(config):
    
    image_size = config['image_size']
    num_cells = config['num_cells']
    num_predictors = config['num_predictors']

    ## INPUTS
    X, y = _inputs(config)

    ## NETWORK
    network_output = _network(X, config)
    _, height, width, filters, dims = network_output.shape.as_list()

    ## PREDICTIONS
    network_output_flat = tf.reshape(
        network_output,
        [-1, 1, height * width * filters, dims],
        name='network_output_flat'
    )

    predictions_flat = tf.layers.conv2d(
        network_output_flat,
        filters=4,
        kernel_size=1,
        strides=1,
        padding='SAME',
        activation=None,
        name='predictions_flat'
    )

    predictions = tf.reshape(
        predictions_flat,
        [-1, num_cells, num_cells, num_predictors, 4],
        name='predictions'
    )

    ## CONFIDENCES
    confidences = norm(network_output)

    ## LOSS
    loss = yolophem_loss(predictions, confidences, y, image_size, name='loss')

    ## OUTPUTS
    outputs = _outputs(predictions, confidences, config)
    
    # Inputs, Outputs, loss
    return [X, y], outputs, loss


def naive(config, feature_size):

    image_size = config['image_size']
    num_cells = config['num_cells']
    num_predictors = config['num_predictors']

    ## INPUTS
    X, y = _inputs(config)

    ## NETWORK
    network_output = _network(X, config)
    _, height, width, filters, dims = network_output.shape.as_list()

    ## FEATURES
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

    ## PREDICTIONS
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

    ## CONFIDENCES
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

    ## OUTPUTS
    outputs = _outputs(predictions, confidences, config)
    
    # Inputs, Outputs, loss
    return [X, y], outputs, loss


def _inputs(config):

    X = tf.placeholder(
        shape=[None, config['image_size'], config['image_size'], 3], 
        dtype=tf.float32, 
        name='X'
    )

    y = tf.placeholder(shape=[None, 5], dtype=tf.int64, name='y')

    return X, y


def _network(inputs, config):
    
    # Convolutional layers
    for idx, conf in enumerate(config['conv']):
        inputs = tf.layers.conv2d(
            inputs, 
            **conf, 
            name='conv' + str(idx + 1)
        )
        
        print(inputs)

    # Primary capsules
    inputs = layers.primaryCaps(
        inputs,
        **config['primaryCaps'],
        name='primaryCaps'
    )

    print(inputs)

    # Convolutional capsules
    for idx, conf in enumerate(config['convCaps']):
        inputs = layers.convCaps(
            inputs,
            **conf,
            name='convCaps' + str(idx + 1)
        )

        print(inputs)

    return inputs


# predictions (b_s, S, S, B, 4)
# confidences (b_s, S, S, B)
def _outputs(predictions, confidences, config):

    image_size = config['image_size']
    num_cells = config['num_cells']
    num_predictors = config['num_predictors']
    
    predictions_globalized = utils.uncenter(
        utils.globalize(predictions, image_size),
        name='predictions_globalized'
    )

    predictions_out = tf.reshape(
        predictions_globalized,
        [-1, num_cells * num_cells * num_predictors, 4],
        name='predictions_out'
    )

    confidences_out = tf.reshape(
        confidences,
        [-1, num_cells * num_cells * num_predictors],
        name='confidences_out'
    )

    return predictions_out, confidences_out


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
