import tensorflow as tf

from yolophem import utils


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
