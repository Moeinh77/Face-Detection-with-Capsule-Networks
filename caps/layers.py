import tensorflow as tf


def primaryCaps(inputs, caps, dims, kernel_size, strides=1, name=None):
    """Primary capsule layer
    
    TODO: expand documentation
    """
    with tf.variable_scope(name, default_name='primaryCaps'):

        # Assert that input is a tensor of feature maps
        tf.assert_rank(
            inputs,
            rank=4,
            message='''`inputs` must be a tensor of feature maps (i.e. of shape
                    (batch_size, height, width, filters))'''
        )

        conv = tf.layers.conv2d(
            inputs,
            filters=caps*dims,
            kernel_size=kernel_size,
            strides=strides,
            padding='VALID',
            activation=tf.nn.relu,
            name='conv'
        )

        # Convert to capsules by reshaping and applying the squash nonlinearity
        _, height, width, _ = conv.shape

        capsules = tf.reshape(
            conv, 
            [-1, height, width, caps, dims], 
            name='capsules'
        )

        outputs = squash(capsules, name='outputs')

    return outputs


def denseCaps(inputs, caps, dims, iterations=2, name=None):
    """Densely-connected capsule layer.

    TODO: expand documentation
    """
    with tf.variable_scope(name, default_name='denseCaps'):

        # There are two possible inputs to a denseCaps layer:
        #
        # 1. a flat (batch_size, caps_in, dims_in) tensor of capsules; this 
        #   happens when denseCaps layer are stacked on another
        #
        # 2. a (batch_size, height, width, filters, dims_in) tensor of spatial
        #   capsule filters; this happens when a denseCaps layer is stacked on 
        #   a convCaps/primaryCaps layer

        # Assert that the input belongs to either of the two cases
        tf.assert_rank_in(
            inputs,
            ranks=[3, 5],
            message='''`inputs` must either be a flat tensor of capsules (i.e.
                    of shape (batch_size, caps_in, dims_in)) or a tensor of
                    capsule filters (i.e. of shape 
                    (batch_size, height, width, filters, dims))'''
        )

        # Both cases can be dealt with by reshaping the input tensor to a 
        # tensor of 1x1 filters, containing one input capsule each, and
        # performing a 1x1 capsule convolution on them with the number of output 
        # filters being the number of desired output capsules
        
        # Compute the number of input 1x1-filters
        inputs_rank = len(inputs.shape)

        if inputs_rank == 3:
            # Case 1 (flat tensor of capsules):
            _, filters_in, dims_in = inputs.shape.as_list()
            
        elif inputs_rank == 5:
            # Case 2 (tensor of spatial capsule filters) 
            _, height, width, filters, dims_in = inputs.shape.as_list()
            filters_in = height * width * filters

        # Reshape input tensor to tensor of 1x1-filters
        inputs_filters = tf.reshape(
            inputs,
            [-1, 1, 1, filters_in, dims_in],
            name='inputs_filters'
        )

        # Perform the 1x1 capsule convolution on the reshaped input tensor
        # This yields a (batch_size, 1, 1, caps, dims) shaped output tensor
        # of 1x1 filters containing one output capsule each
        outputs = _caps_conv2d(
            inputs_filters, 
            filters_out=caps, 
            dims_out=dims, 
            rf_sizes=(1, 1), 
            rf_strides=(1, 1), 
            iterations=iterations
        )

        # Reshape to a flat tensor of output capsules
        outputs = tf.reshape(
            outputs,
            [-1, caps, dims],
            name='outputs'
        )

    return outputs


def denseDecoder(inputs, indices, layers, name=None):
    """Reconstructs inputs from capsule activations"""

    with tf.variable_scope(name, default_name='denseDecoder'):

        _, caps, dims = inputs.shape

        # Convert `indices` to a binary mask of shape (batch_size, caps, 1)
        # to select inputs for reconstruction

        indices_one_hot = tf.one_hot(
            indices,
            depth=caps,
            name='indices_one_hot'
        )

        reconstruction_mask = tf.reshape(
            indices_one_hot,
            shape=[-1, caps, 1],
            name='reconstruction_mask'
        )

        # Apply the mask
        inputs_masked = tf.multiply(
            inputs,
            reconstruction_mask,
            name='inputs_masked'
        )

        # Flatten input for the decoder network
        layer_in = tf.reshape(
            inputs_masked,
            [-1, caps * dims],
            name='decoder_input'
        )

        # Create the hidden layers of the decoder network
        for layer, units in enumerate(layers[:-1]):
            layer_out = tf.layers.dense(
                layer_in,
                units=units,
                activation=tf.nn.relu,
                name='hidden{}'.format(layer + 1)
            )

            layer_in = layer_out

        # Create the ouptut layer of the decoder network
        outputs = tf.layers.dense(
            layer_in,
            units=layers[-1],
            activation=tf.nn.sigmoid,
            name='decoder_output'
        )

        return outputs


def convCaps(
        inputs, 
        filters, 
        dims, 
        rf_size, 
        rf_stride, 
        padding='VALID',
        iterations=2, 
        name=None):
    """2D convolutional capsule layer.

    TODO: expand documentation

    Arguments:
    inputs -- a tensor of shape (batch_size, width, height, filters_in, dims_in)
        containing filters of lower-level capsules.
    filters -- number of output filters in this layer
    dims -- dimensionality of output capsules in this layer
    rf_size -- size of the receptive field of a higher-level capsule, similar
        to the kernel size in a convolutional neural layer
    rf_sride -- stride of the receptive field in the lower-level layer, similar
        to the stride in a convolutional neural layer
    iterations -- number of routing iterations
    """
    with tf.variable_scope(name, default_name='convCaps'):

        tf.assert_rank(
            inputs,
            rank=5,
            message='''`inputs` must be a tensor of capsule filteres, i.e. of 
                    shape (batch_size, height, width, filteres, dims)'''
        )

        outputs = _caps_conv2d(
            inputs, 
            filters, 
            dims, 
            (rf_size, rf_size),
            (rf_stride, rf_stride),
            padding,
            iterations
        )

    return outputs

# Proto-Documentation:
# inputs = input tensor (batch_size, height, width, input_filters, input_dim)
# filters = no. of convolutional filter banks in this convCaps layer
# dims = dimensionality of capsules within each filter
# kernel_size: size of receptive field of output capsules
# stride: stride of receptive field of output capsules
# iterations: number of routing iterations
def _caps_conv2d(
        inputs, 
        filters_out, 
        dims_out, 
        rf_sizes, 
        rf_strides, 
        padding,
        iterations):


    # In a convolutional capsule layer, the lower-level capsule filters are 
    # spatially decomposed into receptvie fields, ie. overlapping patches 
    # of fixed size. For each receptive field, each capsule i in the patch 
    # predicts the output of one higher-level capsule j per output filter. This
    # is done by multiplying capsule i with a learned transformation matrix
    # W_ij. The transformation matrices for a particular output filter are
    # shared for all patches of lower-level capsules.


    # Compute predictions of higher-level outputs
    # This returns a tensor of shape
    # (batch_size, H, W, filters_out, caps_per_patch, dims_out, 1),
    # where H, W are the height and width of a capsule filter, respectively.
    # This can be thought of as a (batch_size, H, W, filters_out)-shaped
    # volume containing linearized patches with `caps_per_patch` 
    # `dims_out`-dimensional column vectors.
    predictions = _predict(
        inputs, 
        filters_out, 
        dims_out, 
        rf_sizes, 
        rf_strides,
        padding
    )
        
    # Compute higher-level capsule outputs from lower-level predictions via
    # "routing by agreement".
    # The result of this operation is a tensor of shape
    # (batch_size, H, W, filters_out, dims_out). 
    outputs = _route(predictions, iterations)
    
    return outputs


def _predict(inputs, filters_out, dims_out, rf_sizes, rf_strides, padding):
    """Compute predictions for higher-level capsules on receptive fields"""
    with tf.variable_scope('predictions'):

        # TODO improve documentation

        # Extract shape of the input tensor
        batch_size = tf.shape(inputs)[0]
        _, height_in, width_in, filters_in, dims_in = inputs.shape.as_list()

        # Concatenate capsule activations accross all input filters
        inputs_flat = tf.reshape(
            inputs, 
            shape=[-1, height_in, width_in, filters_in * dims_in],
            name='inputs_flat'
        )

        # Decompose input capsules tensor into potentially overlapping, 
        # flattened patches
        inputs_patches = tf.extract_image_patches(
            inputs_flat,
            ksizes=[1, rf_sizes[0], rf_sizes[1], 1],
            strides=[1, rf_strides[0], rf_strides[1], 1],
            rates=[1,1,1,1],
            padding=padding
        )

        # Extract shape of output tensor
        _, height_out, width_out, _ = inputs_patches.shape.as_list()

        # number of capsule activations per patch
        caps_per_patch = rf_sizes[0] * rf_sizes[1] * filters_in

        # Reshape inputs patches into an (batch_size, height, width) array of 
        # `caps_per_patch` `dims_in-dimensional` column vectors. To explicitly 
        # represent the vectors as column vectors, we add the trailing 
        # dimension. The other dummy dimension will later serve for tiling 
        # along the `out_filters` axis
        inputs_patches = tf.reshape(
            inputs_patches,
            shape=[-1, height_out, width_out, 1, caps_per_patch, dims_in, 1],
            name='inputs_patches'
        )

        # Replicate array of input capsules along the `out_filters` axis
        inputs_patches_tiled = tf.tile(
            inputs_patches,
            multiples=[1, 1, 1, filters_out, 1, 1, 1],
            name='inputs_patches_tiled'
        )


        # Create a (output_caps, kernel_size * kernel_size * input_caps) array 
        # of (output_dims x input_dims) transformation matrices. This set of 
        # matrices is shared among all input patches. 
        W = tf.Variable(
            tf.random_normal(
                shape=[1, 1, 1, filters_out, caps_per_patch, dims_out, dims_in],
                stddev=0.1,
                dtype=tf.float32
            ),
            name='W'
        )


        # The three additional dummy dimensions in the 'front' of the tensor's
        # shape are used to replicate the shared transformation matrices
        # accross all receptive fields in all samples in the batch.
    
        W_tiled = tf.tile(
            W, 
            multiples=[batch_size, height_out, width_out, 1, 1, 1, 1], 
            name='W_tiled'
        )

        # Matrix multiplication leaves us with a tensor of shape identical to 
        # `inputs_patches_tiled`, only that capsule activations are now 
        # `dims_out` dimensional, ie.
        # (batch_size, height, width, filters_out, caps_per_patch, dims_out, 1)
        predictions = tf.matmul(
            W_tiled,
            inputs_patches_tiled,
            name='predictions'
        )

    return predictions


def _route(predictions, iterations):
    with tf.variable_scope('routing'):

        # TODO improve documentation

        batch_size = tf.shape(predictions)[0]
        _, height, width, filters_out, caps_per_patch, dims_out, _ = predictions.shape.as_list()

        initial_outputs = tf.zeros(
            [batch_size, height, width, filters_out, 1, dims_out, 1],
            name='initial_outputs'
        )

        logits = tf.zeros(
            [batch_size, height, width, filters_out, caps_per_patch, 1, 1],
            name='logits'
        )


        def _routing_iteration(loop_counter, logits_old, outputs_old):
            with tf.variable_scope('routing_iteration'):

                ## TOP-DOWN FEEDBACK

                # Replicate outputs from previous iteration accross 
                # `caps_per_patch` dimension
                outputs_old_tiled = tf.tile(
                    outputs_old,
                    multiples=[1, 1, 1, 1, caps_per_patch, 1, 1],
                    name='outputs_old_tiled'
                )

                agreement = tf.matmul(
                    predictions,
                    outputs_old_tiled,
                    transpose_a=True,
                    name='agreement'
                )

                logits = tf.add(
                    logits_old,
                    agreement,
                    name='logits'
                )

                
                ## BOTTOM-UP ROUTING

                coupling_coefficients = tf.nn.softmax(
                    logits,
                    axis=3, # filters_out
                    name='coupling_coefficients'
                )

                weighted_predictions = tf.multiply(
                    coupling_coefficients,
                    predictions,
                    name='weighted_predictions'
                )

                centroids = tf.reduce_sum(
                    weighted_predictions,
                    axis=4, # caps_per_patch,
                    keepdims=True,
                    name='centroids'
                )

                outputs = squash(centroids, axis=-2, name='outputs')

            return loop_counter + 1, logits, outputs


        loop_counter = tf.constant(1, name='loop_counter')

        # Run a fixed number of iterations
        _, _, outputs_sparse = tf.while_loop(
            cond=lambda c, l, o: tf.less_equal(c, iterations),
            body=_routing_iteration,
            loop_vars=[loop_counter, logits, initial_outputs],
            name='routing_loop'
        )

        # Drop unnecessary dimensions (caps_per_patch and columns)
        outputs = tf.squeeze(outputs_sparse, axis=[4, -1], name='outputs')

    return outputs


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
