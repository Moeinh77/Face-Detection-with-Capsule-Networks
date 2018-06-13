import tensorflow as tf
from utils import squash

def convCaps(
        inputs, 
        filters, 
        dimensions, 
        kernel_size, 
        strides=1, 
        activation=None, 
        name=None):
    """A convolutional capsule layer."""
    # TODO expand documentation
    # TODO make convCaps layer stackable

    with tf.variable_scope(name, default_name="convCaps"):

        # Assert that `inputs` is a tensor of feature maps
        tf.assert_rank(
            inputs, 
            rank=4,
            message="""Expected `inputs` being a tensor of feature maps, ie. of 
                    shape (batch_size, height, width, channels)"""
        )
        
        # Perform a 2D convolution

        conv_params = {
            "filters": filters * dimensions,
            "kernel_size": kernel_size,
            "strides": strides,
            "padding": "valid",
            "activation": activation,
        }

        conv = tf.layers.conv2d(inputs, name="conv", **conv_params)

        # Reshape into capsules
        _, height, width, _ = conv.shape
        shape = [-1, height, width, filters, dimensions]

        capsules = tf.reshape(conv, shape=shape, name="capsules")

        # Convert to proper capsule activations by applying the squash 
        # nonlinearity
        out = squash(capsules, name="out")

        return out


def denseCaps(inputs, capsules, dimensions, iterations=2, name=None):
    """A fully-connected capsule layer, with routing."""
    
    with tf.variable_scope(name, default_name="denseCaps"):

        with tf.name_scope("check_shape"):
            # Assert that `inputs` is either a flat tensor of capsules or a 2D
            # tensor of capsule filters.
            tf.assert_rank_in(
                inputs, 
                ranks=[3, 5],
                message="""Expected `inputs` being either a flat tensor of 
                        capsules (ie. shaped 
                        (batch_size, input_caps, input_dims)) or a tensor of 2D 
                        capsule filters (ie. shaped 
                        (batch_size, height, width, filters, dims))""")

            # If inputs are given as a set of capsule filters, reshape to a 
            # flat tensor of capsules.

            # NOTE This is deliberately done at construction time as the input
            # shape only depends on the graph architecture
            
            inputs_rank = len(inputs.shape)

            if inputs_rank == 5:
                _, height, width, filters, input_dims = inputs.shape.as_list()
                
                inputs = tf.reshape(
                    inputs, 
                    [-1, height * width * filters, input_dims]
                )


        # Rename dimensions for clarity
        _, input_caps, input_dims = inputs.shape.as_list()
        output_caps, output_dims = capsules, dimensions

        with tf.name_scope("predictions"):

            # Every input capsule i predicts the output of every higher-level
            # capsule j by multiplying its own vector of activations with a 
            # learned transformation matrix W_ij.

            # To compute this efficiently, we use the higher-order
            # tensor semantics of tf.matmul(): We first create a 
            # (1, input_caps, output_caps) tensor containing the individual 
            # transformation matrices of shape (output_dims, input_dims).

            # TODO make this initializable when converting `denseCaps()` into
            # a class
            # TODO tf.Variable() or tf.get_variable()

            W = tf.Variable(
                tf.random_normal(
                    shape=[1, input_caps, output_caps, output_dims, input_dims],
                    stddev=0.1,
                    dtype=tf.float32
                ),
                name="W",
            )

            # The W_ij matrices are learned parameters and therefore identical
            # for each input sample. Hence, we replicate the array of
            # transformation matrices accross the `batch_size` dimension; this
            # allows us to compute the higher-level predictions for an entire
            # batch of samples at once.

            batch_size = tf.shape(inputs)[0]
            W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

            # This tensor (ie. W_batch) of transformation matrices replicated
            # accross samples will be multiplied with another tensor containing
            # the lower-level activation vectors.

            # To perform element-wise matrix-vector multiplication on these two
            # arrays, we first transpose the lower-level capsule activations
            # to column vectors
            inputs_transposed = tf.expand_dims(
                inputs, 
                axis=-1, 
                name="inputs_transposed"
            )

            # Since every lower-level capsule predicts the activation of every
            # higher-level capsule, we introduce a `output_caps` dimension and 
            # replicate input activations accross this dimension. This yields a
            # (batch_size, input_caps, output_caps) array of input activation
            # (column-) vectors.
            inputs_expanded = tf.expand_dims(
                inputs_transposed,
                axis=2,
                name="inputs_expanded"
            )

            inputs_tiled = tf.tile(
                inputs_expanded, 
                [1, 1, output_caps, 1, 1], 
                name="inputs_tiled"    
            )

            # We now have two higher-dimensional tensors containing the
            # transformation matrices and lower-level capsule activations (as
            # column vectors). tf.matmul() will perform element-wise
            # matrix-vector multiplication on these tensors, producing a 
            # (batch_size, input_caps, output_caps, ouptut_dims, 1)-shaped
            # tensor containing the predicted (output_dims, 1) activation
            # vectors of the higher-level capsules.
            outputs_predictions = tf.matmul(
                W_tiled, 
                inputs_tiled, 
                name="outputs_predictions"
            )

        
        with tf.name_scope("routing"):

            # Higher-level capsules determine their output by performing 
            # coincidence-filtering on the prediction vectors of lower-level
            # capsules. Intuitively, this corresponds to finding dense clusters
            # in the high-dimensional prediction space.

            # This mechanism is implemented in an iterative approach, similar to
            # 1-Nearest-Neighbor clustering: Each lower-level capsule i
            # maintains a set of probabilities c_ij denoting that the object
            # represented by capsule i is a compositional part of an object
            # represented by a higher-level capsule j. These "routing weights" 
            # are updated iteratively by top-down feedback from the higher-level
            # capsules; the closer the prediction vector of capsule i is located
            # to the mean of a cluster in the prediction space of capsule j, 
            # the higher the corresponding routing_weight. The output of the
            # higher-level capsule is then computed as said cluster mean (ie.
            # the mean of "incoming" predictions multiplied by the corresponding
            # routing weights).
            # In that regard, the c_ij's correspond to cluster indicators in 
            # probabilistic variants of the Nearest-Neighbor Clustering 
            #algorithm.
            
            # The c_ij's are not stored directly, but in the form of logits
            # (unscaled log-probabilities); computing the routing weights as 
            # softmax(logits) ensures that the weights outgoing from a 
            # lower-level capsule sum up to one.

            # Create a tensor of logits for each pair of lower-level and higher-
            # level capsule, accross all input samples. The last two dimensions
            # are added to simplify later multiplication of routing weights and
            # prediction vectors (to that end, logits can be viewed as 1D-column
            # vectors)
            logits = tf.zeros(
                [batch_size, input_caps, output_caps, 1, 1], 
                dtype=tf.float32, 
                name="logits"
            )


            # Initial tensor of higher-level capsules outputs; will be updated
            # during the routing.
            # (The second dimension of 1 is an artifact from summing over all
            # incoming predictions weighted by the corresponding routing 
            # weights, it will be discarded below)
            initial_outputs = tf.zeros(
                [batch_size, 1, output_caps, output_dims, 1],
                dtype=tf.float32,
                name="initial_outputs"
            )


            def routing_iteration(counter, logits_old, outputs_old):
                with tf.name_scope("routing_iteration"):
                    
                    # Update logits via top-down feedback

                    # Replicate outputs from previous iteration accross the 
                    # `input_caps` dimension
                    outputs_old_tiled = tf.tile(
                        outputs_old,
                        [1, input_caps, 1, 1, 1],
                        name="outputs_old_tiled"
                    )

                    # Compute proximity of lower-level predictions to cluster
                    # means by taking the scalar products of the respective
                    # pairs of vectors. 
                    agreement = tf.matmul(
                        outputs_predictions, 
                        outputs_old_tiled, 
                        transpose_a=True, 
                        name="agreement"
                    )

                    # Interprete cluster proximities as log probabilities and
                    # use them as deltas to update the routing logits
                    # NOTE This is not formally sound (as Hinton acknowledges)
                    # but is fast and works well enough.
                    logits = tf.add(
                        logits_old, 
                        agreement, 
                        name="logits"
                    )

                    # Update outputs via bottom-up routing
                
                    # Convert logits to routing weights across the `output_caps`
                    # dimension
                    coupling_coefficients = tf.nn.softmax(
                        logits, 
                        axis=2, 
                        name="coupling_coefficients"
                    )
                
                    # Compute cluster means by weighting outgoing prediction
                    # vectors with their corresponding routing weights and then
                    # summing accross the weighted predictions for each
                    # higher-level capsule
                    weighted_predictions = tf.multiply(
                        coupling_coefficients, 
                        outputs_predictions, 
                        name="weighted_predictions"
                    )
            
                    weighted_sums = tf.reduce_sum(
                        weighted_predictions, 
                        axis=1, 
                        keepdims=True, 
                        name="weighted_sums"
                    )

                    # Convert cluster means into proper capsule activations
                    # by applying the squash nonlinearity
                    # TODO why do we apply along this axis? shouldn't it be 2 because columns?
                    outputs = squash(weighted_sums, axis=-2, name="outputs")

                    return tf.add(counter, 1), logits, outputs


            counter = tf.constant(1, name="loop_counter")

            # Run a specified number of routing iterations
            _, _, outputs_sparse = tf.while_loop(
                cond=lambda c, l, o: tf.less_equal(c, iterations), 
                body=routing_iteration, 
                loop_vars=[counter, logits, initial_outputs],
                name="routing_loop"
            )

    
        # Reshape final output to (batch_size, output_caps, output_dims)
        outputs = tf.squeeze(outputs_sparse, axis=[1, -1], name="outputs")
    
    return outputs


def denseDecoder(inputs, indices, layers, name=None):
    """Reconstructs capsule activations using a FCNN."""
    # TODO expand documentation
    # TODO check input dimensions

    with tf.variable_scope(name, default_name='denseDecoder'):

        _, capsules, dims = inputs.shape

        # To select the capsule activations from which inputs should be
        # reconstructed, convert `indices` into a binary mask of shape 
        # (batch_size, capsules, 1).

        indices_one_hot = tf.one_hot(
            indices, 
            depth=capsules, 
            name="indices_one_hot"
        )

        reconstruction_mask = tf.reshape(
            indices_one_hot, 
            shape=[-1, capsules, 1], 
            name="reconstruction_mask"
        )

        # Apply the mask
        inputs_masked = tf.multiply(
            inputs, 
            reconstruction_mask, 
            name="inputs_masked"
        )

        # Flatten input to feed it into the decoder network
        layer_in = tf.reshape(
            inputs_masked, 
            [-1, capsules * dims], 
            name="decoder_input"
        )

        # Create the hidden (ReLU) layers of the decoder network
        for layer, units in enumerate(layers[:-1]):
            layer_out = tf.layers.dense(
                layer_in, 
                units=units, 
                activation=tf.nn.relu, 
                name="hidden{}".format(layer + 1)
            )

            layer_in = layer_out

        # Create the output layer of the decoder network
        outputs = tf.layers.dense(
            layer_in, 
            units=layers[-1], 
            activation=tf.nn.sigmoid, 
            name="decoder_output"
        )

        return outputs
