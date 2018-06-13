import tensorflow as tf
import numpy as np


def evaluate(
    session,
    inputs,
    dataset,
    batch_size,
    train_step=None,
    feed_dict={},
    metrics=[],
    report_every=None,
    report_hook=None):
    """Performs one epoch of training.
    
    inputs: tensors that will be fed from the dataset in that exact order
    feed_dict: additional feed during training (eg. step_size)
    metrics: tensors to evaluate
    report_hook: callback function, will be fed (session, iteration, iterations, eval metrics)
    """
    
    metric_evaluations = [[] for _ in metrics]
    iterations = len(dataset) // batch_size

    for iteration, batch_inputs in enumerate(dataset.batch(batch_size)):

        # Run a training step with the current batch
        feed = {**dict(zip(inputs, batch_inputs)), **feed_dict}

        if train_step:
            session.run(train_step, feed_dict=feed)
        
        # Evaluate metrics
        results = session.run(metrics, feed_dict=feed)

        for index, result in enumerate(results):
            metric_evaluations[index].append(result)

        # Print progress report
        if report_every and report_hook:
            if (iteration + 1) % report_every == 0:

                print(
                    report_hook(session, iteration + 1, iterations, *results),
                    end=''
                )

    # Return avg metrics
    if metrics:
        return [np.mean(metric) for metric in metric_evaluations]
