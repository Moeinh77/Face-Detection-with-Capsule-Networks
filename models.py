import tensorflow as tf
import numpy as np

def epoch(
    session, 
    inputs, 
    train_step, 
    dataset,
    batch_size,
    feed_dict={}, 
    metrics=[],
    print_every=None
    report=None
    report_formatter=lambda x: x):
    # Performs one epoch of training

    metrics_log = [[] for _ in metrics]
    
    iterations = len(dataset) // batch_size

    for iteration, batch_inputs in enumerate(dataset.batch(batch_size)):
        
        train_feed = {**dict(zip(inputs, batch_inputs)), **feed_dict} 
        
        # Run a trianing step
        session.run(train_step, feed_dict=train_feed)

        # Evaluate metrics
        evaluations = session.run(metrics, feed_dict=train_feed)
            
        for index, evaluation in enumerate(evaluations):
            metrics_log[index].append(evaluation)

        # Report progress
        if print_every and (iteration + 1) % print_every == 0:
            msg = '\rTrain | Iteration {}/{} ({:.1f}%) | '.format(
                iteration + 1, 
                iterations,
                (iteration + 1) * 100 / iterations
            )

            if report:
                report_args = report_formatter(evaluations)
                msg += report.format(report_args)

            print(msg, end="")

        return [np.mean(log) for log in metrics_log]


def validate(session, feed_dict, metrics):
    return session.run(metrics, feed_dict=feed_dict)


