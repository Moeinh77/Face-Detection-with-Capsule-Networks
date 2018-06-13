import os
import pickle
from datetime import datetime

import tensorflow as tf
import numpy as np

def reset(seed=42):
    tf.set_random_seed(seed)
    np.random.seed(seed)

    tf.reset_default_graph()

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


def new_experiment(root_logdir='./experiments', experiment=None, force=False):
    logdir = os.path.join(
        root_logdir,
        experiment or datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
    )

    if os.path.isdir(logdir):
        if force:
            shutil.rmtree(logdir)
        else:
            raise ValueError(
                'Log directory already exists. Use force=True to overwrite'
            )

    os.makedirs(logdir, exist_ok=True)

    return logdir


class Logger:

    def __init__(self, logdir):
        self.log_file = os.path.join(logdir, 'log.pickle')
        self.log = {}

        if os.path.isfile(self.log_file):
            self.log = pickle.load(open(self.log_file, 'rb'))


    def add(self, key, value, timestamp):
        if not key in self.log:
            self.log[key] = []

        self.log[key].append((timestamp, value))


    def get(self, key):
        return self.log[key]


    def write(self):
        pickle.dump(self.log, open(self.log_file, 'wb'))

