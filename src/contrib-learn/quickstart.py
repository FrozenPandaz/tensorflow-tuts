from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Enable Logging
tf.logging.set_verbosity(tf.logging.INFO)

# DATA
IRIS_TRAINING = 'iris/iris_training.csv'
IRIS_TEST = 'iris/iris_test.csv'

# Load
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype = np.int,
    features_dtype = np.float32
)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype = np.int,
    features_dtype = np.float32
)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

validation_metrics = {
    'accuracy': tf.contrib.learn.metric_spec.MetricSpec(
        metric_fn = tf.contrib.metrics.streaming_accuracy,
        prediction_key = tf.contrib.learn.prediction_key.PredictionKey.CLASSES
    ),
    'precision': tf.contrib.learn.metric_spec.MetricSpec(
        metric_fn = tf.contrib.metrics.streaming_precision,
        prediction_key = tf.contrib.learn.prediction_key.PredictionKey.CLASSES
    ),
    'recall': tf.contrib.learn.metric_spec.MetricSpec(
        metric_fn = tf.contrib.metrics.streaming_recall,
        prediction_key = tf.contrib.learn.prediction_key.PredictionKey.CLASSES
    )
}

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    test_set.data,
    test_set.target,
    every_n_steps=50,
    metrics = validation_metrics,
    early_stopping_metric = 'loss',
    early_stopping_metric_minimize = True,
    early_stopping_rounds = 200
)

# DNN Classifier
classifier = tf.contrib.learn.DNNClassifier(
    feature_columns = feature_columns,
    hidden_units = [10, 20, 10],
    n_classes = 3, # 3 types of flowers
    model_dir = 'tmp/iris_model',
    config = tf.contrib.learn.RunConfig(
        save_checkpoints_secs = 1
    )
)

# Fit
classifier.fit(
    x = training_set.data,
    y = training_set.target,
    steps = 2000,
    monitors = [validation_monitor]
)

accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)['accuracy']

print('Accuracy: {0:f}'.format(accuracy_score))

new_samples = np.array([
    [6.4, 3.2, 4.5, 1.5],
    [5.8, 3.1, 5.0, 1.7]
], dtype=float)

y = list(classifier.predict(
    new_samples, as_iterable=True
))

print('Predictions: {}'.format(str(y)))