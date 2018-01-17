# correctness test
# MNIST model taken from github/tensorflow/models

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# import memory_saving_gradients from ..
module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import memory_saving_gradients
import mem_util

TEST_DEVICE='/cpu:0'
USE_REAL_DATA = False
FLAGS_data_dir='/tmp/mnist_data'
FLAGS_model_dir='/tmp/mnist_model'
FLAGS_batch_size=1
FLAGS_data_format=None

# from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

  
def train_dataset(data_dir):
  """Returns a tf.data.Dataset yielding (image, label) pairs for training."""
  data = input_data.read_data_sets(data_dir, one_hot=True).train
  return tf.data.Dataset.from_tensor_slices((data.images, data.labels))


def mnist_model(inputs, mode, data_format):
  """Takes the MNIST inputs and mode and outputs a tensor of logits."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  inputs = tf.reshape(inputs, [-1, 28, 28, 1])

  if data_format is None:
    # When running on GPU, transpose the data from channels_last (NHWC) to
    # channels_first (NCHW) to improve performance.
    # See https://www.tensorflow.org/performance/performance_guide#data_formats
    data_format = ('channels_first'
                   if tf.test.is_gpu_available() else 'channels_last')

  if data_format == 'channels_first':
    inputs = tf.transpose(inputs, [0, 3, 1, 2])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=inputs,
      filters=32,
      kernel_size=[5, 5],
      padding='same',
      activation=tf.nn.relu,
      data_format=data_format)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(
      inputs=conv1, pool_size=[2, 2], strides=2, data_format=data_format)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding='same',
      activation=tf.nn.relu,
      data_format=data_format)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(
      inputs=conv2, pool_size=[2, 2], strides=2, data_format=data_format)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)
  return logits

GLOBAL_PROFILE = True
DUMP_TIMELINES = False
run_metadata = True
def sessrun(*args, **kwargs):
  global sess, run_metadata
  
  if not GLOBAL_PROFILE:
    return sess.run(*args, **kwargs)
  
  run_metadata = tf.RunMetadata()

  kwargs['options'] = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  kwargs['run_metadata'] = run_metadata
  result = sess.run(*args, **kwargs)
  first_entry = args[0]
  if isinstance(first_entry, list):
    if len(first_entry) == 0 and len(args) == 1:
      return None
    first_entry = first_entry[0]

  if DUMP_TIMELINES:
    name = first_entry.name
    name = name.replace('/', '-')

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timelines/%s.json'%(name,), 'w') as f:
      f.write(ctf)
    with open('timelines/%s.pbtxt'%(name,), 'w') as f:
      f.write(str(run_metadata))

  return result


def create_session():
  from tensorflow.core.protobuf import rewriter_config_pb2
  optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
  config = tf.ConfigProto(operation_timeout_in_ms=150000, graph_options=tf.GraphOptions(optimizer_options=optimizer_options))
  config.graph_options.rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
  config.graph_options.place_pruned_graph = True
  return tf.Session(config=config)


sess = None
def train_mnist():
  global sess

  # restrict to cpu:0
  tf.reset_default_graph()
  tf.set_random_seed(1)
  np.random.seed(1)
  tf_dev = tf.device(TEST_DEVICE)
  tf_dev.__enter__()

  #  FLAGS = parse_flags()
  # Train the model


  # replace Dataset ops with constant images because gradient rewriting
  # tries to differentiate graphs containing IteratorGetNext
  # TODO: make it work with Dataset ops
  images = tf.Variable(tf.random_uniform((FLAGS_batch_size, 28**2)))
  labels = tf.Variable(tf.concat([tf.ones((FLAGS_batch_size, 1)),
                                  tf.zeros((FLAGS_batch_size, 9))], axis=1))
  def train_input_fn():
    dataset = train_dataset(FLAGS_data_dir)
    dataset = dataset.batch(FLAGS_batch_size)
    (images, labels) = dataset.make_one_shot_iterator().get_next()
    num_images = FLAGS_batch_size
    return (images[:num_images], labels[:num_images])

  if USE_REAL_DATA:
    images, labels = train_input_fn()
#    images = tf.stop_gradient(images)
#    labels = tf.stop_gradient(labels)

  
  logits = mnist_model(images, tf.estimator.ModeKeys.TRAIN, 'channels_last')
  cross_entropy = tf.losses.softmax_cross_entropy(logits=logits,
                                                  onehot_labels=labels)
  loss = cross_entropy
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)

  vars = tf.trainable_variables()
  grads = tf.gradients(loss, vars)
  grads_and_vars = zip(grads, vars)
  train_op = optimizer.apply_gradients(grads_and_vars)
  
  sess = create_session()
  sess.run(tf.global_variables_initializer())
  print("Loss %.5f" %(sess.run(loss)))
  for i in range(10):
    sessrun(train_op)
    mem_use = mem_util.peak_memory(run_metadata)[TEST_DEVICE]/1e6
    print("Loss %.5f, memory %.2f MB" %(sess.run(loss), mem_use))

  # should print something like this for actual dataset
  # 2.12764
  # 1.87759
  # 1.54445
  # 1.29149
  # 1.18474
  # 0.884424
  # 0.69454
  # 0.770236
  # 0.629259
  # 0.654465

  assert sess.run(loss) < 100


def test_correctness(capsys):
  # enable printing during successful test run under pytest, uncomment these
  # if capsys:
  #   pytest_decorator = capsys.disabled()
  #   pytest_decorator.__enter__()

  # Loss 0.01803, memory 399.10 MB
  # Loss 0.00002, memory 399.10 MB
  # Loss 0.00000, memory 399.10 MB
  # Running with memory saving
  # Extracting /tmp/mnist_data/train-images-idx3-ubyte.gz
  # Extracting /tmp/mnist_data/train-labels-idx1-ubyte.gz
  # Extracting /tmp/mnist_data/t10k-images-idx3-ubyte.gz
  # Extracting /tmp/mnist_data/t10k-labels-idx1-ubyte.gz
  # Loss 0.07283, memory 380.72 MB
  # Loss 0.00398, memory 351.23 MB
  # Loss 0.00035, memory 351.23 MB

  def grads(ys, xs, grad_ys=None, **kwargs):
    return memory_saving_gradients.gradients(ys, xs, grad_ys,
                                             checkpoints='speed', **kwargs)
  old_grads = tf.gradients
  tf.__dict__["gradients"] = grads
  print("Running with memory saving")
  
  train_mnist()

  print("\nRunning with regular gradient")
  tf.__dict__["gradients"] = old_grads
  train_mnist()

def main(unused_argv):
  test_correctness(None)

if __name__ == '__main__':
  main([])
