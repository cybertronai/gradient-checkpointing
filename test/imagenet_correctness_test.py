"""
Test that gradient rewriting still produces mostly the same gradients (small numerical differences are expected because order of computation is affected)
"""

import os, sys
os.environ['TF_CUDNN_USE_AUTOTUNE']='0'  # autotune adds random memory spikes
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # suppress init messages

# folder with memory_saving_gradients
module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
#os.sys.path.append(os.path.dirname(sys.argv[0])+'/..')

import tensorflow.contrib.stateless as stateless

import pytest
import math
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
import time

import memory_saving_gradients

import mem_util
import resnet_model   


# remove spammy debug logs
import logging
logger = logging.getLogger('tensorflow')
logger.setLevel(tf.logging.INFO)

pytestmark = pytest.mark.skipif(not tf.test.is_gpu_available(),
                                reason="needs gpu")

# resnet parameters
#HEIGHT = 32
#WIDTH = 32
#DEPTH = 3
#NUM_CLASSES = 10
#BATCH_SIZE=128
_WEIGHT_DECAY = 2e-4

# valid resnet sizes
#  200 # 18, 34 , 50 , 101, 152, 200
BATCH_SIZE=32
RESNET_SIZE=18

USE_TINY = True

HEIGHT=224
WIDTH=224

_INITIAL_LEARNING_RATE = 0.1 * BATCH_SIZE / 128
_MOMENTUM = 0.9

DEPTH = 3
NUM_CLASSES = 1001


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


# debug parameters
DUMP_GRAPHDEF = True

def create_session():
  global sess
  optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
  config = tf.ConfigProto(operation_timeout_in_ms=150000, graph_options=tf.GraphOptions(optimizer_options=optimizer_options))
  sess = tf.Session(config=config)
  return sess

def grads_and_loss():
  """Creates loss tensor for resnet model."""
  images = tf.ones([BATCH_SIZE, HEIGHT, WIDTH, DEPTH])/1000
  labels = tf.ones(shape=[BATCH_SIZE, NUM_CLASSES])/1000
                                              
  #  images = tf.random_uniform((BATCH_SIZE, HEIGHT, WIDTH, DEPTH), seed=1)
  #  labels = tf.random_uniform((BATCH_SIZE, NUM_CLASSES), seed=1)
  if USE_TINY:
    network = resnet_model.tiny_resnet_v2(resnet_size=RESNET_SIZE, num_classes=NUM_CLASSES)
  else:
    network = resnet_model.resnet_v2(resnet_size=RESNET_SIZE,
                                     num_classes=NUM_CLASSES)

    
  inputs = tf.reshape(images, [BATCH_SIZE, HEIGHT, WIDTH, DEPTH])
  logits = network(inputs,True)
  cross_entropy = tf.losses.softmax_cross_entropy(logits=logits,
                                                  onehot_labels=labels)

  loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
    [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
  
  global_step = tf.train.get_or_create_global_step()
  optimizer = tf.train.MomentumOptimizer(
    learning_rate=_INITIAL_LEARNING_RATE,
    momentum=_MOMENTUM)

  # Batch norm requires update_ops to be added as a train_op dependency.
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    grads = tf.gradients(loss, tf.trainable_variables())
    # TODO: move to train_op
    # train_op = optimizer.minimize(loss, global_step)
  return grads, loss


def run_grads():
  """Runs optimization for few steps, returns loss."""
  tf.reset_default_graph()
  tf.set_random_seed(1)
  
  grads, loss = grads_and_loss()

  g = tf.get_default_graph()
  ops = g.get_operations()
  
  for op in ge.filter_ops_from_regex(ops, "block_layer"):
    tf.add_to_collection("remember", op.outputs[0])

  sess = create_session()
  sess.run(tf.global_variables_initializer())
  return sess.run(grads)[0]

def test():
  old_gradients = tf.gradients

  if len(sys.argv)>1:
    mode = sys.argv[1]

  def gradients_auto(ys, xs, grad_ys=None, **kwargs):
    return memory_saving_gradients.gradients(ys, xs, grad_ys,
                                             checkpoints='memory', **kwargs)

  tf.__dict__["gradients"] = gradients_auto
  print("Running with automatically selected checkpoints")
  grads1 = run_grads()
  
  # restore old gradients
  tf.__dict__["gradients"] = old_gradients
  print("Running without checkpoints")
  grads2 = run_grads()

  assert np.max(np.abs(grads1-grads2)) < 1e-6
  
  print("Test passed")

if __name__=='__main__':
  test()

  
