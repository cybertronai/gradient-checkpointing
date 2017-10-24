"""Resnet test that uses new API.

Expected result

Running with automatically selected checkpoints
Calling memsaving gradients with memory
Graph construction: 1785.14 ms
Compute time: 414.17 ms
Memory used: 567.97 MB 
Running without checkpoints
Graph construction: 1234.51 ms
Compute time: 365.91 ms
Memory used: 1110.45 MB 
"""

import os
os.environ['TF_CUDNN_USE_AUTOTUNE']='0'  # autotune adds random memory spikes

import math
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
import time

assert os.getcwd().endswith("/test"), "must run from 'test' directory"
sys.path.extend([".."])   # folder with memory_saving_gradients
import memory_saving_gradients

import resnet_model   

# resnet parameters
#HEIGHT = 32
#WIDTH = 32
#DEPTH = 3
#NUM_CLASSES = 10
#BATCH_SIZE=128
_WEIGHT_DECAY = 2e-4

BATCH_SIZE=32
RESNET_SIZE=18 #  200 # 18, 34 , 50 , 101, 152, 200
RESNET_SIZE=34 #  200 # 18, 34 , 50 , 101, 152, 200

# following sizes don't work
# RESNET_SIZE=50 # unable to find #  200 # 18, 34 , 50 , 101, 152, 200

  
HEIGHT=224
WIDTH=224

_INITIAL_LEARNING_RATE = 0.1 * BATCH_SIZE / 128
_MOMENTUM = 0.9

DEPTH = 3
NUM_CLASSES = 1001

# debug parameters
DUMP_GRAPHDEF = True

def create_session():
  optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
  config = tf.ConfigProto(operation_timeout_in_ms=150000, graph_options=tf.GraphOptions(optimizer_options=optimizer_options))
  return tf.Session(config=config)

def create_train_op_and_loss():
  """Creates loss tensor for resnet model."""
  images = tf.random_uniform((BATCH_SIZE, HEIGHT, WIDTH, DEPTH))
  labels = tf.random_uniform((BATCH_SIZE, NUM_CLASSES))
  network = resnet_model.resnet_v2(resnet_size=RESNET_SIZE,
                                   num_classes=NUM_CLASSES)
  
  inputs = tf.reshape(images, [BATCH_SIZE, HEIGHT, WIDTH, DEPTH])
  logits = network(inputs,False)
  cross_entropy = tf.losses.softmax_cross_entropy(logits=logits,
                                                  onehot_labels=labels)

  #  loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
  #      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
  loss = cross_entropy
  global_step = tf.train.get_or_create_global_step()
  #  optimizer = tf.train.MomentumOptimizer(
  #        learning_rate=_INITIAL_LEARNING_RATE,
  #        momentum=_MOMENTUM)

  # Batch norm requires update_ops to be added as a train_op dependency.
  #  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  #  with tf.control_dependencies(update_ops):

  if DUMP_GRAPHDEF:
    open('imagenet_%d.pbtxt'%(RESNET_SIZE,), 'w').write(str(tf.get_default_graph().as_graph_def()))

  grads = tf.gradients(loss, tf.trainable_variables())
    #train_op = optimizer.minimize(loss, global_step)
    #    grads_and_vars = list(zip(grads, tf.trainable_variables()))
    #    train_op = optimizer.apply_gradients(grads_and_vars)

    #  return train_op, loss
  return grads, loss


def gradient_memory_test():
  """Evaluates gradient, prints peak memory."""
  start_time0 = time.perf_counter()
  start_time = start_time0
  train_op, loss = create_train_op_and_loss()
  print("Graph construction: %.2f ms" %(1000*(time.perf_counter()-start_time)))


  # use block_layer1, block_layer2, block_layer3 as remember nodes
  g = tf.get_default_graph()
  ops = g.get_operations()
  for op in ge.filter_ops_from_regex(ops, "block_layer"):
    tf.add_to_collection("remember", op.outputs[0])

  sess = create_session()
  sess.run(tf.global_variables_initializer())
  start_time = time.perf_counter()
  sess.run(train_op)
  start_time = time.perf_counter()
  sess.run(train_op)
  loss0 = sess.run(loss)
  print("Compute time: %.2f ms" %(1000*(time.perf_counter()-start_time)))

  mem_op = tf.contrib.memory_stats.MaxBytesInUse()
  mem_use = sess.run(mem_op)/1e6
  print("Memory used: %.2f MB "%(mem_use))
  total_time = time.perf_counter()-start_time0
  assert total_time < 100
  return mem_use


if __name__=='__main__':
  assert tf.test.is_gpu_available(), "Memory tracking only works on GPU"
  old_gradients = tf.gradients

  if len(sys.argv)>1:
    mode = sys.argv[1]

  # automatic checkpoint selection
  def gradients_auto(ys, xs, grad_ys=None, **kwargs):
    return memory_saving_gradients.gradients(ys, xs, grad_ys,
                                             remember='memory', **kwargs)

  tf.__dict__["gradients"] = gradients_auto
  print("Running with automatically selected checkpoints")
  gradient_memory_test() < 590
    
  # restore old gradients
  tf.__dict__["gradients"] = old_gradients
  tf.reset_default_graph()
  print("Running without checkpoints")
  gradient_memory_test() < 1200

  print("Test passed")
