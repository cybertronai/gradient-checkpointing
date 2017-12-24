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

import os, sys
os.environ['TF_CUDNN_USE_AUTOTUNE']='0'  # autotune adds random memory spikes
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # suppress init messages

# folder with memory_saving_gradients
module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
#os.sys.path.append(os.path.dirname(sys.argv[0])+'/..')

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

# size, old_mbs->new_mbs
# RESNET_SIZE=34 # 1024->662
# RESNET_SIZE=50 # 
# RESNET_SIZE=101 # 3961->1352
RESNET_SIZE=200   # OOM -> 1791.44 MB (1337.58 ms)
RESNET_SIZE=152   # 5605.75 -> 1405.99, 18% increase in compute (895.86 to 1059.85)

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

def create_train_op_and_loss():
  """Creates loss tensor for resnet model."""
  images = tf.random_uniform((BATCH_SIZE, HEIGHT, WIDTH, DEPTH))
  labels = tf.random_uniform((BATCH_SIZE, NUM_CLASSES))
  if USE_TINY:
    network = resnet_model.tiny_resnet_v2(resnet_size=RESNET_SIZE, num_classes=NUM_CLASSES)
  else:
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


def gradient_memory_mbs():
  """Evaluates gradient, prints peak memory."""
  start_time0 = time.perf_counter()
  start_time = start_time0
  tf.reset_default_graph()
  tf.set_random_seed(1)
  
  train_op, loss = create_train_op_and_loss()
  print("Graph construction: %.2f ms" %(1000*(time.perf_counter()-start_time)))

  g = tf.get_default_graph()
  ops = g.get_operations()
  
  for op in ge.filter_ops_from_regex(ops, "block_layer"):
    tf.add_to_collection("checkpoints", op.outputs[0])

  sess = create_session()
  sessrun(tf.global_variables_initializer())
  start_time = time.perf_counter()
  sessrun(train_op)
  start_time = time.perf_counter()
  print("loss %f"%(sess.run(loss),))
  
  print("Compute time: %.2f ms" %(1000*(time.perf_counter()-start_time)))

  mem_use = mem_util.peak_memory(run_metadata)['/gpu:0']/1e6
  print("Memory used: %.2f MB "%(mem_use))
  total_time = time.perf_counter()-start_time0
  assert total_time < 100
  return mem_use

def test_memory_automatic():
#  assert tf.test.is_gpu_available(), "Memory tracking only works on GPU"
  old_gradients = tf.gradients

  if len(sys.argv)>1:
    mode = sys.argv[1]

  # automatic checkpoint selection TODO: find why it doesn't work with 0
  memory_saving_gradients.MIN_CHECKPOINT_NODE_SIZE = 1024
  def gradients_auto(ys, xs, grad_ys=None, **kwargs):
    return memory_saving_gradients.gradients(ys, xs, grad_ys,
                                             checkpoints='memory', **kwargs)

  tf.__dict__["gradients"] = gradients_auto
  print("Running with automatically selected checkpoints")
  peak_memory = gradient_memory_mbs()
  print(peak_memory) # < 700 # 662 on Nov 27
#  assert peak_memory < 2000   # 1405.99 on Nov 29
  # restore old gradients
  tf.__dict__["gradients"] = old_gradients
  print("Running without checkpoints")
  peak_memory = gradient_memory_mbs()
#  assert peak_memory > 5000 # 5605.75 on Nov 29

  print("Test passed")

if __name__=='__main__':
  test_memory_automatic()
