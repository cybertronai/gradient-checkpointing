"""Resnet test that uses new API.

Expected result

Calling memsaving gradients with  collection
Memory used: 700.98 MB
Running without checkpoints
Memory used: 1236.68 MB
"""

import os
os.environ['TF_CUDNN_USE_AUTOTUNE']='0'  # autotune adds random memory spikes
os.environ['CUDA_VISIBLE_DEVICES']=''

import math
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
import time
from tensorflow.core.protobuf import rewriter_config_pb2

assert os.getcwd().endswith("/test"), "must run from 'test' directory"
sys.path.extend([".."])   # folder with memory_saving_gradients
import memory_saving_gradients
import memory_util

import resnet_model   


# add_2:0, add_7:0, add_12:0, add_17:0, add_22:0, add_27:0, add_32:0, add_37:0, add_42:0, add_47:0, add_52:0, add_57:0, 
USE_TINY = False

# resnet parameters
HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10
if USE_TINY:
  BATCH_SIZE=10
else:
  BATCH_SIZE=128
_WEIGHT_DECAY = 2e-4
_INITIAL_LEARNING_RATE = 0.1 * BATCH_SIZE / 128
_MOMENTUM = 0.9
RESNET_SIZE=122

# debug parameters
DUMP_GRAPHDEF = False

def create_session():
  optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
  config = tf.ConfigProto(operation_timeout_in_ms=150000, graph_options=tf.GraphOptions(optimizer_options=optimizer_options))
  config.graph_options.rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
  return tf.Session(config=config)

def create_loss():
  """Creates loss tensor for resnet model."""
  images = tf.random_uniform((BATCH_SIZE, HEIGHT, WIDTH, DEPTH))
  labels = tf.random_uniform((BATCH_SIZE, NUM_CLASSES))
  # channels_last for CPU
  if USE_TINY:
    network = resnet_model.tiny_cifar10_resnet_v2_generator(RESNET_SIZE, NUM_CLASSES, data_format='channels_last')
  else:
    network = resnet_model.cifar10_resnet_v2_generator(RESNET_SIZE, NUM_CLASSES, data_format='channels_last')
  inputs = tf.reshape(images, [BATCH_SIZE, HEIGHT, WIDTH, DEPTH])
  logits = network(inputs,True)
  cross_entropy = tf.losses.softmax_cross_entropy(logits=logits,
                                                  onehot_labels=labels)
  l2_penalty = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
  loss = cross_entropy + _WEIGHT_DECAY * l2_penalty
  return loss

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

def gradient_memory_test():
  """Evaluates gradient, prints peak memory."""
  global sess
  
  start_time0 = time.perf_counter()
  loss = create_loss()

  if DUMP_GRAPHDEF:
    open('graphdef.txt', 'w').write(str(tf.get_default_graph().as_graph_def()))

  # use block_layer1, block_layer2, block_layer3 as remember nodes
  g = tf.get_default_graph()
  ops = g.get_operations()
  for op in ge.filter_ops_from_regex(ops, "block_layer"):
    tf.add_to_collection("remember", op.outputs[0])

  start_time = time.perf_counter()
  grads = tf.gradients(loss, tf.trainable_variables())
  print("Graph construction time: %.2f sec" %(time.perf_counter()-start_time))

  
  start_time = time.perf_counter()
  sess = create_session()
  print("Sess create time: %.2f sec" %(time.perf_counter()-start_time))
  start_time = time.perf_counter()
  sessrun(tf.global_variables_initializer())
  print("Init time: %.2f sec" %(time.perf_counter()-start_time))
  start_time = time.perf_counter()
  sessrun(grads)
  print("Compute time0: %.2f sec" %(time.perf_counter()-start_time))
  start_time = time.perf_counter()
  sessrun(grads)
  print("Compute time: %.2f sec" %(time.perf_counter()-start_time))

  #  mem_op = tf.contrib.memory_stats.MaxBytesInUse()
  #  mem_use = sessrun(mem_op)/1e6
  mem_use = memory_util.peak_memory2(None, run_metadata)/1e6
  
  print("Memory used: %.2f MB "%(mem_use))
  total_time = time.perf_counter()-start_time0
  print("Total time: %.2f sec"%(total_time))
  assert total_time < 100
  return mem_use


def main():
  #  assert tf.test.is_gpu_available(), "Memory tracking only works on GPU"
  old_gradients = tf.gradients

  # TODO: find why this doesn't work with this set to 0
  memory_saving_gradients.MIN_CHECKPOINT_NODE_SIZE = 100
  
  # automatic checkpoint selection
  def gradients_auto(ys, xs, grad_ys=None, **kwargs):
    return memory_saving_gradients.gradients(ys, xs, grad_ys,
                                             remember='memory', **kwargs)
  tf.__dict__["gradients"] = gradients_auto
  print("Running with automatically selected checkpoints")
  assert(gradient_memory_test() < 720)
  
  # replace tf.gradients with custom version
  def gradients_collection(ys, xs, grad_ys=None, **kwargs):
    return memory_saving_gradients.gradients(ys, xs, grad_ys,
                                             remember='collection', **kwargs)
  tf.__dict__["gradients"] = gradients_collection
  print("Running with manual checkpoints")
  #  assert(gradient_memory_test() < 730)
  assert(gradient_memory_test() < 1000)


  # restore old gradients
  tf.__dict__["gradients"] = old_gradients
  
  print("Running without checkpoints")
  assert(gradient_memory_test() < 1350)
  print("Test passed")

if __name__=='__main__':
  main()
