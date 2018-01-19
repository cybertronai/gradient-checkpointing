"""Benchmark for memory reduction in deep resnet."""

import argparse
import os
import sys
import time

parser = argparse.ArgumentParser(description='deep resnet benchmark')
parser.add_argument('--name', type=str, default='deep',
                     help="name of benchmark run")
parser.add_argument('--max_blocks', type=int, default=10,
                     help="maximum number of blocks to add to resnet")
parser.add_argument('--outdir', type=str, default='.',
                     help="where to save results")
parser.add_argument('--disable_batch_norm', type=int, default=0,
                     help="where to save results")
args = parser.parse_args()

module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')

os.environ['TF_CUDNN_USE_AUTOTUNE']='0'  # autotune adds random memory spikes
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # silence tf init messages

import math
import numpy as np
import os
import pytest
import sys
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
import time

import memory_saving_gradients
import resnet_model   
import mem_util

pytestmark = pytest.mark.skipif(not tf.test.is_gpu_available(),
                                reason="needs gpu")
resnet_model._DISABLE_BATCH_NORM=bool(args.disable_batch_norm)

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

# add_2:0, add_7:0, add_12:0, add_17:0, add_22:0, add_27:0, add_32:0, add_37:0, add_42:0, add_47:0, add_52:0, add_57:0, 
USE_TINY = False

BATCH_SIZE=32
RESNET_SIZE=18

_WEIGHT_DECAY = 2e-4
_INITIAL_LEARNING_RATE = 0.1 * BATCH_SIZE / 128
_MOMENTUM = 0.9


# debug parameters
DUMP_GRAPHDEF = False

def create_session():
  optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
  config = tf.ConfigProto(operation_timeout_in_ms=150000, graph_options=tf.GraphOptions(optimizer_options=optimizer_options))
  #  config.graph_options.rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
  config.graph_options.place_pruned_graph = True
  return tf.Session(config=config)

def create_loss():
  """Creates loss tensor for resnet model."""
  images = tf.random_uniform((BATCH_SIZE, HEIGHT, WIDTH, DEPTH))
  labels = tf.random_uniform((BATCH_SIZE, NUM_CLASSES))
  network = resnet_model.resnet_v2(resnet_size=RESNET_SIZE,
                                   num_classes=NUM_CLASSES)

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

RESNET_SIZE=-1
def memory_test(size):
  """Evaluates gradient, returns memory in MB's and gradient eval time in
  seconds."""
  global sess, RESNET_SIZE

  RESNET_SIZE = size
  
  start_time0 = time.perf_counter()
  tf.reset_default_graph()
  
  loss = create_loss()

  start_time = time.perf_counter()
  grads = tf.group(tf.gradients(loss, tf.trainable_variables()))

  sess = create_session()
  sessrun(tf.global_variables_initializer())
  times = []
  memories = []
  for i in range(3):
    start_time = time.perf_counter()
    sessrun(grads)
    elapsed_time = time.perf_counter() - start_time
    times.append(elapsed_time)
    mem_use = mem_util.peak_memory(run_metadata)['/gpu:0']/1e6
    memories.append(mem_use)

  return np.min(memories), np.min(times)


def main():
  old_gradients = tf.gradients

  # automatic checkpoint selection
  def gradients_memory(ys, xs, grad_ys=None, **kwargs):
    return memory_saving_gradients.gradients(ys, xs, grad_ys,
                                             checkpoints='memory', **kwargs)
  print("Running with checkpoints")
  tf.__dict__["gradients"] = gradients_memory
  memories, times = [],[]

  outf = open(args.outdir+'/'+args.name+'.csv', 'w')

  # with checkpoints
  # 18                505
  # 34                661
  # 50                984
  # 101              1408
  # 152              1334
  # 200              1411
  #
  # without checkpoints
  # 18                818
  # 34               1166
  # 50               2694
  # 101              4112
  # 152              5826
  # 200              8320
  
  valid_sizes = [18,34,50,101,152,200]
  
  print("size      memory_cost")
  for size in valid_sizes:
    memory_cost, time_cost = memory_test(size=size)
    print("%-10d %10d"%(size, memory_cost))
    
    memories.append(memory_cost)
    times.append(time_cost)

  def tostr(l): return [str(e) for e in l]
  outf.write(','.join(str(i) for i in range(1, args.max_blocks))+'\n')
  outf.write(','.join(tostr(memories))+'\n')
  outf.write(','.join(tostr(times))+'\n')
  

  # 34               1167
  # 50               2685
  # 101              4111
  # 152              5826

  # restore old gradients
  print("Running without checkpoints")
  tf.__dict__["gradients"] = old_gradients
  memories, times = [],[]
  for size in valid_sizes:
    memory_cost, time_cost = memory_test(size=size)
    print("%-10d %10d"%(size, memory_cost))
    memories.append(memory_cost)
    times.append(time_cost)
    
  print(memories)
  print(times)

  outf.write(','.join(tostr(memories))+'\n')
  outf.write(','.join(tostr(times))+'\n')
  outf.close()

if __name__=='__main__':
  main()
