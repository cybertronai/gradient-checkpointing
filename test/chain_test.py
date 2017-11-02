import os
os.environ['TF_CUDNN_USE_AUTOTUNE']='0'  # autotune adds random memory spikes

import math
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
import time
import memory_saving_gradients
import memory_util
memory_util.vlog(1)   # vlog=2 on GPU machine will spam gpu "polling" msgs
import util


def create_session():
  config = tf.ConfigProto(log_device_placement=False, graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
  return tf.InteractiveSession(config=config)


def make_chain_tanh(length=100, name_prefix="a", node_mbs=1):
  """Creates chain of length length. First node is Variable, rest are tanh.
  Returns nodes. Note, if length is 1, there are no non-linearities in the
  graph, hence gradients do not need to store any activations."""

  node_mbs = 1
  dtype = np.float32
  n = node_mbs * 250000
  a0_ = tf.ones((n,), dtype=dtype)
  a0 = tf.Variable(a0_, name=name_prefix+"00")
  a = a0
  nodes = [a]
  for i in range(1, length):
    name = "%s%02d"%(name_prefix, i)
    a = tf.tanh(a, name=name)
    nodes.append(a)
    
  return nodes


def main():
  tf.reset_default_graph()
  n = 100

  nodes = make_chain_tanh(n)
  a0 = nodes[0]
  a = nodes[-1]
  grad = memory_saving_gradients.gradients_memory([a], [a0])[0]

  sess = create_session()
  sess.run(tf.global_variables_initializer())

  with memory_util.capture_stderr() as stderr:
    sess.run(grad.op)

  peak_memory1 = memory_util.peak_memory(stderr.getvalue())
  # 20 mem used with following tensors picked automatically as bottlenecks
  # ['a10:0', 'a19:0', 'a28:0', 'a37:0', 'a46:0', 'a55:0', 'a64:0', 'a73:0',
  # 'a82:0', 'a91:0']

  # method 2
  mem_op = tf.contrib.memory_stats.MaxBytesInUse()
  peak_memory2 = sess.run(mem_op)

  # method 3
  run_metadata = tf.RunMetadata()
  run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

  sess.run(grad.op, run_metadata=run_metadata, options=run_options)
  peak_memory3 = memory_util.peak_from_metadata(run_metadata)['gpu']
  print(peak_memory1)
  print(peak_memory2)
  print(peak_memory3)
  
  expected_peak = 20 * 10**6 
  util.report_memory(peak_memory1, expected_peak)

  assert abs(peak_memory - expected_peak) < 1000, "Difference too large."

if __name__=='__main__':
  assert tf.test.is_gpu_available(), "Memory tracking only works on GPU"
  main()
