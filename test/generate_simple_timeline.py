import tensorflow as tf

import os, sys, time
import inspect
import numpy as np
import tensorflow as tf
import pdb


def make_chain_tanh_new(length=100, name_prefix="a", node_mbs=1):
  """Creates chain of length length. First node is Variable, rest are tanh.
  Returns nodes. Note, if length is 1, there are no non-linearities in the
  graph, hence gradients do not need to store any activations."""

  dtype = np.float32
  n = node_mbs * 250000
  val = tf.constant(1, dtype=dtype)
  a0_ = tf.fill((n,), val)
  #  a0_ = tf.ones((n,), dtype=dtype)
  a0 = tf.Variable(a0_, name=name_prefix+"00")
  a = a0
  nodes = [a]
  for i in range(1, length):
    name = "%s%02d"%(name_prefix, i)
    a = tf.tanh(a, name=name)
    nodes.append(a)
    
  return nodes

def main():
  sess = tf.Session()
  run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()
  from tensorflow.python.client import timeline
  
  nodes = make_chain_tanh_new(10)

  a0 = nodes[0]
  a = nodes[-1]

  sess.run(tf.global_variables_initializer())
  grad = tf.gradients([a], [a0])[0]

  results = sess.run(grad,
                     options=run_options,
                     run_metadata=run_metadata)

  tl = timeline.Timeline(step_stats=run_metadata.step_stats)
  ctf = tl.generate_chrome_trace_format(show_memory=True,
                                          show_dataflow=False)
  open("simple_stepstats.pbtxt", "w").write(str(run_metadata.step_stats))
  open("simple_timeline.json", "w").write(ctf)

if __name__=='__main__':
  main()
