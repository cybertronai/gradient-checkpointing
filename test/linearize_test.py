import linearize as linearize_lib
import util

import os, sys, time
import inspect
import numpy as np
import tensorflow as tf
import pdb
import math
import pprint
import toposort


def create_session():
  config = tf.ConfigProto(log_device_placement=False, graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
  return tf.InteractiveSession(config=config)


def setup_env():
  """Sets up test environment."""
  
  # # download memory_util if needed
  # memory_util_url = "https://raw.githubusercontent.com/yaroslavvb/memory_util/master/memory_util.py"
  # if os.path.exists('memory_util.py'):
  #   size = len(open('memory_util.py').read())
  # else:
  #   size = 0
    
  # if size != 13636:
  #   print("Size changed or 0, redownloading memory_util.py")
  #   import urllib.request
  #   response = urllib.request.urlopen(memory_util_url)
  #   open("memory_util.py", "wb").write(response.read())

    
def test_print():
  """Should print:
  leaf1 -> merge1
  leaf0 -> merge0
  merge1 -> merge2
  merge0 -> merge1
  leaf2 -> merge2
  leaf0/shape -> leaf0
  leaf1/shape -> leaf1
  leaf2/shape -> leaf2
  """
  
  nodes = util.make_caterpillar_graph(length=2)
  linearize_lib.print_tf_graph(linearize_lib.get_graph())
  

def test_toposort():
  nodes = util.make_caterpillar_graph(length=2)
  graph = linearize_lib.get_graph()
  print(list(toposort.toposort(graph)))


def test_caterpillar():
  tf.reset_default_graph()
  nodes = util.make_caterpillar_graph(10)
  linearize_lib.linearize()

  sess = create_session()

  import memory_util
  memory_util.vlog(1)
  with memory_util.capture_stderr() as stderr:
    sess.run(nodes[-1].op)
  #  memory_util.print_memory_timeline(stderr, ignore_less_than_bytes=1000)
  
  # optimal order needs 3MB to execute regardless of length
  peak_memory = memory_util.peak_memory(stderr)
  expected_peak = 3*10**6
  util.report_memory(peak_memory, expected_peak)
  assert abs(peak_memory - expected_peak) < 10000, "Difference too large."

def test_caterpillar_duplicate():
  """Make sure it works if we linearize twice."""
  tf.reset_default_graph()
  nodes = util.make_caterpillar_graph(20)
  linearize_lib.linearize()
  old_version = tf.get_default_graph()._version
  linearize_lib.linearize()
  assert old_version == tf.get_default_graph()._version

  sess = create_session()

  import memory_util
  memory_util.vlog(1)
  with memory_util.capture_stderr() as stderr:
    sess.run(nodes[-1].op)
  #  memory_util.print_memory_timeline(stderr, ignore_less_than_bytes=1000)
  
  # optimal order needs 3MB to execute regardless of length
  peak_memory = memory_util.peak_memory(stderr)
  expected_peak = 3*10**6
  util.report_memory(peak_memory, expected_peak)
  assert abs(peak_memory - expected_peak) < 10000, "Difference too large."


def test_chain_gradient_existing_dep():
  n = 2
  tf.reset_default_graph()
  nodes = util.make_chain_tanh(n)
  a0 = nodes[0]
  a = nodes[-1]
  with tf.control_dependencies([a]):
    grad = tf.gradients([a], [a0])[0]
  linearize_lib.linearize()
  #  linearize_lib.print_tf_graph(linearize_lib.get_graph())
  sess = create_session()
  sess.run(tf.global_variables_initializer())
  with memory_util.capture_stderr() as stderr:
    sess.run(grad.op)

  peak_memory = memory_util.peak_memory(stderr.getvalue())
  expected_peak = (n+1)*10**6 # 1 for each node + 1 for generated
  # "loss" tensor
  util.report_memory(peak_memory, expected_peak)
  assert abs(peak_memory - expected_peak) < 1e6, "Difference too large."


def test_chain_gradient():
  n = 5
  tf.reset_default_graph()
  nodes = util.make_chain_tanh(n)
  a0 = nodes[0]
  a = nodes[-1]
  grad = tf.gradients([a], [a0])[0]
        
  linearize_lib.linearize()
  sess = create_session()
  sess.run(tf.global_variables_initializer())
  with memory_util.capture_stderr() as stderr:
    sess.run(grad.op)

  peak_memory = memory_util.peak_memory(stderr.getvalue())
  expected_peak = (n+1)*10**6 # 1 for each node + 1 for generated
  # "loss" tensor
  util.report_memory(peak_memory, expected_peak)
  assert abs(peak_memory - expected_peak) < 2e6, "Difference too large."

def test_chain_gradient_nomodify():
  n = 5
  tf.reset_default_graph()
  nodes = util.make_chain_tanh(n)
  a0 = nodes[0]
  a = nodes[-1]
  grad = tf.gradients([a], [a0])[0]
        
  order = linearize_lib.linearize(modify_graph=False)
  golden_order = ['a00/read', 'a01', 'a02', 'a03', 'gradients/Const',
                  'gradients/Shape', 'gradients/Fill', 'a04',
                  'gradients/a04_grad/TanhGrad', 'gradients/a03_grad/TanhGrad',
                  'gradients/a02_grad/TanhGrad', 'gradients/a01_grad/TanhGrad']
  assert [n.name for n in order] == golden_order


# test linearizing for different targets, make sure correct number of
# dependencies is added, check that linearizing twice is a no-op
def test_targets():
  tf.reset_default_graph()
  n = 5
  g = tf.get_default_graph()
  nodes1 = util.make_chain_tanh_constant(n, "a")
  nodes2 = util.make_chain_tanh_constant(n, "b")
    
  a0,b0 = nodes1[0], nodes2[0]
  a, b = nodes1[-1], nodes2[-1]
  grad1 = tf.gradients([a], [a0, b0])
  grad2 = tf.gradients([b], [a0, b0])
  assert linearize_lib.linearize(grad1) == 3
  old_version = g._version
  assert linearize_lib.linearize(grad1) == 0
  assert g._version == old_version  
  
  assert linearize_lib.linearize(grad2) == 3
  assert linearize_lib.linearize(grad2) == 0

  
def create_session():
  config = tf.ConfigProto(log_device_placement=False, graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
  return tf.InteractiveSession(config=config)


if __name__=='__main__':
  setup_env()
  import memory_util

  memory_util.vlog(1)
  
  # test below got broken between September and 1.4rc0
  # test_chain_gradient_nomodify()
  test_caterpillar()
  test_caterpillar_duplicate()
  test_chain_gradient()
  test_chain_gradient_existing_dep()
  test_targets()
  print("%s tests succeeded"%(sys.argv[0],))
