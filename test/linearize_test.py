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


def test_golden_order():
  n = 5
  tf.reset_default_graph()
  nodes = util.make_chain_tanh(n)
  a0 = nodes[0]
  a = nodes[-1]
  grad = tf.gradients([a], [a0])[0]
  print(tf.get_default_graph().as_graph_def())
  
  order = linearize_lib.linearize(modify_graph=False)
  golden_order = ['a00', 'a00/read', 'a01', 'a02', 'a03', 'gradients/Shape', 'gradients/grad_ys_0', 'gradients/Fill', 'a04', 'gradients/a04_grad/TanhGrad', 'gradients/a03_grad/TanhGrad', 'gradients/a02_grad/TanhGrad', 'ones', 'a00/Assign', 'gradients/a01_grad/TanhGrad']
  observed_order = [n.name for n in order]
  assert observed_order == golden_order


def test_chain_linearize():
  n = 5
  tf.reset_default_graph()
  nodes = util.make_chain_tanh_constant(n)
  a0 = nodes[0]
  a = nodes[-1]
  order1 = linearize_lib.obtain_linear_order()
  observed_order1 = [n.name for n in order1]
  
  num_new_deps = linearize_lib.linearize()
  assert num_new_deps == 0


def test_caterpillar_linearize():
  n = 5
  tf.reset_default_graph()
  nodes = util.make_caterpillar_graph(n)
  a0 = nodes[0]
  a = nodes[-1]
  grad = tf.gradients([a], [a0])[0]
  print(tf.get_default_graph().as_graph_def())
  
  order1 = linearize_lib.obtain_linear_order()
  observed_order1 = [n.name for n in order1]
  
  g = tf.get_default_graph()
  # g.version should track if graph was modified, but it doesn't
  # https://github.com/tensorflow/tensorflow/issues/14233
  num_new_deps = linearize_lib.linearize()
  assert num_new_deps > 0

  order2 = linearize_lib.obtain_linear_order()
  observed_order2 = [n.name for n in order2]
  assert observed_order1 == observed_order2
  
  num_new_deps = linearize_lib.linearize()
  assert num_new_deps == 0
  

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


def run_all_tests(module):
  all_functions = inspect.getmembers(module, inspect.isfunction)
  for name,func in all_functions:
    if name.endswith("_test"):
      print("Testing "+name)
      with timeit():
        func()
  print(module.__name__+" tests passed.")


if __name__=='__main__':
  #run_all_tests(sys.modules[__name__])
  #  test_double_linearize()
  print("run pytest instead")
