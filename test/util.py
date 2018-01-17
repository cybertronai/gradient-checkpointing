# utility functions for memory_saving_gradients

import contextlib
import inspect
import math
import networkx as nx
import numpy as np
import os
import tempfile
import time

import tensorflow as tf
from tensorflow.python.ops import gen_random_ops # for low overhead random
from tensorflow.contrib import graph_editor as ge

DEBUG_LOGGING = False

def report_memory(peak_memory, expected_peak):
  """Helper utility to print 2 memory stats side by side, used in memory
  tests."""
  
  parent_name = inspect.stack()[1][0].f_code.co_name
  print("%s: peak memory: %.3f MB, "
        "expected peak:  %.3f MB" % (parent_name,
                                     peak_memory/10**6,
                                     expected_peak/10**6))


def enable_debug():
  """Turn on debug logging."""
  
  global DEBUG_LOGGING
  DEBUG_LOGGING = True

def disable_debug():
  """Turn off debug logging."""
  global DEBUG_LOGGING
  DEBUG_LOGGING = False


def format_ops(ops, sort_outputs=False):
  """Helper method for printing ops. Converts Tensor/Operation op to op.name,
  rest to str(op)."""
    
  if hasattr(ops, '__iter__') and not isinstance(ops, str):
    l = [(op.name if hasattr(op, "name") else str(op)) for op in ops]
    if sort_outputs:
      return sorted(l)
    return l
  else:
    return ops.name if hasattr(ops, "name") else str(ops)
    

# TODO(y): add support for empty s
def debug_print(s, *args):
  """Like logger.log, but also replaces all TensorFlow ops/tensors with their
  names. Sensitive to value of DEBUG_LOGGING, see enable_debug/disable_debug

  Usage:
    debug_print("see tensors %s for %s", tensorlist, [1,2,3])
  """
  
  if DEBUG_LOGGING:
    formatted_args = [format_ops(arg) for arg in args]
    print("DEBUG "+s % tuple(formatted_args))


def debug_print2(s, *args):
  """Like logger.log, but also replaces all TensorFlow ops/tensors with their
  names. Not sensitive to value of DEBUG_LOGGING

  Usage:
    debug_print2("see tensors %s for %s", tensorlist, [1,2,3])
  """
  
  formatted_args = [format_ops(arg, sort_outputs=False) for arg in args]
  print("DEBUG2 "+s % tuple(formatted_args))



def print_tf_graph(graph):
  """Prints tensorflow graph in dictionary form."""
  for node in graph:
    for child in graph[node]:
      print("%s -> %s" % (node.name, child.name))
  

def save_tf_graph(graph, fn):
  """Prints tensorflow graph in dictionary form."""
  out = open(fn, 'w')
  for node in graph:
    for child in graph[node]:
      out.write("%s -> %s\n" % (node.name, child.name))
  out.close()


# TODO: turn lists into sets (required by toposort)
# TODO: ordered dict instead of dict
def tf_ops_to_graph(ops):
  """Creates op->children dictionary from list of TensorFlow ops."""

  def flatten(l): return [item for sublist in l for item in sublist]
  def children(op): return flatten(tensor.consumers() for tensor in op.outputs)
  return {op: children(op) for op in ops}


def tf_ops_to_nx_graph(ops):
  """Convert Tensorflow graph to NetworkX graph."""
  
  return nx.Graph(tf_ops_to_graph(ops))


def hash_graph(graph):
  """Convert graph nodes to hashes (networkx is_isomorphic needs integer
  nodes)."""
  
  return {hash(key): [hash(value) for value in graph[key]] for key in graph}


def graphs_isomorphic(graph1, graph2):
  """Check if two graphs are isomorphic."""
  
  return nx.is_isomorphic(nx.Graph(hash_graph(graph1)),
                          nx.Graph(hash_graph(graph2)))


def set_equal(one, two):
  """Converts inputs to sets, tests for equality."""
  
  return set(one) == set(two)


def make_caterpillar_graph(length=5, node_mbs=1):
  """Length is number of concats."""
  
  n = node_mbs * 250000
  n2 = int(math.sqrt(n))
  dtype = tf.float32
    
  def make_leaf(i):
    name = "leaf"+str(i)
    val = gen_random_ops._random_uniform((n2, n2), dtype, name=name)
    return val
   
  def make_merge(a, b, i):
    name = "merge"+str(i)
    merge_node = tf.matmul(a, b, name=name)
    #    nonlinear_node = tf.tanh(merge_node, name="tanh"+str(i))
    #nonlinear_node = tf.identity(merge_node, name="tanh"+str(i))
    return merge_node

  leaf0 = make_leaf(0)
  node0 = tf.identity(leaf0, name="merge0")
  node = node0
  nodes = [node]
  
  for i in range(1, length+1):
    leaf = make_leaf(i)
    node = make_merge(node, leaf, i)
    nodes.append(node)
  return nodes


def make_chain_tanh(length=100, name_prefix="a", node_mbs=1):
  """Creates chain of length length. First node is Variable, rest are tanh.
  Returns nodes. Note, if length is 1, there are no non-linearities in the
  graph, hence gradients do not need to store any activations."""

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


def make_chain_tanh_constant(length=100, name_prefix="a", node_mbs=1):
  """Creates chain of length length. First node is constant, rest are tanh.
  Returns list of nodes. Advantage over make_chain_tanh is that
  unlike Variable, memory for constant is allocated in the same run call, so
  easier to track."""

  dtype = np.float32
  n = node_mbs * 250000
  a0 = tf.ones((n,), dtype=dtype, name=name_prefix+"00")
  a = a0
  nodes = [a]
  for i in range(1, length):
    name = "%s%02d"%(name_prefix, i)

    a = tf.tanh(a, name=name)
    nodes.append(a)
    
  return nodes

def make_chain_tanh_fill(length=100, name_prefix="a", node_mbs=1):
  """Creates chain of length length. First node is Variable, rest are tanh.
  Returns nodes. Note, if length is 1, there are no non-linearities in the
  graph, hence gradients do not need to store any activations."""

  dtype = np.float32
  n = node_mbs * 250000
  val = tf.constant(1, dtype=dtype)
  a0 = tf.fill((n,), val)
  a = a0
  nodes = [a]
  for i in range(1, length):
    name = "%s%02d"%(name_prefix, i)
    a = tf.tanh(a, name=name)
    nodes.append(a)
  return nodes

def make_resnet(length=100, name_prefix="a", node_mbs=1):
  """Creates resnet-like chain of length length. First node is constant,
  rest are tanh.  Returns list of nodes. Has length - 2 articulation points (ie 
  for length=3 there is 1 articulation point."""

  dtype = np.float32
  n = node_mbs * 250000
  a0 = tf.ones((n,), dtype=dtype, name=name_prefix+"00")
  a = a0
  nodes = [a]
  for i in range(1, length):
    name = "%s%02d"%(name_prefix, i)
    a_nonlin = tf.tanh(a, name=name+"_tanh")
    a = tf.add(a, a_nonlin, name=name+"_add")
    nodes.append(a)
    
  return nodes

def make_resnet_custom(length=100, name_prefix="a", node_mbs=1):
  """Creates resnet-like chain of length length. First node is constant,
  rest are tanh.  Returns list of nodes. Has length - 2 articulation points (ie 
  for length=3 there is 1 articulation point."""

  dtype = np.float32
  n = node_mbs * 250000
  a0 = tf.ones((n,), dtype=dtype, name=name_prefix+"00")
  a = a0
  nodes = [a]
  for i in range(1, length):
    name = "%s%02d"%(name_prefix, i)
    a_nonlin = tf.tanh(a, name=name+"_tanh")
    a = tf.sigmoid(tf.add(a, a_nonlin, name=name+"_add"))
    nodes.append(a)
    
  return nodes


STDOUT=1
STDERR=2
class capture_stderr:
  """Utility to capture output, use as follows
     with util.capture_stderr() as stderr:
        sess = tf.Session()

    print("Captured:", stderr.getvalue()).
    """

  def __init__(self, fd=STDERR):
    self.fd = fd
    self.prevfd = None

  def __enter__(self):
    t = tempfile.NamedTemporaryFile()
    self.prevfd = os.dup(self.fd)
    os.dup2(t.fileno(), self.fd)
    return TemporaryFileHelper(t)

  def __exit__(self, exc_type, exc_value, traceback):
    os.dup2(self.prevfd, self.fd)

class capture_stdout:
  """Utility to capture output, use as follows
     with util.capture_stderr() as stderr:
        sess = tf.Session()

    print("Captured:", stderr.getvalue()).
    """

  def __init__(self, fd=STDOUT):
    self.fd = fd
    self.prevfd = None

  def __enter__(self):
    t = tempfile.NamedTemporaryFile()
    self.prevfd = os.dup(self.fd)
    os.dup2(t.fileno(), self.fd)
    return TemporaryFileHelper(t)

  def __exit__(self, exc_type, exc_value, traceback):
    os.dup2(self.prevfd, self.fd)


class TemporaryFileHelper:
  """Provides a way to fetch contents of temporary file.""" 
  def __init__(self, temporary_file):
    self.temporary_file = temporary_file
  def getvalue(self):
    return open(self.temporary_file.name).read() 
    

@contextlib.contextmanager
def capture_ops():
  """Decorator to capture ops created in the block.
  with capture_ops() as ops:
    # create some ops
  print(ops) # => prints ops created.
  """

  micros = int(time.time()*10**6)
  scope_name = str(micros)
  op_list = []
  with tf.name_scope(scope_name):
    yield op_list

  g = tf.get_default_graph()
  op_list.extend(ge.select_ops(scope_name+"/.*", graph=g))

        
def pick_every_k(l, k):
  """Picks out every k'th element from the sequence, using round()
  when k is not integer."""
  
  result = []
  x = k
  while round(x) < len(l):
    result.append(l[int(round(x))])
    print("picking ", int(round(x)))
    x += k
  return result


def pick_n_equispaced(l, n):
  """Picks out n points out of list, at roughly equal intervals."""

  assert len(l) >= n
  r = (len(l) - n)/float(n)
  result = []
  pos = r
  while pos < len(l):
    result.append(l[int(pos)])
    pos += 1+r
  return result


def sort(nodes, total_order, dedup=False):
  """Sorts nodes according to order provided.
  
  Args:
    nodes: nodes to sort
    total_order: list of nodes in correct order
    dedup: if True, also discards duplicates in nodes

  Returns:
    Iterable of nodes in sorted order.
  """

  total_order_idx = {}
  for i, node in enumerate(total_order):
    total_order_idx[node] = i
  if dedup:
    nodes = set(nodes)
  return sorted(nodes, key=lambda n: total_order_idx[n])

def to_ops(iterable):
  if not is_iterable(iterable):
    return iterable
  return [to_op(i) for i in iterable]


def intercept_op_creation(op_type_name_to_intercept):
  """Drops into PDB when particular op type is added to graph."""
  from tensorflow.python.framework import op_def_library
  old_apply_op = op_def_library.OpDefLibrary.apply_op
  def my_apply_op(obj, op_type_name, name=None, **keywords):
    import pdb; pdb.set_trace()
    print(op_type_name+"-"+str(name))
    if op_type_name == op_type_name_to_intercept:
      import pdb; pdb.set_trace()
    return(old_apply_op(obj, op_type_name, name=name, **keywords))
  op_def_library.OpDefLibrary.apply_op=my_apply_op
