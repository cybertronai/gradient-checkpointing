# Collection of tests that use memory_util to measure amount of memory
# used by TensorFlow and validate it against prediction on simple graphs
# with and without memory-saving rewriting

# TODO: make tests deterministic with linearize
REMOVE_ASSERTS = False


import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES']='' # disable GPU


import inspect
import numpy as np
import tensorflow as tf
import pdb

from tensorflow.python.ops import gen_random_ops

import memory_saving_gradients

from util import make_chain_tanh
from util import make_chain_tanh_constant
from util import make_resnet
from util import debug_print
import util

import linearize as linearize_lib
import memory_util


run_metadata = None
DO_TRACING = True
def sessrun(*args, **kwargs):
  global sess, run_metadata
  
  if not DO_TRACING:
    return sess.run(*args, **kwargs)
  
  run_metadata = tf.RunMetadata()
  kwargs['options'] = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  kwargs['run_metadata'] = run_metadata
  result = sess.run(*args, **kwargs)
  return result

sess = None
def create_session():
  global sess
  config = tf.ConfigProto(log_device_placement=False, graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
  sess = tf.InteractiveSession(config=config) # todo: replace with regular sess
  return sess

# def setup_env():
#   """Sets up test enviornment."""

#   if not os.path.exists('../memory_util.py'):
#     assert False, "no memory_util"
#   return

#   # # download memory_util if needed
#   # memory_util_url = "https://raw.githubusercontent.com/yaroslavvb/memory_util/master/memory_util.py"
#   # if os.path.exists('memory_util.py'):
#   #   size = len(open('memory_util.py').read())
#   # else:
#   #   size = 0
    
#   # if size != 13636:
#   #   print("Size changed or 0, redownloading memory_util.py")
#   #   import urllib.request
#   #   response = urllib.request.urlopen(memory_util_url)
#   #   open("memory_util.py", "wb").write(response.read())

    
def make_chain_minmax(length, node_mbs=1):
  """Creates chain of nodes alternating minimum/maximum."""
    
  tf.reset_default_graph()
  n = node_mbs * 250000
  dtype = tf.float32
  upper = gen_random_ops._random_uniform((n,), dtype, name="u")
  y = gen_random_ops._random_uniform((n,), dtype, name="x")
  min_nodes = []
  max_nodes = []
  nodes = [y]
  for i in range(length):
    y = tf.maximum(upper, y, name="cl")
    max_nodes.append(y)
    nodes.append(y)
    y = tf.minimum(upper, y, name="cu")
    min_nodes.append(y)
    nodes.append(y)
  return nodes
    

# use class for now in case want to transition to tf.test later
class ChainTest:
  def test_chain(self):
    """Runs regular chain gradient, makes sure memory usage makes sense."""

    tf.reset_default_graph()
    
    n = 5
    nodes = make_chain_tanh(n)
    a0 = nodes[0]
    a = nodes[-1]
    with tf.control_dependencies([a]):
        grad = tf.gradients([a], [a0])[0]
        
    #linearize_lib.linearize()
    
    sess = create_session()
    sessrun(tf.global_variables_initializer())
    
    with memory_util.capture_stderr() as stderr:
      sessrun(grad.op)

    peak_memory = memory_util.peak_memory2(stderr.getvalue(), run_metadata)
    expected_peak = (n)*10**6
    
    # "loss" tensor
    util.report_memory(peak_memory, expected_peak)
    if not REMOVE_ASSERTS:
      assert abs(peak_memory - expected_peak) < 10000, "Difference too large."
    
  def test_chain_rewrite(self, linearize=False):
    """Take chain of length 5, save 2 nodes, make sure 2 units of RAM is
    saved."""
    
    tf.reset_default_graph()
    n = 5
    
    a0, a1, a2, a3, a4 = make_chain_tanh(n)
    grad = memory_saving_gradients.gradients([a4], [a0], remember=[a1,a3])[0]
    expected_peak = (n+1-2)*10**6  # subtract 2 since we recompute 2

    sess = create_session()
    sessrun(tf.global_variables_initializer())
    
    with memory_util.capture_stderr() as stderr:
      sessrun(grad.op)
    if linearize:
      linearize_lib.linearize()

    peak_memory = memory_util.peak_memory2(stderr.getvalue(), run_metadata)
    util.report_memory(peak_memory, expected_peak)

    if not REMOVE_ASSERTS:
      assert abs(peak_memory - expected_peak) < 10000, "Difference too large."


  def test_chain_rewrite_save_last(self):
    """Take chain of length 5, save last node. This saved no memory, and is 
    and edge case that should raise exception by rewriter."""
    
    tf.reset_default_graph()
    n = 5
    
    a0, a1, a2, a3, a4 = make_chain_tanh(n)
    try:
        grad = memory_saving_gradients.gradients([a4], [a0], remember=[a4])[0]
    except Exception:
        return
    else:
      if not REMOVE_ASSERTS:
        assert "Should've been 'no remember nodes found' exception"
        
  def test_chain_rewrite_save_one_before_last(self):
    """Take chain of length 5, save first node."""
    
    tf.reset_default_graph()
    n = 5
    
    a0, a1, a2, a3, a4 = make_chain_tanh_constant(n)
    grad = memory_saving_gradients.gradients([a4], [a0], remember=[a2])[0]
    expected_peak = (n+1-2)*10**6 

    sess = create_session()
    sessrun(tf.global_variables_initializer())
    
    with memory_util.capture_stderr() as stderr:
      sessrun(grad.op)

    peak_memory = memory_util.peak_memory2(stderr.getvalue(), run_metadata)
    util.report_memory(peak_memory, expected_peak)

    if not REMOVE_ASSERTS:
      assert abs(peak_memory - expected_peak) < 1.1e6, "Difference too large."

  def test_chain_rewrite_save_first(self):
    """Take chain of length 5, save first node."""
    
    tf.reset_default_graph()
    n = 5
    
    a0, a1, a2, a3, a4 = make_chain_tanh_constant(n)
    grad = memory_saving_gradients.gradients([a4], [a0], remember=[a0, a2])[0]
    expected_peak = (n+1-2)*10**6 

    sess = create_session()
    sessrun(tf.global_variables_initializer())
    
    with memory_util.capture_stderr() as stderr:
      sessrun(grad.op)

    peak_memory = memory_util.peak_memory2(stderr.getvalue(), run_metadata)
    util.report_memory(peak_memory, expected_peak)

    if not REMOVE_ASSERTS:
      assert abs(peak_memory - expected_peak) < 1.1e6, "Difference too large."

  def test_minmax(self):
    """Make a chain of min/max nodes, only save memory for mins."""

    tf.reset_default_graph()

    nodes = make_chain_minmax(length=11, node_mbs=10)
    x = nodes[0]
    loss = tf.reduce_sum(nodes[-1], name="loss")
    grad = tf.gradients(loss, [x])[0]

    with memory_util.capture_stderr() as suppress:
        sess = create_session()
        sessrun(tf.global_variables_initializer())
    
    with memory_util.capture_stderr() as stderr:
      sessrun(grad.op)
    debug_print(memory_util.peak_memory2(stderr), run_metadata)

    # 265,000,308
    
  def test_minmax_rewrite(self):
    """Make a chain of min/max nodes, only save memory for mins."""

    tf.reset_default_graph()
    nodes = make_chain_minmax(length=11, node_mbs=10)
    x = nodes[0]
    loss = tf.reduce_sum(nodes[-1], name="loss")
    grad = memory_saving_gradients.gradients([loss], [x],
                                             remember=nodes[::2])[0]

    sess = create_session()
    sessrun(tf.global_variables_initializer())
    
    with memory_util.capture_stderr() as stderr:
      sessrun(grad.op)

  def test_minmax_memory(self):
    """Make a chain of min/max nodes, only save memory for mins."""

    tf.reset_default_graph()
  
    nodes = make_chain_minmax(length=11, node_mbs=10)
    x = nodes[0]
    loss = tf.reduce_sum(nodes[-1], name="loss")
#    grad = tf.gradients(loss, [x])[0]
    grad = memory_saving_gradients.gradients([loss], [x],
                                             remember="memory")[0]

    sess = create_session()
    sessrun(tf.global_variables_initializer())
    
    with memory_util.capture_stderr() as stderr:
      sessrun(grad.op)


  def test_dual_chain(self):
    """Runs regular chain gradient, makes sure memory usage makes sense."""
    

    tf.reset_default_graph()
    n = 5
    nodes1 = make_chain_tanh_constant(n, "a")
    nodes2 = make_chain_tanh_constant(n, "b")
    
    a0,b0 = nodes1[0], nodes2[0]
    a, b = nodes1[-1], nodes2[-1]
    grad = tf.gradients([a+b], [a0, b0])
    
    sess = create_session()
    sessrun(tf.global_variables_initializer())
    
    with memory_util.capture_stderr() as stderr:
      sessrun([grad[0].op, grad[1].op])

    peak_memory = memory_util.peak_memory2(stderr.getvalue(), run_metadata)
    expected_peak = (2*n+1)*10**6
    util.report_memory(peak_memory, expected_peak)

    # 1 unit of memory slack since parallel computation chains adds
    # scheduling variablity
    if not REMOVE_ASSERTS:
      assert abs(peak_memory - expected_peak) < 1.1*10**9, "Difference too large."

  def test_dual_chain_rewrite(self):
    """Runs regular chain gradient, makes sure memory usage makes sense."""
    

    tf.reset_default_graph()
    n = 5
    nodes1 = make_chain_tanh_constant(n, "a")
    nodes2 = make_chain_tanh_constant(n, "b")
    
    a0,b0 = nodes1[0], nodes2[0]
    a, b = nodes1[-1], nodes2[-1]

    grad = memory_saving_gradients.gradients([a+b], [a0, b0],
                                             remember=[nodes1[2], nodes2[2]])
    
    sess = create_session()
    sessrun(tf.global_variables_initializer())
    
    with memory_util.capture_stderr() as stderr:
      sessrun([grad[0].op, grad[1].op])

    peak_memory = memory_util.peak_memory2(stderr.getvalue(), run_metadata)
    # normal usage comes from 2*n nodes + default ygrad node + 2 gradient nodes
    # here we save two 2 units of memory by dropping 2 activations (a1/b1) temporarily
    # also, this moves "peak memory" scenario lower down the chain
    # where the final addition node activations are no longer needed (another -1)
    expected_peak = (2*(n-1)+1)*10**6 
    util.report_memory(peak_memory, expected_peak)

    # since two independent chains, some variability in node scheduling
    # allow 1MB slack
    if not REMOVE_ASSERTS:
      assert abs(peak_memory - expected_peak) < 4.1e6, "Difference too large."

  def test_chain_memory(self, linearize=False):
    """Like test_chain, but use automatic rewriting with remember="memory" strat."""

    tf.reset_default_graph()
    n = 6  # for n=5, only choice of a2 saves memory, and alg picks a3
           # hence use n>5 to avoid this edge condition

    nodes = make_chain_tanh_constant(n)
    a0 = nodes[0]
    a = nodes[-1]
    grad = memory_saving_gradients.gradients_memory([a], [a0])[0]
    
    sess = create_session()
    sessrun(tf.global_variables_initializer())
    
    with memory_util.capture_stderr() as stderr:
      sessrun(grad.op)
      
    if linearize:
      linearize_lib.linearize()

    peak_memory = memory_util.peak_memory2(stderr.getvalue(), run_metadata)
    expected_peak = (n+1-1)*10**6  # 1 for each node + 1 for generated - 1 saved
                                   # "loss" tensor
    util.report_memory(peak_memory, expected_peak)
    if not REMOVE_ASSERTS:
      assert abs(peak_memory - expected_peak) < 10000, "Difference too large."
      
  def test_chain_tarjan(self, linearize=False):
    """Like test_chain, but use automatic rewriting with remember="tarjan"
    strategy."""

    print("Tarjan test disabled")
    return

    tf.reset_default_graph()
    n = 6  # for n=5, only choice of a2 saves memory, and alg picks a3
           # hence use n>5 to avoid this edge condition

    nodes = make_chain_tanh_constant(n)
    a0 = nodes[0]
    a = nodes[-1]
    grad = memory_saving_gradients.gradients_tarjan([a], [a0])[0]
    
    sess = create_session()
    sessrun(tf.global_variables_initializer())
    
    with memory_util.capture_stderr() as stderr:
      sessrun(grad.op)
      
    if linearize:
      linearize_lib.linearize()

    peak_memory = memory_util.peak_memory2(stderr.getvalue(), run_metadata)
    expected_peak = (n+1-1)*10**6  # 1 for each node + 1 for generated - 1 saved
                                   # "loss" tensor
    util.report_memory(peak_memory, expected_peak)
    if not REMOVE_ASSERTS:
      assert abs(peak_memory - expected_peak) < 1.1*10**6, "Difference too large."
      
  def test_long_chain_memory(self, linearize=False):
    """Like test_chain, but use automatic rewriting with remember="memory" 
    strategy."""

    tf.reset_default_graph()
    n = 100

    nodes = make_chain_tanh_constant(n)
    a0 = nodes[0]
    a = nodes[-1]
    tf.add_to_collection("remember", nodes[10])
    tf.add_to_collection("remember", nodes[20])
    #grad = memory_saving_gradients.gradients_collection([a], [a0])[0]
    grad = memory_saving_gradients.gradients_memory([a], [a0])[0]
    
    sess = create_session()
    sessrun(tf.global_variables_initializer())
    
    with memory_util.capture_stderr() as stderr:
      sessrun(grad.op)

    if linearize:
      added = linearize_lib.linearize()
      print("Added deps: ", added)

    peak_memory = memory_util.peak_memory2(stderr.getvalue(), run_metadata)
    # 20 mem used with following tensors picked automatically as bottlenecks
    # ['a10:0', 'a19:0', 'a28:0', 'a37:0', 'a46:0', 'a55:0', 'a64:0', 'a73:0',
    # 'a82:0', 'a91:0']
    expected_peak = 20 * 10**6 
    util.report_memory(peak_memory, expected_peak)

    if not REMOVE_ASSERTS:
      assert abs(peak_memory - expected_peak) < 1.1e6, "Difference too large."
      

  def test_long_chain_tarjan(self, linearize=False):
    """Like test_chain, but use automatic rewriting with remember="tarjan" 
    strategy."""

    print("Tarjan test disabled")
    return
  
    tf.reset_default_graph()
    n = 100

    nodes = make_chain_tanh_constant(n)
    a0 = nodes[0]
    a = nodes[-1]
    grad = memory_saving_gradients.gradients_tarjan([a], [a0])[0]
    
    sess = create_session()
    sessrun(tf.global_variables_initializer())
    
    with memory_util.capture_stderr() as stderr:
      sessrun(grad.op)

    if linearize:
      added = linearize_lib.linearize()
      print("Added deps: ", added)

    peak_memory = memory_util.peak_memory2(stderr.getvalue(), run_metadata)
    # 20 mem used with following tensors picked automatically as bottlenecks
    # ['a10:0', 'a19:0', 'a28:0', 'a37:0', 'a46:0', 'a55:0', 'a64:0', 'a73:0',
    # 'a82:0', 'a91:0']
    expected_peak = 20 * 10**6 
    util.report_memory(peak_memory, expected_peak)

    if not REMOVE_ASSERTS:
      assert abs(peak_memory - expected_peak) < 1.1*10**6, "Difference too large."
      

  def test_minimal_resnet(self, linearize=False):
    tf.reset_default_graph()
    n = 3

    nodes = make_resnet(n)
    a0 = nodes[0]
    a = nodes[-1]

    with tf.control_dependencies([a]):
        grad = tf.gradients([a], [a0])[0]
    
    sess = create_session()
    sessrun(tf.global_variables_initializer())
    
    with memory_util.capture_stderr() as stderr:
      sessrun(grad.op)

    if linearize:
      added = linearize_lib.linearize()
      print("Added deps: ", added)

    peak_memory = memory_util.peak_memory2(stderr, run_metadata)
    # 1 for activation of each tanh node + 1 for initial backprop node
    # + 1 temporary memory for computing the adds
    expected_peak = (n+1)*10**6 
    
    util.report_memory(peak_memory, expected_peak)
    if not REMOVE_ASSERTS:
      assert abs(peak_memory - expected_peak) < 1.1e6, "Difference too large."

  def test_resnet(self):
    tf.reset_default_graph()
    n = 6

    nodes = make_resnet(n)
    a0 = nodes[0]
    a = nodes[-1]

    with tf.control_dependencies([a]):
        grad = tf.gradients([a], [a0])[0]
    
    sess = create_session()
    sessrun(tf.global_variables_initializer())
    
    with memory_util.capture_stderr() as stderr:
      sessrun(grad.op)

    peak_memory = memory_util.peak_memory2(stderr, run_metadata)
    # 1 for activation of each tanh node + 1 for initial backprop node
    # + 1 temporary memory for computing the adds
    expected_peak = (n+1)*10**6 
    util.report_memory(peak_memory, expected_peak)

    if not REMOVE_ASSERTS:
      assert abs(peak_memory - expected_peak) < 10000, "Difference too large."

  def test_resnet_rewrite(self, linearize=False):
    tf.reset_default_graph()
    n = 6

    nodes = make_resnet(n)
    a0 = nodes[0]
    a = nodes[-1]

    remember = [nodes[3], nodes[5]] # ['a03_add:0', 'a05_add:0']
    grad = memory_saving_gradients.gradients([a], [a0], remember=[nodes[2]])[0]
    if linearize:
      added = linearize_lib.linearize(grad.op)
      print("Added deps: ", added)


    sess = create_session()
    sessrun(tf.global_variables_initializer())
    
    with memory_util.capture_stderr() as stderr:
      sessrun(grad.op)


    peak_memory = memory_util.peak_memory2(stderr.getvalue(), run_metadata)
    # 1 for activation of each tanh node + 1 for initial backprop node
    # + 1 temporary memory for computing the adds,
    # -1 for discarding, then recomputing a1_tanh
    expected_peak = (n-1)*10**6 
    util.report_memory(peak_memory, expected_peak)

    if not REMOVE_ASSERTS:
      assert abs(peak_memory - expected_peak) < 1.1*10**6, "Difference too large."

  def test_long_resnet(self):
    tf.reset_default_graph()
    n = 100
    nodes = make_resnet(n)
    a0 = nodes[0]
    a = nodes[-1]

    with tf.control_dependencies([a]):
        grad = tf.gradients([a], [a0])[0]
    
    sess = create_session()
    sessrun(tf.global_variables_initializer())
    
    with memory_util.capture_stderr() as stderr:
      sessrun(grad.op)

    peak_memory = memory_util.peak_memory2(stderr, run_metadata)
    # 1 for activation of each tanh node + 1 for initial backprop node
    # + 1 temporary memory for computing the adds
    expected_peak = (n+1)*10**6 
    util.report_memory(peak_memory, expected_peak)

    if not REMOVE_ASSERTS:
      assert abs(peak_memory - expected_peak) < 1.1e6, "Difference too large."
    
  def test_long_resnet_rewrite_memory(self, linearize=False):
    tf.reset_default_graph()
    n = 100
    nodes = make_resnet(n)
    a0 = nodes[0]
    a = nodes[-1]

    start_time = time.time()
    with tf.control_dependencies([a]):
        grad = memory_saving_gradients.gradients_memory([a], [a0])[0]
    print("Elapsed time, %.1f ms" %( (time.time()-start_time)*1000))

    start_time = time.time()
    if linearize:
      added = linearize_lib.linearize(grad.op)
      print("Added deps: ", added)

    print("Elapsed time, %.1f ms" %( (time.time()-start_time)*1000))
    sess = create_session()
    sessrun(tf.global_variables_initializer())
    
    with memory_util.capture_stderr() as stderr:
      sessrun(grad.op)

    peak_memory = memory_util.peak_memory2(stderr, run_metadata)
    # 20 mem used with following tensors picked automatically
    # ['a10_add:0', 'a19_add:0', 'a28_add:0', 'a37_add:0', 'a46_add:0',
    # 'a55_add:0', 'a64_add:0', 'a73_add:0', 'a82_add:0', 'a91_add:0']
    
    expected_peak = 20 * 10**6 
    util.report_memory(peak_memory, expected_peak)

    if not REMOVE_ASSERTS:
      assert abs(peak_memory - expected_peak) < 10000, "Difference too large."
    
  def test_long_resnet_rewrite_tarjan(self, linearize=False):
    print("Tarjan test disabled")
    return
  
    tf.reset_default_graph()
    n = 100
    nodes = make_resnet(n)
    a0 = nodes[0]
    a = nodes[-1]

    start_time = time.time()
    with tf.control_dependencies([a]):
        grad = memory_saving_gradients.gradients_tarjan([a], [a0])[0]
    print("Elapsed time, %.1f ms" %( (time.time()-start_time)*1000))

    start_time = time.time()
    if linearize:
      added = linearize_lib.linearize(grad.op)
      print("Added deps: ", added)

    print("Elapsed time, %.1f ms" %( (time.time()-start_time)*1000))
    sess = create_session()
    sessrun(tf.global_variables_initializer())
    
    with memory_util.capture_stderr() as stderr:
      sessrun(grad.op)

    peak_memory = memory_util.peak_memory2(stderr, run_metadata)
    # 20 mem used with following tensors picked automatically
    # ['a10_add:0', 'a19_add:0', 'a28_add:0', 'a37_add:0', 'a46_add:0',
    # 'a55_add:0', 'a64_add:0', 'a73_add:0', 'a82_add:0', 'a91_add:0']
    
    expected_peak = 20 * 10**6 
    util.report_memory(peak_memory, expected_peak)

    if not REMOVE_ASSERTS:
      assert abs(peak_memory - expected_peak) < 10000, "Difference too large."
    

  def test_resnet_rewrite_memory(self, linearize=False):
    tf.reset_default_graph()
    n = 6   # use n>5 (see test_chain_memory)

    nodes = make_resnet(n)
    a0 = nodes[0]
    a = nodes[-1]


    remember = [nodes[3], nodes[5]] # ['a03_add:0', 'a05_add:0']
    grad = memory_saving_gradients.gradients_memory([a], [a0])[0]
    if linearize:
      added = linearize_lib.linearize(grad.op)
      print("Added deps: ", added)

    sess = create_session()
    sessrun(tf.global_variables_initializer())
    
    with memory_util.capture_stderr() as stderr:
      sessrun(grad.op)

    peak_memory = memory_util.peak_memory2(stderr.getvalue(), run_metadata)
    # 1 for activation of each tanh node + 1 for initial backprop node
    # + 1 temporary memory for computing the adds,
    # -1 for discarding, then recomputing a1_tanh
    expected_peak = (n+1+1-1)*10**6 
    util.report_memory(peak_memory, expected_peak)

    if not REMOVE_ASSERTS:
      assert abs(peak_memory - expected_peak) < 1.1*10**6, "Difference too large."

  def test_resnet_rewrite_tarjan(self, linearize=False):
    print("Tarjan test disabled")
    return
  
    tf.reset_default_graph()
    n = 6   # use n>5 (see test_chain_memory)

    nodes = make_resnet(n)
    a0 = nodes[0]
    a = nodes[-1]


    remember = [nodes[3], nodes[5]] # ['a03_add:0', 'a05_add:0']
    grad = memory_saving_gradients.gradients_tarjan([a], [a0])[0]
    if linearize:
      added = linearize_lib.linearize(grad.op)
      print("Added deps: ", added)

    sess = create_session()
    sessrun(tf.global_variables_initializer())
    
    with memory_util.capture_stderr() as stderr:
      sessrun(grad.op)

    peak_memory = memory_util.peak_memory2(stderr.getvalue(), run_metadata)
    # 1 for activation of each tanh node + 1 for initial backprop node
    # + 1 temporary memory for computing the adds,
    # -1 for discarding, then recomputing a1_tanh
    expected_peak = (n+1+1-1)*10**6 
    util.report_memory(peak_memory, expected_peak)

    if not REMOVE_ASSERTS:
      assert abs(peak_memory - expected_peak) < 1.1*10**6, "Difference too large."

      
if __name__ == '__main__':
  test = ChainTest()
  #  setup_env()

  # disable GPUs for consistent results between gpu/non-gpu machines
#  os.environ['CUDA_VISIBLE_DEVICES']=''
  memory_util.vlog(1)   # vlog=2 on GPU machine will spam gpu "polling" msgs

  test.test_chain()
  test.test_chain_rewrite(linearize=True)
  test.test_chain_rewrite_save_first()
  test.test_chain_rewrite_save_last()
  test.test_chain_rewrite_save_one_before_last()
  test.test_dual_chain()
  test.test_dual_chain_rewrite()
  
  #  test.test_chain_memory()
  #  test.test_chain_memory(linearize=True)

  # TODO: fix, now it uses only ['a01:0'] as remember node
  #  test.test_long_chain_memory(linearize=False)
  # test.test_long_chain_memory(linearize=True)
  
  test.test_minimal_resnet()
  test.test_minimal_resnet(linearize=True)
  test.test_resnet_rewrite()
  test.test_resnet_rewrite(linearize=True)
  #  test.test_resnet_rewrite_memory()
  #  test.test_resnet_rewrite_memory(linearize=True)
  test.test_long_resnet()
  #  test.test_long_resnet_rewrite_memory()
  #  test.test_long_resnet_rewrite_memory(linearize=True)
  
  # test.test_chain_tarjan()
  # test.test_long_chain_tarjan()
  # test.test_long_chain_tarjan(linearize=True)

  # test.test_resnet_rewrite_tarjan()
  # test.test_long_resnet_rewrite_tarjan(linearize=True)

  print("%s tests succeeded"%(sys.argv[0],))
