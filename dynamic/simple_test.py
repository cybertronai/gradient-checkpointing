import tensorflow as tf
import toposort
import networkx as nx

from tensorflow.python.client import timeline
from tensorflow.python.ops import gen_math_ops
tanh_grad = gen_math_ops._tanh_grad

import os, sys, time
import inspect
import numpy as np
import tensorflow as tf
import pdb
import tensorflow.contrib.graph_editor as ge

sys.path.insert(0, os.environ["HOME"]+"/git0/gradient-checkpointing/test")
import memory_util
import linearize as linearize_lib
from linearize import OrderedSet

from collections import OrderedDict

size_mbs = 1   # size of nodes
size = size_mbs * 250000  

# workaround for https://github.com/tensorflow/tensorflow/issues/13754
setattr(tf.GraphKeys, "VARIABLES", "variables")

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


def print_ops(ops):
  """Print list of ops nicely."""
  print(format_ops(ops))
  
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


def create_session():
  """Create session with optimizations disabled."""
  from tensorflow.core.protobuf import rewriter_config_pb2
  optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
  config = tf.ConfigProto(operation_timeout_in_ms=150000,
                          graph_options=tf.GraphOptions(optimizer_options=optimizer_options))
  config.graph_options.rewrite_options.constant_folding=rewriter_config_pb2.RewriterConfig.OFF
  config.graph_options.place_pruned_graph = True
  return tf.Session(config=config)


################################################################################
# Simple tests for chain.
#
# Create chain of n tanh operations A0->A1->...->An
# and n tanh_grad operations
    #        Bn
    #        |
    #        v
    # An ->  Bn-1
    #        |
    #        v
    
    #        ...
    #        |
    #        v
    #        B1
    #        |
    #        v
    # A1 ->  B0
################################################################################


def backward_automatic(n):
  """Creates forward backward graph using tf.gradients."""

  def forward(A0, n):
    """Takes A0, applies n operations to it, returns An."""

    A = A0
    for L in range(1, n+1): # op_i produces A_i
      A = tf.tanh(A, name="A"+str(L))
    return A

  def backward(A0, An, Bn, n):
    B0 = tf.gradients([An], [A0], grad_ys=[Bn])[0]
    return B0

  A0 = tf.fill((size,), 1.0, name="A0")
  An = forward(A0, n)
  Bn = tf.fill((size,), 1.0, name="Bn")
  B0 = tf.gradients([An], [A0], grad_ys=[Bn])[0]
  return B0


def backward_manual(n):
  """Creates forward backward graph manually."""

  from tensorflow.python.ops import gen_math_ops
  tanh_grad = gen_math_ops._tanh_grad

  A = [None]*(n+1)
  A[0] = tf.fill((size,), 1.0, name="A0")
  for L in range(1, n+1):
    name = "A"+str(L)
    A[L] = tf.tanh(A[L-1], name=name)

  B = [None]*(n+1)
  with tf.control_dependencies([A[n]]):
    B[n] = tf.fill((size,), 1.0, name="B"+str(n))
  for L in range(n-1, -1, -1):
    name = "B"+str(L)
    B[L] = tanh_grad(A[L+1], B[L+1], name=name)

  return B[0]


def test_automatic_gradients():
  tf.reset_default_graph()
  sess = create_session()
  B0 = backward_automatic(2)
  assert abs(sess.run(B0)[0]-0.24686792)<1e-6
  

def test_manual_gradients():
  tf.reset_default_graph()
  sess = create_session()
  B0 = backward_manual(2)
  assert abs(sess.run(B0)[0]-0.24686792)<1e-6
  

def visualize_automatic_gradients():
  n = 3
  tf.reset_default_graph()
  B0 = backward_automatic(n)
  with open('chain_automatic_%d.pbtxt'%(n,), 'w') as f:
    f.write(str(tf.get_default_graph().as_graph_def()))

def visualize_manual_gradients():
  n = 3
  tf.reset_default_graph()
  B0 = backward_manual(n)
  with open('chain_manual_%d.pbtxt'%(n,), 'w') as f:
    f.write(str(tf.get_default_graph().as_graph_def()))

# visualize_manual_gradients

# visualize_manual_timeline
# visualize_automatic_timeline

def print_parsed_timeline(timeline, gpu_only=False, ignore_less_than_bytes=0):
  """pretty print parsed memory timeline."""

  total_memory = 0
  timestamps = []
  data = []
  first_timestamp = timeline[0][0]
  for record in timeline:
    timestamp, kernel_name, allocated_bytes, allocator_type = record
    allocated_bytes = int(allocated_bytes)
        
    if abs(allocated_bytes)<ignore_less_than_bytes:
      continue  # ignore small allocations
        
    total_memory += allocated_bytes
    print("%6d %10d %10d %s"%(timestamp-first_timestamp, total_memory,
                              allocated_bytes, kernel_name))

def get_cpu_timeline():
  global run_metadata
  cpu_stats = memory_util._retrieve_cpu_gpu_stats(run_metadata)[0]
  return memory_util.timeline_from_nodestats(cpu_stats)

def pp():
  cpu_stats = memory_util._retrieve_cpu_gpu_stats(run_metadata)[0]
  parsed_timeline = memory_util.timeline_from_nodestats(cpu_stats)
  print_parsed_timeline(parsed_timeline, ignore_less_than_bytes=0)

def visualize_memory():
  n = 3
  B0 = backward_manual(n)
  sess = create_session()

  run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()

  results = sess.run(B0.op,
                     options=run_options,
                     run_metadata=run_metadata)

  
  tl = timeline.Timeline(step_stats=run_metadata.step_stats)
  ctf = tl.generate_chrome_trace_format(show_memory=True,
                                          show_dataflow=False)
  fn_timeline = "chain_manual_%d_timeline.json"%(n,)
  open(fn_timeline, "w").write(ctf)

  fn_stepstats = "chain_manual_%d_stepstats.json" % (n,)
  open(fn_stepstats, "w").write(str(run_metadata.step_stats))

  pp()
  peak_memory = memory_util.peak_memory2(None, run_metadata)
  print("Peak memory: %.1f" %(peak_memory,))
  print("timeline dumped into ", fn_timeline)
  print("stepstats dumped into ", fn_stepstats)

def replace_input(op, old_input, new_input):
  """Replaces old input with new input in op"""
  assert old_input in op.inputs
  ge.reroute_ts([new_input], [old_input], can_modify=[op])


def recompute_tensor(target, known_values, preceding_op=None,
                     copy_known_values=False):
  """Computes target tensor from known_values. If preceding_op is not None,
  adds necessary control dependencies such that newly created computation takes
  place after preceding_op. 

  If copy_known_values is set, also copies known_values (for nicer graph
  visualization)
  """

  assert is_computable(target, known_values)
  
  # position of target in parent op
  target_pos = list(target.op.outputs).index(target)

  if copy_known_values:
    computation = ge.get_backward_walk_ops(target)
  else:
    computation = ge.get_backward_walk_ops(target, stop_at_ts=known_values)
    
  # create copy of computation
  copied_sgv, info = ge.copy_with_input_replacements(ge.sgv(computation), {})

  # find our target tensor in the new computation
  new_target_op = info._transformed_ops[target.op]
  new_target = new_target_op.outputs[target_pos]
  new_computation = list(info._transformed_ops.values())

  # restrict computation to run after given op
  SAVE_ON_CONTROL_EDGES = True

  if SAVE_ON_CONTROL_EDGES:
    # only add "run_after" control dependencies to root of computation,
    # the rest automatically runs after because of data dependencies
    # TODO: more efficient implementation by walking back from new_target
    # instead of whole graph
    computation_graph = linearize_lib.get_graph(restrict_to=new_computation)

    # note, toposort order is reversed from networkx/mine convention
    computation_root = list(toposort.toposort(computation_graph))[-1]
    for op in computation_root:
      run_after(op, preceding_op)
  else:
    if preceding_op is not None:
      for op in info._transformed_ops.values():
        run_after(op, preceding_op)
  return new_target

def replace_input(op, old_input, new_input):
  """Replaces old input with new input in op"""
  ge.reroute_ts([new_input], [old_input], can_modify=[op])

# TODO: rename to "before", after"
def run_after(a, b):
  """Rewrites the graph to run a after b."""
  already_after = (b in a.control_inputs) or (b in [i.op for i in a.inputs])

  if already_after:
    return 0
  ge.reroute.add_control_inputs(a, [b])
  return 1


def positions(ll, item):
  """Return all positions of item in list."""
  
  start_pos = 0
  position_list = []
  try:
    while True:
      pos = ll.index(item, start_pos)
      position_list.append(pos)
      start_pos = pos+1
  except ValueError:
    pass
  return position_list


def is_computable(result, known_values):
  """Returns true if given tensor is computable from known values."""

  computable_ops = ge.get_forward_walk_ops([val.op for val in known_values])
  return result.op in computable_ops

  
def visualize_rewritten_gradients():
  """Recompute gradients, visualize graph."""

  nice_visualization = False
  global sess
  
  n = 5

  A = [None]*(n+1)
  A[0] = tf.fill((size,), 1.0, name="A0")
  if nice_visualization:
    A[0] = tf.constant(np.ones((size,), np.float32), name="A0")
  for L in range(1, n+1):
    name = "A"+str(L)
    A[L] = tf.tanh(A[L-1], name=name)

  B = [None]*(n+1)
  #  with tf.control_dependencies([A[n]]):
  B[n] = tf.fill((size,), 1.0, name="B"+str(n))
  if nice_visualization:
    B[n] = tf.constant(np.ones((size,), np.float32), name="A0")
  if not nice_visualization:
    run_after(B[n].op, A[n].op)
  for L in range(n-1, -1, -1):
    name = "B"+str(L)
    B[L] = tanh_grad(A[L+1], B[L+1], name=name)

  sess = create_session()
  
  fwd_ops = ge.get_backward_walk_ops(A[n].op, stop_at_ts=[A[0]], inclusive=True)
  fwd_ops.append(A[0].op)  # backward_walk_ops doesn't include endpoint
  assert fwd_ops == [a.op for a in reversed(A)]
  
  # make sure to only provide tensors and not ops
  #  https://github.com/tensorflow/tensorflow/issues/14858
  bwd_ops = ge.get_forward_walk_ops(B[n].op, inclusive=True)

  prev_op = B[n]
  for op in bwd_ops:
    for op_input in op.inputs:
      if op_input.op not in fwd_ops:
        continue

      # recompute all tensors that have been computed before.
      # a tensor has been computed before if it's consumed by some op
      # in the forward graph
      input_consumers = op_input.op.outputs[0].consumers()
      fwd_consumers = [c for c in input_consumers if c in fwd_ops]
      bwd_consumers = [c for c in input_consumers if c in bwd_ops]
      if not fwd_consumers:
        print("tensor %s doesn't get used by fwd graph, skip copy" %(op_input.name))
        continue


      if nice_visualization:
        new_input = recompute_tensor(op_input, [A[0]], copy_known_values=True)
      else:
        new_input = recompute_tensor(op_input, [A[0]], preceding_op=prev_op)

        #      print("Replacing %s input with %s" %(op_input.name, new_input.name))
      replace_input(op, old_input=op_input, new_input=new_input)
    prev_op = op

  
  with open('chain_rewritten_%d.pbtxt'%(n,), 'w') as f:
    f.write(str(tf.get_default_graph().as_graph_def()))

  sess = create_session()
  sessrun(B[0].op)
  tl = timeline.Timeline(step_stats=run_metadata.step_stats)
  ctf = tl.generate_chrome_trace_format(show_memory=True,
                                          show_dataflow=False)
  with open('chain_rewritten_%d_timeline.json'%(n,), 'w') as f:
    f.write(ctf)

  with open('chain_rewritten_%d_stepstats.pbtxt'%(n,), 'w') as f:
    f.write(str(run_metadata.step_stats))

  pp()
  peak_memory = memory_util.peak_memory2(None, run_metadata)
  print("Peak memory: %.1f" %(peak_memory,))

  # bug workaround, https://github.com/tensorflow/tensorflow/issues/14233
  #  tf.get_default_graph().version+=1
  a = tf.constant(1)
  
  

def test_rewritten_gradients():
  """Test for rewriting method that gives constant memory usage (recompute
  everything."""

  global sess
  tf.reset_default_graph()
  gg = tf.get_default_graph()
  
  n = 20   # chain of length 20
  A = [None]*(n+1)
  A[0] = tf.fill((size,), 1.0, name="A0")
  for L in range(1, n+1):
    name = "A"+str(L)
    A[L] = tf.tanh(A[L-1], name=name)

  B = [None]*(n+1)
  B[n] = tf.fill((size,), 1.0, name="B"+str(n))
    
  run_after(B[n].op, A[n].op)
  for L in range(n-1, -1, -1):
    name = "B"+str(L)
    B[L] = tanh_grad(A[L+1], B[L+1], name=name)

  # for each op, obtain steps during which any output of this op is consumed
  sess = create_session()
  execution_order = linearize_lib.get_execution_order(B[0])
  consuming_schedule = OrderedDict()
  for op in gg.get_operations():
    consuming_ops = OrderedSet()  # OrderedSet for determinism
    for output in op.outputs:
      consuming_ops.update(output.consumers())
    consuming_schedule[op] = [execution_order.index(c) for c in consuming_ops]

  for step, op in enumerate(execution_order):
    for op_input in op.inputs:
      # get all the times when this input is consumed
      consume_times = consuming_schedule[op_input.op]
      assert step in consume_times

      # if it's been consumed before, save memory by recomputing it
      consumed_before = len([t for t in consume_times if t<step]) > 0
      if consumed_before:
        assert step>0
        # want recomputation to happen as late as possible, schedule to run
        # it after the op that was scheduled to execute right before this op
        prev_op = execution_order[step-1]
        new_input = recompute_tensor(op_input, known_values=[A[0]],
                                     preceding_op=prev_op)
        replace_input(op, old_input=op_input, new_input=new_input)

  sess = create_session()
  sessrun(B[0].op)
  peak_memory = memory_util.peak_memory2(None, run_metadata)
  parsed_timeline = get_cpu_timeline()
  desired_peak = 3000000.0
  assert len(parsed_timeline) == 258  # change detector
  assert peak_memory == desired_peak, "%.2f MB observed, %.2f expected" % (peak_memory, desired_peak)


def main():
  #  test_automatic_gradients()
  #  test_manual_gradients()
  #  visualize_automatic_gradients()
  #  visualize_manual_gradients()
  #  visualize_memory()
  #  visualize_rewritten_gradients()
  test_rewritten_gradients()
  
if __name__=='__main__':
  main()
