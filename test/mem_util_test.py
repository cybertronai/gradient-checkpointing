# tests for memory tracking routines

import pytest
import tensorflow as tf

import mem_util
import util

size_mbs = 1   # size of nodes in MB
size = size_mbs * 250000


def _chain_backprop(n):
  """Creates forward backward graph using tf.gradients.

  A0->A1->A2->..->An
    /    /       /
  B0<-B1<-B2<-..<-Bn
  """

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

run_metadata = None
DO_TRACING = True
def sessrun(*args, **kwargs):
  """Helper method to use instead of sess.run that will automatically
  capture run_metadata."""
  global sess, run_metadata
  
  if not DO_TRACING:
    return sess.run(*args, **kwargs)
  
  run_metadata = tf.RunMetadata()
  kwargs['options'] = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  kwargs['run_metadata'] = run_metadata
  result = sess.run(*args, **kwargs)
  return result


def create_session():
  """Create session with optimizations disabled."""
  from tensorflow.core.protobuf import rewriter_config_pb2
  optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
  config = tf.ConfigProto(operation_timeout_in_ms=150000, graph_options=tf.GraphOptions(optimizer_options=optimizer_options))
  config.graph_options.rewrite_options.constant_folding=rewriter_config_pb2.RewriterConfig.OFF
  config.graph_options.place_pruned_graph = True
  return tf.Session(config=config)


def test_peak():
  global sess, run_metadata
  tf.reset_default_graph()
  
  # create backprop for A0->A1->A2->A3
  with tf.device("/cpu:0"):
    b0 = _chain_backprop(3)

  # this needs 4 MB of memory
  # A0/A1 share memory since A0 is not consumed by anyone, therefore at peak
  # we have A1,A2,A3,B0 stored in memory
  
  sess = create_session()
  sessrun(b0.op)
  peak_cpu = mem_util.peak_memory(run_metadata)['/cpu:0']
  assert abs(peak_cpu - 4e6) < 1e4


@pytest.mark.skipif(not tf.test.is_gpu_available(), reason="requires GPU")
def test_peak_gpu():
  global sess, run_metadata
  tf.reset_default_graph()
  
  assert tf.test.is_gpu_available(), "This test requires GPU"
  # create backprop for A0->A1->A2->A3
  with tf.device("/cpu:0"):
    b0 = _chain_backprop(3)

  # create backprop for A0->A1->A2->A3
  with tf.device("/gpu:0"):
    c0 = _chain_backprop(3)

  sess = create_session()
  sessrun(tf.group(b0.op, c0.op))
  peak_cpu = mem_util.peak_memory(run_metadata)['/cpu:0']
  peak_gpu = mem_util.peak_memory(run_metadata)['/gpu:0']
  assert abs(peak_cpu - 4e6) < 1e4
  assert abs(peak_gpu - 4e6) < 1e4
  
  
@pytest.mark.skip(reason="can't run under pytest since it intercepts stdout")
def test_print():
  global sess, run_metadata
  tf.reset_default_graph()
  
  with tf.device("/cpu:0"):
    b0 = _chain_backprop(3)
  sess = create_session()
  sessrun(b0.op)
  with util.capture_stdout() as stdout:
    mem_util.print_memory_timeline(run_metadata)

  4# should print something like this
  #    0          0          0 _SOURCE
  #   31          0          0 A0/dims
  #   47          0          0 A0/value
  #   55          0          0 Bn/dims
  #   59          0          0 Bn/value
  #   70    1000000    1000000 Bn
  #   95    2000000    1000000 A0
  #  436    2000000          0 gradients/grad_ys_0
  #  587    2000000          0 A1
  #  732    3000000    1000000 A2
  # 1308    4000000    1000000 A3
  # 2026    4000000          0 gradients/A3_grad/TanhGrad
  # 2102    3000000   -1000000 Bn
  # 2108    3000000          0 gradients/A2_grad/TanhGrad
  # 2165    2000000   -1000000 A3
  # 2170    2000000          0 gradients/A1_grad/TanhGrad
  # 2224    1000000   -1000000 A2
  # 2227          0   -1000000 A0

  print(stdout.getvalue().strip())
  assert('4000000    1000000 A3' in stdout.getvalue())

  
def main():
  global run_metadata, sess
  
  test_peak()
  test_print()
  test_peak_gpu()
  
if __name__=='__main__':
  main()
