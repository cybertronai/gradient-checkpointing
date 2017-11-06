import tensorflow as tf
import memory_util
import numpy as np


def memory_timeline_from_nodestats(nodestats):
  lines = []
  for node in nodestats:
    mem = node.memory
    assert(len(mem) == 1), str(node)
    records = mem[0].allocation_records
    if len(records)>0:
      assert len(records)<=2
      for record in records:
        line = [record.alloc_micros, node.node_name, record.alloc_bytes, "unknown"]
        lines.append(line)
    else:
      output_bytes = -1
      try:
        output_bytes = node.output[0].tensor_description.allocation_description.requested_bytes
      except:
        pass
      line = [node.all_start_micros, node.node_name, 0, "unknown"]
      lines.append(line)
  def first_key(x): return x[0]
  return sorted(lines, key=first_key)


def plot_parsed_timeline(timeline, gpu_only=False, ignore_less_than_bytes=1000):
  total_memory = 0
  timestamps = []
  data = []
  current_time = 0
  for record in timeline:
    timestamp, kernel_name, allocated_bytes, allocator_type = record
    allocated_bytes = int(allocated_bytes)

    if abs(allocated_bytes)<ignore_less_than_bytes:
      continue  # ignore small allocations
    if gpu_only:
      if not record[3].lower().startswith("gpu"):
        continue
    timestamps.append(current_time-.00000001)
    data.append(total_memory)
    total_memory += int(record[2])
    timestamps.append(current_time)
    data.append(total_memory)
    current_time+=1
  plt.plot(timestamps, data)


def print_parsed_timeline(timeline, gpu_only=False, ignore_less_than_bytes=0):
  """Prints timeline from stdout log."""

  total_memory = 0
  timestamps = []
  data = []
  first_timestamp = timeline[0][0]
  peak_memory = -123
  for record in timeline:
    timestamp, kernel_name, allocated_bytes, allocator_type = record
    allocated_bytes = int(allocated_bytes)

    if abs(allocated_bytes)<ignore_less_than_bytes:
      continue  # ignore small allocations

    total_memory += allocated_bytes
    peak_memory = max(total_memory, peak_memory)
    print("%6d %10d %10d %s"%(timestamp-first_timestamp, total_memory, allocated_bytes, kernel_name, ))
  return peak_memory

def make_chain_tanh_augmented(length=100, name_prefix="a", node_mbs=1):
  """Increase dimensionality of backprop."""
  
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

  from tensorflow.python.ops import math_grad as math_grad
  tanh_grad = math_grad._TanhGrad

  aug_size = 496
  aug_n = int(aug_size/4)
  backprop = tf.fill((n+aug_n,), val)
  while nodes:
    node = nodes.pop()
    aug_node = tf.concat([node, tf.ones((aug_n,))], 0)
    backprop = tanh_grad(aug_node.op, backprop)
  return backprop


def _retrieve_cpu_gpu_stats(run_metadata):
  cpu_stats = None
  gpu_stats = None
  step_stats = run_metadata.step_stats
  for ds in step_stats.dev_stats:
    if "cpu:0" in ds.device[-5:].lower():
      cpu_stats = ds.node_stats
    if "gpu:0" == ds.device[-5:].lower():
      gpu_stats = ds.node_stats
  return cpu_stats, gpu_stats


def main():
  sess = tf.Session()
  run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()

  with tf.device("/cpu:0"):
    grad = make_chain_tanh_augmented(5)

  sess.run(tf.global_variables_initializer())
  sess.run(grad.op, options=run_options,
                     run_metadata=run_metadata)
  cpu, gpu = _retrieve_cpu_gpu_stats(run_metadata)
  
  result = print_parsed_timeline(memory_timeline_from_nodestats(_retrieve_cpu_gpu_stats(run_metadata)[0]))

  target = 7001984
  # todo: make this deterministic with linearize_lib to turn into proper test
  if result != target:
    print("Expected {:,} bytes observed {:,} bytes".format(target, result))

if __name__=='__main__':
  main()
