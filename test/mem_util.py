# Utilities to figure out memory usage of run call
#
# Usage:
#   import mem_util
#   run_metadata = tf.RunMetadata()
#   options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#   sess.run(tensor, options=options, run_metadata=run_metadata)
#   print(mem_util.peak_memory(run_metadata))
#
#  To print memory usage for particular device:
#   print(mem_util.peak_memory(run_metadata)["/gpu:0"])


# Developer notes:
# RunMetadata
# https://github.com/tensorflow/tensorflow/blob/fc49f43817e363e50df3ff2fd7a4870ace13ea13/tensorflow/core/protobuf/config.proto#L349
#
# StepStats (run_metadata.step_stats)
# https://github.com/tensorflow/tensorflow/blob/a2d9b3bf5f9e96bf459074d079b01e1c74b25afa/tensorflow/core/framework/step_stats.proto
#
# NodeExecStats (run_metadata.step_stats.dev_stats[0].step_stats[0])
# https://github.com/tensorflow/tensorflow/blob/a2d9b3bf5f9e96bf459074d079b01e1c74b25afa/tensorflow/core/framework/step_stats.proto#L52


# Note, there are several methods of tracking memory allocation. There's
# requested bytes, and allocated bytes. Allocator may choose to give more bytes
# than is requested. Currently allocated_bytes is used to give more realistic
# results
#
# allocation_description {
#   requested_bytes: 1000000
#   allocated_bytes: 1000192
#   allocator_name: "GPU_0_bfc"
#
#
# There's also additional field in NodeExecStats which tracks allocator state
# node_stats {
#   node_name: "Bn_1/value"
#   all_start_micros: 1512081861177033
#   op_start_rel_micros: 1
#   op_end_rel_micros: 3
#   all_end_rel_micros: 5
#   memory {
#     allocator_name: "GPU_0_bfc"
#     allocator_bytes_in_use: 3072
#   }
#
# Additionally one could use LOG_MEMORY messages to get memory allocation info
# See multiple_memory_obtain_test.py for details on using these additional
# methods

def peak_memory(run_metadata):
  """Return dictionary of peak memory usage (bytes) for each device.

  {"cpu:0": 20441, ...
  """
  
  assert run_metadata != None
  assert hasattr(run_metadata, "step_stats")
  assert hasattr(run_metadata.step_stats, "dev_stats")
  
  dev_stats = run_metadata.step_stats.dev_stats
  result = {}
  for dev_stat in dev_stats:
    device_name = _simplify_device_name(dev_stat.device)
    result[device_name] = _peak_from_nodestats(dev_stat.node_stats)
  return result


def _timeline_from_nodestats(nodestats):
  """Return sorted memory allocation records from list of nodestats
  [NodeExecStats, NodeExecStats...], it's the
  run_metadata.step_stats.dev_stats[0].step_stats object.

  Timeline looks like this:

 timestamp         nodename  mem delta, allocator name
 [1509481813012895, 'concat', 1000496, 'cpu'],
 [1509481813012961, 'a04', -1000000, 'cpu'],
 [1509481813012968, 'TanhGrad', 0, 'cpu'], 

  0 memory allocation is reported for nodes without allocation_records
  """
  
  lines = []
  if not nodestats:
    return []
  for node in nodestats:
    for mem in node.memory:  # can have both cpu and gpu allocator for op
      try:
        records = mem.allocation_records
      except:
        records = []
      allocator = mem.allocator_name
      if len(records)>0:
        for record in records:
          line = [record.alloc_micros, node.node_name, record.alloc_bytes,
                  allocator]
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

# todo: get rid of "timeline_from_nodestats"

def _position_of_largest(my_list):
  """Return index of largest entry """
  import operator
  index, value = max(enumerate(my_list), key=operator.itemgetter(1))
  return index

def _peak_from_nodestats(nodestats):
  """Given a list of NodeExecStats messages, construct memory timeline."""
  
  timeline = _timeline_from_nodestats(nodestats)
  timestamps = []
  data = []

  total_memory = 0
  peak_memory = total_memory
  for record in timeline:
    timestamp, kernel_name, allocated_bytes, allocator_type = record
    allocated_bytes = int(allocated_bytes)
    total_memory += allocated_bytes
    peak_memory = max(total_memory, peak_memory)
  return peak_memory


def _print_parsed_timeline(timeline, gpu_only=False, ignore_less_than_bytes=0):
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


def _simplify_device_name(device):
  """/job:localhost/replica:0/task:0/device:CPU:0 -> /cpu:0"""

  prefix = '/job:localhost/replica:0/task:0/device:'
  if device.startswith(prefix):
    device = '/'+device[len(prefix):]
  return device.lower()
  

def _device_stats_dict(run_metadata):
  """Returns dictionary of device_name->[NodeExecStats, NodeExecStats...]"""
  result = {}
  for dev_stat in run_metadata.step_stats.dev_stats:
    device_name = _simplify_device_name(dev_stat.device)
    result[device_name] = dev_stat.node_stats

  return result


def print_memory_timeline(run_metadata, device=None):
  """Human readable timeline of memory allocation/deallocation for given
  device. If device is None, prints timeline of the device with highest memory
  usage"""
  

  if device is None:
    peak_dict = peak_memory(run_metadata)
    peak_pairs = list(peak_dict.items())
    chosen_peak = _position_of_largest([peak for (dev, peak) in peak_pairs])
    device = peak_pairs[chosen_peak][0]


  device_metadata = _device_stats_dict(run_metadata)
  print("Printing timeline for "+device)
  parsed_timeline = _timeline_from_nodestats(device_metadata[device])
  _print_parsed_timeline(parsed_timeline)
