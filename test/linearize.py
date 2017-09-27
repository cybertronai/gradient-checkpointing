"""Library to force a memory-efficient execution order on TensorFlow graph.
"""

import tensorflow as tf
import tensorflow.contrib.graph_editor as ge

from collections import OrderedDict

DEBUG = False

def run_after(a, b):
  """Force operation a to run after b. Do not add control dependencies
  to ops that already run after. Returns 0 if no dependencies were added,
  1 otherwise."""

  already_after = (b in a.control_inputs) or (b in [i.op for i in a.inputs])

  if already_after:
    return 0
  ge.reroute.add_control_inputs(a, [b])
  return 1


controls = None
controls_graph = None
def initialize_control_outputs(g):
  global controls
  global controls_graph
  controls_graph = g
  controls = tf.contrib.graph_editor.ControlOutputs(g)


################################################################################
# Graph ops
################################################################################
# Computation flows from parents to children.
#
# Two sets of dependencies, regular (parents/children) and regular + control
# parents_with_controls, children_with_controls.
#
# The former is more relevant for estimating memory usage because a control
# dependency on node doesn't imply that its outputs have to be kept in memory,
# whereas input dependency does.

# TODO(y): rename for consistency with memsortedbvb
def nodesort(ops):
  """sort list by op.name."""
  return sorted(ops, key=lambda op: op.name)


def parents_with_controls(op):
  result = set(input.op for input in op.inputs)
  result.update(op.control_inputs)
  return nodesort(result)


def parents(op):
  return nodesort(set(input.op for input in op.inputs))


def children(op):
  return nodesort(set(op_ for out in op.outputs for op_ in out.consumers()))


def children_with_controls(op):
  """Returns children, counting control outputs as children."""

  if not controls:
    initialize_control_outputs(op.graph)
  assert controls_graph == op.graph, ("graph changed, rerun "
                                      "initialize_control_outputs")
  result = set(op for out in op.outputs for op in out.consumers())
  result.update(controls.get(op))
  return nodesort(result)


def get_graph(g=None, as_hashes=False, exclude_controls=False):
  """Creates dictionary {node: {child1, child2, ..},..} for current
  TensorFlow graph (control edges included).

  Args:
    g: tf.Graph to use or None, in which case take default graph
    as_hashes: whether to replace nodes with their hashes (needed for
                                                           nx.DiGraph)
    exclude_controls: if True, don't count control deps as edges
  """

  if not g:
    g = tf.get_default_graph()

  result = OrderedDict()
  for op in nodesort(g.get_operations()):
    if as_hashes:
      key = hash(op)
    else:
      key = op
    if exclude_controls:
      result[key] = children(op)
    else:
      result[key] = children_with_controls(op)
  return result

def print_tf_graph(graph):
  """Prints tensorflow graph in dictionary form."""

  for node in graph:
    for child in graph[node]:
      print("%s -> %s" % (node.name, child.name))


def memsorted(nodes):
  """Sort nodes by estimated memory usage."""

  def node_memory(unused_node, default_memory=1):
    return default_memory
  def node_name(node):
    return node.name

  def subtree_memory(node):
    return (node_memory(node) + sum(node_memory(parent) for
                                    parent in parents(node)))

  # sort by estimated memory, break ties in reverse alphabetical order
  # this gives regular alphabetic order when used with linearize
  nodes = sorted(nodes, key=node_name, reverse=True)
  nodes = sorted(nodes, key=subtree_memory)
  return nodes


def is_iterable(o):
  try:
    _ = iter(o)
  except Exception:
    return False
  return True


# TODO(y): prevent deadlocks by avoiding linearizing anything with downstream
# Assign dependencies
def linearize(targets=None, modify_graph=True):
  """Obtain a single valid execution order which approximately minimizes
  peak memory usage.

  Args:
    targets: specifies list of computation Tensor or op targets. Nones are
        skipped. If not specified, all terminal nodes that are not Assign
        /AssignAdd/NoOp nodes are considered targets. The reason for excluding
        these ops is because they are usually targets for variable modifying
        session.run calls, whereas linearization assumes graph is purely
        functional.
    modify_graph: if True, will add control dependencies to force this order

  Returns:
    Number of control dependencies that were added if modify_graph=True,
    otherwise returns list of ops in this order.
  """

  g = tf.get_default_graph()
  initialize_control_outputs(g)  # needed for children_with_controls
  graph = get_graph(g)

  # The algorithm works by keeping an "active" set nodes that have no
  # unscheduled children, hence are ready for execution.  At each iteration,
  # schedule all nodes from this set for execution with least memory-hungry
  # nodes first and repeat to convergence.

  def to_op(tensor_or_op):
    if hasattr(tensor_or_op, "op"):
      return tensor_or_op.op
    return tensor_or_op
  
  if is_iterable(targets):
    active = [to_op(target) for target in targets if target is not None]
  elif targets is not None:
    active = [to_op(targets)]
  else:
    active = []
    for node in graph:
      if not graph[node]:  # no children
        if (node.type != "Assign" and node.type != "AssignAdd" and
            node.type != "NoOp"):
          active.append(node)

  last_node = None

  # count of unscheduled children for each node
  unscheduled = OrderedDict()
  for node in graph:
    unscheduled[node] = len(graph[node])
  control_edges_added = 0
  order = []
  while active:
    new_active = []
    for node in memsorted(active):
      assert unscheduled[node] == 0

      order.append(node)
      if DEBUG:
        print("Executing ", node.name)

      if last_node:
        if modify_graph:
          control_edges_added += run_after(last_node, node)
      last_node = node
        

      # this node is scheduled, so update unscheduled counts of parents
      # TODO(y) do we need this second memsorted?
      for parent in memsorted(parents_with_controls(node)):
        assert unscheduled[parent] > 0
        unscheduled[parent] -= 1
        if unscheduled[parent] == 0:
          new_active.append(parent)

    active = new_active

  if modify_graph:
    return control_edges_added
  else:
    return reversed(order)
