"""Utilities for generating/enforcing memory-efficient execution order on a
TensorFlow graph.
"""
import pdb
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
from toposort import toposort

from collections import OrderedDict
import util
import networkx as nx

DEBUG = False


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
#
# Node that for toposort, children are dependencies, so order of graph is
# reversed.


def _run_after(a, b):
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
def initialize_control_outputs(g=None):
  global controls
  global controls_graph
  if g is None:
    controls_graph = tf.get_default_graph()
  else:
    controls_graph = g
  controls = tf.contrib.graph_editor.ControlOutputs(controls_graph)


def alphasorted(ops):
  """sort list by op.name."""
  return sorted(ops, key=lambda op: op.name)


def parents_with_controls(op):
  result = set(input.op for input in op.inputs)
  result.update(op.control_inputs)
  return alphasorted(result)


def parents(op):
  return alphasorted(set(input.op for input in op.inputs))


def children(op, restrict_to=None):
  result = set(op_ for out in op.outputs for op_ in out.consumers())
  if restrict_to is not None:
    restrict_to = to_ops(restrict_to)
    result = set(op for op in result if op in restrict_to)
  return alphasorted(result)


def children_with_controls(op, restrict_to=None):
  """Returns children, counting control outputs as children."""

  if not controls:
    initialize_control_outputs(op.graph)
  assert controls_graph == op.graph, ("graph changed, rerun "
                                      "initialize_control_outputs")
  result = set(op for out in op.outputs for op in out.consumers())
  result_controls = controls.get(op)
  if restrict_to is not None:
    result = set(op for op in result if op in restrict_to)
    result_controls = set(op for op in result_controls if op in restrict_to)
  result.update(result_controls)
  return alphasorted(result)


def to_op(tensor_or_op):
  if hasattr(tensor_or_op, "op"):
    return tensor_or_op.op
  return tensor_or_op

# Todo: dedup with memory_saving_gradients.to_ops?
def to_ops(ll):
  if ll is None:
    return None
  elif is_list_or_tuple(ll):
    return [to_op(t) for t in ll]
  else:
    assert False, ("op list has unsupported type %s, must be list or "
                   "tuple"%(ll.__class__.__name__))



def print_ops(ops):
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


# TODO: change to OrderedSet, fix tests
def get_graph(g=None, as_names=False, as_hashes=False, as_indices=False,
              exclude_controls=False,
              restrict_to=None):
  """Creates dictionary {node: {child1, child2, ..},..} for current
  TensorFlow graph.

  Args:
    g: tf.Graph to use or None, in which case take default graph
    as_hashes: whether to replace nodes with their hashes (needed for
                                                           nx.DiGraph)
    exclude_controls: if True, don't count control deps as edges
    restrict_to: restricts to given set of nodes
  """

  if not g:
    g = tf.get_default_graph()

  assert not (as_hashes and as_names)
  assert not (as_hashes and as_indices)
  assert not (as_names and as_indices)
  initialize_control_outputs(g=g)

  result = OrderedDict()
  if restrict_to is not None:
    restrict_to = to_ops(restrict_to)

  def format_children(children):
    if as_names:
      return set([op.name for op in children])
    if as_hashes:
      return set([hash(op) for op in children])
    return set(children)
      
  for op in alphasorted(g.get_operations()):
    if restrict_to is not None and op not in restrict_to:
      continue
    if as_hashes:
      key = hash(op)
    if as_names:
      key = op.name
    else:
      key = op
    if exclude_controls:
      result[key] = format_children(children(op, restrict_to=restrict_to))
    else:
      result[key] = format_children(children_with_controls(op, restrict_to=restrict_to))

  # renumber nodes as sequential indices
  if as_indices:
    nodes = set([child for ll in result.values() for child in ll])
    nodes.update(result.keys())
    m = {node: idx+1 for idx,node in enumerate(nodes)}
    return {m[key]:set(m[v] for v in result[key]) for key in
            result.keys()}
  
  return result

def prune_graph(graph, targets):
  """Return parts of the graph needed to compute targets."""

  targets = to_ops(targets)
  parent_graph = reversed_graph(graph)
  visited = OrderedSet()

  active = targets
  wave_number = 0
  while active:
    #    print(active)
    wave_number+=1
    new_active = OrderedSet()
    for node in memsorted(active):
      new_active.update(parent_graph[node])
    visited.update(active)
    active = new_active

  # convert graph to edges
  edges = []
  new_graph = OrderedDict()
  for parent in graph:
    for child in graph[parent]:
      edges.append((parent, child))

  edges_pruned = [e for e in edges if (e[0] in visited and e[1] in visited)]

  nodes = OrderedSet(node for edge in edges_pruned for node in edge)
  nodes.update(targets)   # singleton targets nodes

  for node in nodes:
    new_graph[node] = OrderedSet()
  
  for (parent, child) in edges_pruned:
    container = OrderedSet()
    new_graph.setdefault(parent, container).update([])
    if child != parent:
      new_graph[parent].update([child])

  return new_graph
  

def copy_graph(graph):
  """Return parts of the graph needed to compute targets."""

  new_graph = OrderedDict()
  for node in graph:
    new_graph[node] = OrderedSet()
    for child in graph[node]:
      new_graph[node].update([child])
  return new_graph
  

def _is_variable_op(op):
  if ('Assign' in op.type or 'Apply' in op.type or 'Variable' in op.type or
      'VariableV2' in op.type or 'VarHandleOp' in op.type):
    return True
  return False

def remove_variable_ops_from_graph(graph):
  """Remove ops which either require initialization (Variable) or update values.
  This ensure that linearizing the whole graph doesn't break initialization."""
  new_graph = OrderedDict()
  for parent in graph:
    if _is_variable_op(parent):
      continue
    new_graph[parent] = OrderedSet()
    for child in graph[parent]:
      if not _is_variable_op(child):
        new_graph[parent].update([child])
  return new_graph
    

def print_graph(graph):
  """Prints tensorflow graph in dictionary form."""

  for node in graph:
    for child in graph[node]:
      print("%s -> %s" % (node.name, child.name))
    if not graph[node]: 
      print("%s" % (node.name,))

class OrderedSet:
  def __init__(self, items=None):
    self.d = OrderedDict()
    self.update(items)
    
  def update(self, items):
    if items is not None:
      for item in items:
        self.d[item] = 1
  def __iter__(self):
    return self.d.__iter__()

  def __contains__(self, key):
    return self.d.__contains__(key)

  def __delitem__(self, key):
    return self.d.__delitem__(key)

  def __len__(self):
    return self.d.__len__()

  def add(self, x):
    return update(self, [x])

  def discard(self, x):
    if self.__contains__(x):
      return self.__del__(x)

  def remove(self, x):
    if not self.__contains__(x):
      raise KeyError
    return self.__del__(x)

  def _format_op(self, op):
    if hasattr(op, 'name'):
      return op.name
    return str(op)
  
  def __repr__(self):
    if not self:
      return '%s()' % (self.__class__.__name__,)
    return '{%r}' % (','.join([self._format_op(op) for op in self]),)

  # def __equal__
  #  def pop...
  #  def clear()...

def reversed_graph(graph, deterministic=False):
  """Reverses direction of all edges in the graph."""

  edges = []
  for parent in graph:
    for child in graph[parent]:
      edges.append((parent, child))
  edges = [(child, parent) for (parent, child) in edges]
  nodes = OrderedSet(node for edge in edges for node in edge)
  nodes.update(graph.keys())   # singleton nodes

  new_graph = OrderedDict()
  for node in nodes:
    if deterministic:
      container = OrderedSet()
    else:
      container = set()
    new_graph[node] = container

  for (parent, child) in edges:
    if child != parent:
      new_graph[parent].update([child])
  return new_graph


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

def is_list_or_tuple(o):
  return isinstance(o, list) or isinstance(o, tuple)

def obtain_linear_order(targets=None):
  return linearize(targets=targets, modify_graph=False)


def _process_targets(targets):
  """Helper utility to help variation in targets input type:
  -- it can be iterable vs simple list
  -- it can contain Operation objects or Tensor objects, convert to Operation
  -- it can be a single target

  Ignores variable ops (Variable and Assign) when making graph because adding
  dependencies to those breaks initialization.

  Returns: graph in "source_node->target_node" form
           targets as list of Operation objects
  """
  
  g = tf.get_default_graph()
  graph = get_graph(g)

  if is_list_or_tuple(targets):
    targets = to_ops(targets)  # convert Tensors to ops if needed
    targets = [t for t in targets if t is not None]
  elif targets is not None:
    targets = [to_op(targets)]
  else:
    targets = list(graph.keys())

  graph = prune_graph(graph, targets)
  graph = remove_variable_ops_from_graph(graph)
  return graph, targets

def get_execution_order(targets=None):
  """Return deterministic execution order which approximately minimizes peak
  memory usage.
  Args:
    targets: specifies list of computation Tensor or op targets or a single
        target.

  Returns:
    list of Operation objects in execution order."""

  return linearize(targets, modify_graph=False)

def linearize(targets=None, modify_graph=True):
  """Obtain a single valid execution order which approximately minimizes
  peak memory usage.

  TODO: deprecate/hide modify_graph arg
  Args:
    targets: specifies list of computation Tensor or op targets or a single
        target.
        skipped. If None, all nodes are considered targets.
    modify_graph: if True, will add control dependencies to force this order

  Returns:
    Number of control dependencies that were added if modify_graph=True,
    otherwise returns list of ops in this order.
  """

  graph, targets = _process_targets(targets)
  parent_graph = reversed_graph(graph)

  toposort(copy_graph(graph))   # check for cycles (raises exception)

  # The algorithm works by keeping an "active" set nodes that have no
  # unscheduled children, hence are ready for execution.  At each iteration,
  # schedule all nodes ready for for execution with least memory-hungry
  # nodes first and repeat to convergence.
  last_node = None

  # count of unscheduled children for each node
  unscheduled = OrderedDict()
  active = []
  for node in graph:
    unscheduled[node] = len(graph[node])
    if unscheduled[node] == 0:
      active.append(node)

  assert len(active)>0, "List of targets contains a cycle"

  for node in active:
    assert unscheduled[node] == 0
  
  control_edges_added = 0
  order = []
  wave_number = 0
  while active:
#    print("wave %d, %d"%(wave_number, len(active)))
    wave_number+=1
    new_active = []
    for node in memsorted(active):
      assert unscheduled[node] == 0

      order.append(node)
      if DEBUG:
        print("Executing ", node.name)

      if last_node:
        if modify_graph:
          control_edges_added += _run_after(last_node, node)
      last_node = node
        

      # this node is scheduled, so update unscheduled counts of parents
      for parent in parent_graph[node]:
        assert unscheduled[parent] > 0
        unscheduled[parent] -= 1
        if unscheduled[parent] == 0:
          new_active.append(parent)
#          print("Adding %s to active" % (parent,))

    active = new_active  # end while

  if modify_graph:
    return control_edges_added
  else:
    result = list(reversed(order))
    return result

def _sort(nodes, total_order, dedup=False):
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
    nodes = OrderedSet(nodes)
  return sorted(nodes, key=lambda n: total_order_idx[n])


def sorted_articulation_points(targets):
  """Returns list of articulation points (cut vertices) sorted in according
  to the execution order provided by linearize."""
  
  graph, targets = _process_targets(targets)
  sorted_list = list(obtain_linear_order(targets))

  nx_graph = nx.Graph(graph)
  points = _sort(nx.articulation_points(nx_graph),
                 total_order=sorted_list, dedup=True)
  return points

