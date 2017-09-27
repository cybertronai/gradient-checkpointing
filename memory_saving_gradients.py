import numpy as np
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
from toposort import toposort
import networkx as nx
import math
import contextlib
import time

from tensorflow.python.ops import gradients as tf_gradients_lib
tf_gradients = tf_gradients_lib.gradients

# backward compatibility
if 'reroute_a2b_ts' in dir(ge):
    my_reroute_ts = ge.reroute_a2b_ts
else:
    assert 'reroute_ts' in dir(ge)
    my_reroute_ts = ge.reroute_ts



# refers back to current module if we decide to split helpers out
import sys
util = sys.modules[__name__]   

import linearize as linearize_lib

# specific versions we can use to do process-wide replacement of tf.gradients
def gradients_speed(ys, xs, grad_ys=None, **kwargs):
    return gradients(ys, xs, grad_ys, remember='speed', **kwargs)

def gradients_memory(ys, xs, grad_ys=None, **kwargs):
    return gradients(ys, xs, grad_ys, remember='memory', **kwargs)
        
def gradients_tarjan(ys, xs, grad_ys=None, **kwargs):
    return gradients(ys, xs, grad_ys, remember='tarjan', **kwargs)

#MERGE!
# change default to collection
def gradients(ys, xs, grad_ys=None, remember='speed', **kwargs):
    '''
    Authors: Tim Salimans & Yaroslav Bulatov, OpenAI

    memory efficient gradient implementation inspired by "Training Deep Nets with Sublinear Memory Cost"
    by Chen et al. 2016 (https://arxiv.org/abs/1604.06174)

    ys,xs,grad_ys,kwargs are the arguments to standard tensorflow tf.gradients
    (https://www.tensorflow.org/versions/r0.12/api_docs/python/train.html#gradients)

    'remember' can either be
        - a list consisting of tensors from the forward pass of the neural net
          that we should re-use when calculating the gradients in the backward pass
          all other tensors that do not appear in this list will be re-computed
        - a string specifying how this list should be determined. currently we support
            - 'speed':  remember all outputs of convolutions and matmuls. these ops are usually the most expensive,
                        so remembering them maximizes the running speed
                        (this is a good option if nonlinearities, concats, batchnorms, etc are taking up a lot of memory)
            - 'memory': try to minimize the memory usage
                        (currently using a very simple strategy that identifies a number of bottleneck tensors in the graph to remember)
            - 'collection': look for a tensorflow collection named 'remember', which holds the tensors to remember
           - 'tarjan': use Tarjan's algorithm to find articulation points, and use those are remember tensors
    '''
    print("Calling memsaving gradients with ", remember)
    if not isinstance(ys,list):
        ys = [ys]
    if not isinstance(xs,list):
        xs = [xs]

    #    print("Calling memory_saving_gradients")
    # get "forward" graph
    # MERGE!:
    # tim version had
    #      forward_inclusive=False, backward_inclusive=True)
    fwd_seeds = [x.op for x in xs]
    bwd_seeds = [y.op for y in ys]

    # todo: do we need control_inputs like in intersection ops?
    bwd_ops = ge.get_backward_walk_ops(bwd_seeds,
                                       inclusive=True)
    fwd_ops = ge.get_forward_walk_ops(fwd_seeds,
                                      inclusive=True,
                                      within_ops=bwd_ops)
    #        fwd_ops = ge.get_walks_intersection_ops(forward_seed_ops=fwd_seeds, backward_seed_ops=bwd_seeds, forward_inclusive=True, backward_inclusive=True)

    # remove all placeholders, variables and assigns
    fwd_ops = [op for op in fwd_ops if op._inputs]
    fwd_ops = [op for op in fwd_ops if not '/read' in op.name]
    fwd_ops = [op for op in fwd_ops if not '/Assign' in op.name]
    fwd_ops = [op for op in fwd_ops if not '/assign' in op.name]

    # construct list of tensors to remember from forward pass, if not given as input
    if type(remember) is not list:
        if remember == 'collection':
            remember = tf.get_collection('remember')

        elif remember == 'speed':
            # remember all expensive ops to maximize running speed
            remember = ge.filter_ts_from_regex(fwd_ops, 'conv2d|Conv|MatMul')

            
        elif remember == 'memory':

            # get all tensors in the fwd graph
            ts = ge.filter_ts(fwd_ops, True)
            debug_print("Filtering tensors: %s", ts)

            # filter out all tensors that are inputs of the backward graph
            with util.capture_ops() as bwd_ops:
                gs = tf_gradients(ys, xs, grad_ys, **kwargs)

            bwd_inputs = [t for op in bwd_ops for t in op.inputs]
            # list of tensors in forward graph that is in input to bwd graph
            ts_filtered = list(set(bwd_inputs).intersection(ts))
            debug_print("Using tensors %s", ts_filtered)

            for rep in range(2): # try two slightly different ways of getting bottlenecks tensors to remember

                if rep==0:
                    ts_rep = ts_filtered # use only filtered candidates
                else:
                    ts_rep = ts # if not enough bottlenecks found on first try, use all tensors as candidates

                # get all bottlenecks in the graph
                bottleneck_ts = []
                for t in ts_rep:
                    # find all nodes before current node + current node
                    b = set(ge.get_backward_walk_ops(t.op, inclusive=True,
                                                     within_ops=fwd_ops))
                    # find all nodes after current node
                    f = set(ge.get_forward_walk_ops(t.op, inclusive=False,
                                                    within_ops=fwd_ops))

                    # check that all of the graph runs through this tensor
                    ops_left_out = set(fwd_ops) - b - f - set([t.op])
                    if not ops_left_out:

                        # check that current node separates the graph
                        b_inp = [inp for op in b for inp in op.inputs]
                        f_inp = [inp for op in f for inp in op.inputs]
                        debug_print("Looking at %s with inputs %s and %s",
                                    [t], b_inp, f_inp)
                        debug_print("Intersection is %s", set(b_inp).intersection(f_inp))
                        if not set(b_inp).intersection(f_inp):  # we have a bottleneck!
                            bottleneck_ts.append(t)

                # success? or try again without filtering?
                if len(bottleneck_ts) >= np.sqrt(len(ts_filtered)): # yes, enough bottlenecks found!
                    break

            if not bottleneck_ts:
                raise('unable to find bottleneck tensors! please provide remember nodes manually, or use remember="speed".')

            # sort the bottlenecks
            bottlenecks_sorted_lists = tf_toposort(bottleneck_ts)
            sorted_bottlenecks = [t for ts in bottlenecks_sorted_lists for t in ts]

            # save an approximately optimal number ~ sqrt(N)
            N = len(ts_filtered)
            k = np.minimum(int(np.floor(np.sqrt(N))), len(sorted_bottlenecks)//2)
            remember = sorted_bottlenecks[k:N:k]

        # use Tarjan's algorithm to find articulation points, use those
        # as remember nodes
        # TODO(y): this alg needs to restrict attention to tensors used in
        # bwd pass
        elif remember == 'tarjan':
            graph = util.tf_ops_to_nx_graph(fwd_ops)
            # TODO: sorted_list missing some nodes
            # KeyError: <tf.Operation 'batch_normalization_4/FusedBatchNorm' type=FusedBatchNorm>
            sorted_list = linearize_lib.linearize(modify_graph=False)
            points = util.sort(nx.articulation_points(graph),
                               total_order=sorted_list, dedup=True)

            
            # estimate number of ops that are cached for backward pass
            # by counting tensors in fwd graph that are inputs to backprop
            fwd_ts = ge.filter_ts(fwd_ops, True)
            with util.capture_ops() as bwd_ops:
                gs = tf_gradients(ys, xs, grad_ys, **kwargs)
            bwd_inputs = [t for op in bwd_ops for t in op.inputs]
            fwd_ts_needed = list(set(bwd_inputs).intersection(fwd_ts))

            # can either take sqrt of fwd_ts_needed or sqrt of bottlenecks
            # the latter works better on tensorflow/models/resnet
            #num_to_save = math.ceil(np.sqrt(len(fwd_ts_needed)))
            num_to_save = math.ceil(np.sqrt(len(points)))
            
            remember_ops = util.pick_n_equispaced(points, num_to_save)

            if len(remember_ops) != len(set(remember_ops)):
                debug_print("warning, some points repeated when saving")
                assert False, "TODO(y): add deduping"

            remember = []
            for op in remember_ops:
                assert len(op.outputs) == 1, ("Don't know how to handle this "
                                              "many outputs")
                remember.append(op.outputs[0])
              
        else:
            raise Exception('%s is unsupported input for "remember"'%
                            (remember,))

    # at this point automatic selection happened and remember is list of nodes
    assert isinstance(remember, list)
    
    debug_print("Remember nodes used: %s", remember)
    # better error handling of special cases
    # xs are already handled as remember nodes, so no need to include them
    xs_intersect_remember = set(xs).intersection(set(remember))
    if xs_intersect_remember:
        debug_print("Warning, some inputs nodes are also remember nodes: %s",
              format_ops(xs_intersect_remember))
    ys_intersect_remember = set(ys).intersection(set(remember))
    debug_print("ys: %s, remember: %s, intersect: %s", ys, remember,
                ys_intersect_remember)
    # saving an output node (ys) gives no benefit in memory while creating
    # new edge cases, exclude them
    if ys_intersect_remember:
        debug_print("Warning, some output nodes are also remember nodes: %s",
              format_ops(ys_intersect_remember))
        
    # remove terminal nodes from remember list if present
    
    ########################################
    # MERGE!:
    # tim version had
    #   remember = list(set(remember) - set(ys))
    #   remember_sorted_lists = tf_toposort(remember)
    #
    # but I needed to subtract xs to handle some edge cases in tests
    ########################################
    # my version had
    #    remember = list(set(remember) - set(ys) - set(xs))
    # tim version had
    #    remember = list(set(remember) - set(ys))

    remember = list(set(remember) - set(ys) - set(xs))
    remember_sorted_lists = tf_toposort(remember)
    
    # check that we have some nodes to remember
    ########################################
    # MERGE:
    # tim version had
    # but this fails with TypeError: exceptions must derive from BaseException
    ########################################
    if not remember:
        raise Exception('no remember nodes found or given as input! ')

    # disconnect dependencies between remembered tensors
    ########################################
    # MERGE!:
    # My version commented out this line, not sure why
    ########################################
    remember_disconnected = {x: tf.stop_gradient(x) for x in remember}
    for x in remember:
        if x.op and x.op.name is not None:
            grad_node = tf.stop_gradient(x, name=x.op.name+"_sg")
        else:
            grad_node = tf.stop_gradient(x)
        remember_disconnected[x] = grad_node

    # partial derivatives to the remembered tensors and xs
    ops_to_copy = my_backward_ops(within_ops=fwd_ops,
                                  seed_ops=[y.op for y in ys],
                                  stop_at_ts=remember)
    debug_print("Found %s ops to copy within ys %s, seed %s, stop_at %s",
                    len(ops_to_copy), fwd_ops, [r.op for r in ys],
                    remember)
    debug_print("Processing list %s", ys)
    copied_sgv, info = ge.copy_with_input_replacements(ge.sgv(ops_to_copy), {})
    copied_ops = info._transformed_ops.values()
    debug_print("Copied %s to %s", ops_to_copy, copied_ops)
    my_reroute_ts(remember_disconnected.values(), remember_disconnected.keys(), can_modify=copied_ops)
    debug_print("Rewired %s in place of %s restricted to %s",
                remember_disconnected.values(), remember_disconnected.keys(), copied_ops)
    for op in copied_ops:
        ge.add_control_inputs(op, [y.op for y in ys])
        #        debug_print("Run %s after %s"%(op.name, [y.op.name for y in ys]))


    ########################################
    # MERGE:
    # Tim's version has
    #    dv = tf.gradients([info._transformed_ops[y.op]._outputs[0] for y in ys], list(remember_disconnected.values())+xs, grad_ys, **kwargs)
    # d_remember = {r: dr for r,dr in zip(remember_disconnected.keys(), dv[:len(remember_disconnected)])}
    ########################################

    # get gradients with respect to current boundary + original x's
    copied_ys = [info._transformed_ops[y.op]._outputs[0] for y in ys]
    boundary = list(remember_disconnected.values())
    dv = tf_gradients(copied_ys, boundary+xs, grad_ys, **kwargs)
    debug_print("Got gradients %s", dv)
    debug_print("for %s", copied_ys)
    debug_print("with respect to %s", boundary+xs)


    # partial derivatives to the remembered nodes
    # dictionary of "node: backprop" for nodes in the boundary
    d_remember = {r: dr for r,dr in zip(remember_disconnected.keys(),
                                        dv[:len(remember_disconnected)])}
    # partial derivatives to xs (usually the params of the neural net)
    d_xs = dv[len(remember_disconnected):]

    ########################################
    # MERGE:
    # Tim's version had
    #
    #
    # get ops to copy for memory saving
    #    ops_to_copy = {}
    #    for ts in remember_sorted_lists[::-1]:
    #        remember_other = [r for r in remember if r not in ts]
    #        ops_to_copy[ts] = my_backward_ops(within_ops=fwd_ops, seed_ops=[r.op for r in ts], stop_at_ts=remember_other)
    ########################################

    # incorporate derivatives flowing through the remembered nodes
    remember_sorted_lists = tf_toposort(remember)
    for ts in remember_sorted_lists[::-1]:
        ########################################
        # MERGE:
        # Tim's version had
        #        if not ops_to_copy[ts]: # we're done!
        #            break
        debug_print("Processing list %s", ts)
        remember_other = [r for r in remember if r not in ts]
        remember_disconnected_other = [remember_disconnected[r] for r in remember_other]

        # copy part of the graph below current remember node, stopping at other remember nodes
        ops_to_copy = my_backward_ops(within_ops=fwd_ops, seed_ops=[r.op for r in ts], stop_at_ts=remember_other)
        debug_print("Found %s ops to copy within %s, seed %s, stop_at %s",
                    len(ops_to_copy), fwd_ops, [r.op for r in ts],
                    remember_other)
        if not ops_to_copy: # we're done!
            break
        copied_sgv, info = ge.copy_with_input_replacements(ge.sgv(ops_to_copy), {})
        copied_ops = info._transformed_ops.values()
        debug_print("Copied %s to %s", ops_to_copy, copied_ops)
        my_reroute_ts(remember_disconnected_other, remember_other, can_modify=copied_ops)
        debug_print("Rewired %s in place of %s restricted to %s",
                    remember_disconnected_other, remember_other, copied_ops)
        for op in copied_ops:
            ge.add_control_inputs(op, [d_remember[r].op for r in ts])
            #            debug_print("Run %s after %s", op, [d_remember[r].op for r in ts])

        # gradient flowing through the remembered node
        boundary = [info._transformed_ops[r.op]._outputs[0] for r in ts]
        substitute_backprops = [d_remember[r] for r in ts]
        dv = tf_gradients(boundary,
                          remember_disconnected_other+xs,
                          grad_ys=substitute_backprops, **kwargs)
            #        import pdb; pdb.set_trace()
        debug_print("Got gradients %s", dv[:len(remember_other)])
        debug_print("for %s", boundary)
        debug_print("with respect to %s", remember_disconnected_other+xs)
        debug_print("with boundary backprop substitutions %s", substitute_backprops)

        # partial derivatives to the remembered nodes
        for r, dr in zip(remember_other, dv[:len(remember_other)]):
            if dr is not None:
                if d_remember[r] is None:
                    d_remember[r] = dr
                else:
                    d_remember[r] += dr

        # partial derivatives to xs (usually the params of the neural net)
        d_xs_new = dv[len(remember_other):]
        for j in range(len(xs)):
            if d_xs_new[j] is not None:
                if d_xs[j] is None:
                    d_xs[j] = d_xs_new[j]
                else:
                    d_xs[j] += d_xs_new[j]

    return d_xs

########################################
# MERGE:
# TODO: use tim's version
# Tim's version turned tensors into ints to make things faster
########################################


def tf_toposort(ts):
    all_ops = ge.get_forward_walk_ops([x.op for x in ts])
    deps = {}
    for op in all_ops:
        for o in op.outputs:
            deps[o] = set(op.inputs)
    sorted_ts = toposort(deps)

    # only keep the tensors from our original list
    ts_sorted_lists = []
    for l in sorted_ts:
        keep = list(set(l).intersection(ts))
        if keep:
            ts_sorted_lists.append(keep)

    return ts_sorted_lists

def my_backward_ops(within_ops, seed_ops, stop_at_ts):
    bwd_ops = set(ge.get_backward_walk_ops(seed_ops, stop_at_ts=stop_at_ts))
    ops = bwd_ops.intersection(within_ops).difference([t.op for t in stop_at_ts])
    return list(ops)

########################################
# MERGE
# functions from util copied here
########################################

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


# TODO(y): add support for empty s
DEBUG_LOGGING=False
def debug_print(s, *args):
  """Like logger.log, but also replaces all TensorFlow ops/tensors with their
  names. Sensitive to value of DEBUG_LOGGING, see enable_debug/disable_debug

  Usage:
    debug_print("see tensors %s for %s", tensorlist, [1,2,3])
  """

  if DEBUG_LOGGING:
    formatted_args = [format_ops(arg) for arg in args]
    print("DEBUG "+s % tuple(formatted_args))

def format_ops(ops, sort_outputs=True):
  """Helper method for printing ops. Converts Tensor/Operation op to op.name,
  rest to str(op)."""
    
  if hasattr(ops, '__iter__') and not isinstance(ops, str):
    l = [(op.name if hasattr(op, "name") else str(op)) for op in ops]
    if sort_outputs:
      return sorted(l)
    return l
  else:
    return ops.name if hasattr(ops, "name") else str(ops)


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

