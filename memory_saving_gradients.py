from toposort import toposort
import contextlib
import numpy as np
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
import time

# save original gradients since tf.gradient could be monkey-patched to point to our version
from tensorflow.python.ops import gradients as tf_gradients_lib
tf_gradients = tf_gradients_lib.gradients

from tensorflow.python.ops.control_flow_ops import MaybeCreateControlFlowState
import sys
sys.setrecursionlimit(10000)


# tf.gradients slowness work-around: https://github.com/tensorflow/tensorflow/issues/9901
def _MyPendingCount(graph, to_ops, from_ops, colocate_gradients_with_ops):

    # get between ops, faster for large graphs than original implementation
    between_op_list = ge.get_backward_walk_ops(to_ops, stop_at_ts=[op.outputs[0] for op in from_ops], inclusive=False)
    between_op_list += to_ops + from_ops
    between_op_list = list(set(between_op_list))
    between_ops = [False] * (graph._last_id + 1)
    for op in between_op_list:
        between_ops[op._id] = True

    # 'loop_state' is None if there are no while loops.
    loop_state = MaybeCreateControlFlowState(between_op_list, between_ops, colocate_gradients_with_ops)
    # Initialize pending count for between ops.
    pending_count = [0] * (graph._last_id + 1)
    for op in between_op_list:
        for x in op.inputs:
            if between_ops[x.op._id]:
                pending_count[x.op._id] += 1
    return pending_count, loop_state
from tensorflow.python.ops import gradients_impl
gradients_impl._PendingCount = _MyPendingCount


# getting rid of "WARNING:tensorflow:VARIABLES collection name is deprecated"
# spam inside graph_editor
setattr(tf.GraphKeys, "VARIABLES", "variables")

# refers back to current module if we decide to split helpers out
import sys
util = sys.modules[__name__]   

# specific versions we can use to do process-wide replacement of tf.gradients
def gradients_speed(ys, xs, grad_ys=None, **kwargs):
    return gradients(ys, xs, grad_ys, remember='speed', **kwargs)

def gradients_memory(ys, xs, grad_ys=None, **kwargs):
    return gradients(ys, xs, grad_ys, remember='memory', **kwargs)
        
def gradients_collection(ys, xs, grad_ys=None, **kwargs):
    return gradients(ys, xs, grad_ys, remember='collection', **kwargs)
        
def gradients(ys, xs, grad_ys=None, remember='collection', **kwargs):
    '''
    Authors: Tim Salimans & Yaroslav Bulatov

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
    '''
    print("Calling memsaving gradients with", remember)
    if not isinstance(ys,list):
        ys = [ys]
    if not isinstance(xs,list):
        xs = [xs]

    bwd_ops = ge.get_backward_walk_ops([y.op for y in ys],
                                       inclusive=True)
    fwd_ops = ge.get_forward_walk_ops([x.op for x in xs],
                                      inclusive=True,
                                      within_ops=bwd_ops)

    # remove all placeholders and assigns
    fwd_ops = [op for op in fwd_ops if op._inputs]
    fwd_ops = [op for op in fwd_ops if not '/assign' in op.name]
    fwd_ops = [op for op in fwd_ops if not '/Assign' in op.name]
    fwd_ops = [op for op in fwd_ops if not '/read' in op.name]

    # get the tensors, remove variables and very small tensors
    ts_all = ge.filter_ts(fwd_ops, True)
    ts_all = [t for t in ts_all if '/read' not in t.name]
    ts_all = [t for t in ts_all if 'L2Loss' not in t.name]
    ts_all = [t for t in ts_all if 'entropy' not in t.name]
    nr_elem = lambda t: np.prod([s if s>0 else 64 for s in t.shape])
    ts_all = [t for t in ts_all if nr_elem(t)>1024]
    ts_all = set(ts_all) - set(xs)

    debug_print("Filtering tensors: %s", ts_all)

    # construct list of tensors to remember from forward pass, if not given as input
    if type(remember) is not list:
        if remember == 'collection':
            remember = tf.get_collection('remember')
            remember = list(set(remember).intersection(ts_all))
            
        elif remember == 'speed':
            # remember all expensive ops to maximize running speed
            remember = ge.filter_ts_from_regex(fwd_ops, 'conv2d|Conv|MatMul')
            
        elif remember == 'memory':

            # filter out all tensors that are inputs of the backward graph
            with util.capture_ops() as bwd_ops:
                tf_gradients(ys, xs, grad_ys, **kwargs)

            bwd_inputs = [t for op in bwd_ops for t in op.inputs]
            # list of tensors in forward graph that is in input to bwd graph
            ts_filtered = list(set(bwd_inputs).intersection(ts_all))
            debug_print("Using tensors %s", ts_filtered)

            for ts in [ts_filtered, ts_all]:  # try two slightly different ways of getting bottlenecks tensors to remember

                # get all bottlenecks in the graph
                bottleneck_ts = []
                for t in ts:
                    b = set(ge.get_backward_walk_ops(t.op, inclusive=False, within_ops=fwd_ops))
                    f = set(ge.get_forward_walk_ops(t.op, inclusive=False, within_ops=fwd_ops))

                    # check that there are not shortcuts
                    b_inp = set([inp for op in b for inp in op.inputs]).intersection(ts_all)
                    f_inp = set([inp for op in f for inp in op.inputs]).intersection(ts_all)
                    if not set(b_inp).intersection(f_inp) and len(b_inp)+len(f_inp) >= len(ts_all)-1:
                        bottleneck_ts.append(t)  # we have a bottleneck!

                # success? or try again without filtering?
                if len(bottleneck_ts) >= np.sqrt(len(ts_filtered)): # yes, enough bottlenecks found!
                    break

            if not bottleneck_ts:
                raise Exception('unable to find bottleneck tensors! please provide remember nodes manually, or use remember="speed".')

            # sort the bottlenecks
            bottlenecks_sorted_lists = tf_toposort(bottleneck_ts, within_ops=fwd_ops)
            sorted_bottlenecks = [t for ts in bottlenecks_sorted_lists for t in ts]

            # save an approximately optimal number ~ sqrt(N)
            N = len(ts_filtered)
            if len(bottleneck_ts) < np.sqrt(N):
                remember = sorted_bottlenecks
            else:
                step = int(np.ceil(len(bottleneck_ts) / np.sqrt(N)))
                remember = sorted_bottlenecks[step::step]
            
        else:
            raise Exception('%s is unsupported input for "remember"' % (remember,))

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

    # remove initial and terminal nodes from remember list if present
    remember = list(set(remember) - set(ys) - set(xs))
    
    # check that we have some nodes to remember
    if not remember:
        raise Exception('no remember nodes found or given as input! ')

    # disconnect dependencies between remembered tensors
    remember_disconnected = {}
    for x in remember:
        if x.op and x.op.name is not None:
            grad_node = tf.stop_gradient(x, name=x.op.name+"_sg")
        else:
            grad_node = tf.stop_gradient(x)
        remember_disconnected[x] = grad_node

    # partial derivatives to the remembered tensors and xs
    ops_to_copy = fast_backward_ops(seed_ops=[y.op for y in ys], stop_at_ts=remember, within_ops=fwd_ops)
    debug_print("Found %s ops to copy within ys %s, seed %s, stop_at %s",
                    len(ops_to_copy), fwd_ops, [r.op for r in ys], remember)
    debug_print("Processing list %s", ys)
    copied_sgv, info = ge.copy_with_input_replacements(ge.sgv(ops_to_copy), {})
    copied_ops = info._transformed_ops.values()
    debug_print("Copied %s to %s", ops_to_copy, copied_ops)
    ge.reroute_ts(remember_disconnected.values(), remember_disconnected.keys(), can_modify=copied_ops)
    debug_print("Rewired %s in place of %s restricted to %s",
                remember_disconnected.values(), remember_disconnected.keys(), copied_ops)

    # get gradients with respect to current boundary + original x's
    copied_ys = [info._transformed_ops[y.op]._outputs[0] for y in ys]
    boundary = list(remember_disconnected.values())
    dv = tf_gradients(copied_ys, boundary+xs, grad_ys, **kwargs)
    debug_print("Got gradients %s", dv)
    debug_print("for %s", copied_ys)
    debug_print("with respect to %s", boundary+xs)

    inputs_to_do_before = [y.op for y in ys]
    if grad_ys is not None:
        inputs_to_do_before += grad_ys
    wait_to_do_ops = list(copied_ops) + [g.op for g in dv if g is not None]
    my_add_control_inputs(wait_to_do_ops, inputs_to_do_before)

    # partial derivatives to the remembered nodes
    # dictionary of "node: backprop" for nodes in the boundary
    d_remember = {r: dr for r,dr in zip(remember_disconnected.keys(),
                                        dv[:len(remember_disconnected)])}
    # partial derivatives to xs (usually the params of the neural net)
    d_xs = dv[len(remember_disconnected):]

    # incorporate derivatives flowing through the remembered nodes
    remember_sorted_lists = tf_toposort(remember, within_ops=fwd_ops)
    for ts in remember_sorted_lists[::-1]:
        debug_print("Processing list %s", ts)
        remember_other = [r for r in remember if r not in ts]
        remember_disconnected_other = [remember_disconnected[r] for r in remember_other]

        # copy part of the graph below current remember node, stopping at other remember nodes
        ops_to_copy = fast_backward_ops(within_ops=fwd_ops, seed_ops=[r.op for r in ts], stop_at_ts=remember_other)
        debug_print("Found %s ops to copy within %s, seed %s, stop_at %s",
                    len(ops_to_copy), fwd_ops, [r.op for r in ts],
                    remember_other)
        if not ops_to_copy: # we're done!
            break
        copied_sgv, info = ge.copy_with_input_replacements(ge.sgv(ops_to_copy), {})
        copied_ops = info._transformed_ops.values()
        debug_print("Copied %s to %s", ops_to_copy, copied_ops)
        ge.reroute_ts(remember_disconnected_other, remember_other, can_modify=copied_ops)
        debug_print("Rewired %s in place of %s restricted to %s",
                    remember_disconnected_other, remember_other, copied_ops)

        # gradient flowing through the remembered node
        boundary = [info._transformed_ops[r.op]._outputs[0] for r in ts]
        substitute_backprops = [d_remember[r] for r in ts]
        dv = tf_gradients(boundary,
                          remember_disconnected_other+xs,
                          grad_ys=substitute_backprops, **kwargs)
        debug_print("Got gradients %s", dv[:len(remember_other)])
        debug_print("for %s", boundary)
        debug_print("with respect to %s", remember_disconnected_other+xs)
        debug_print("with boundary backprop substitutions %s", substitute_backprops)

        inputs_to_do_before = [d_remember[r].op for r in ts]
        wait_to_do_ops = list(copied_ops) + [g.op for g in dv if g is not None]
        my_add_control_inputs(wait_to_do_ops, inputs_to_do_before)

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


def tf_toposort(ts, within_ops=None):
    all_ops = ge.get_forward_walk_ops([x.op for x in ts], within_ops=within_ops)

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


def fast_backward_ops(within_ops, seed_ops, stop_at_ts):
    bwd_ops = set(ge.get_backward_walk_ops(seed_ops, stop_at_ts=stop_at_ts))
    ops = bwd_ops.intersection(within_ops).difference([t.op for t in stop_at_ts])
    return list(ops)


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

def my_add_control_inputs(wait_to_do_ops, inputs_to_do_before):
    for op in wait_to_do_ops:
        ci = [i for i in inputs_to_do_before if op.control_inputs is None or i not in op.control_inputs]
        ge.add_control_inputs(op, ci)

