import networkx as nx
import sys
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge

#import memory_saving_gradients as util
import util

def test_resnet_structure():
  """sanity check on TF resnet structure."""
  tf.reset_default_graph()
  nodes = util.make_resnet(3)
  all_ops = ge.get_forward_walk_ops(seed_ops=nodes[0].op)
  desired_graph = {0: [1, 2], 1: [2], 2: [3, 4], 3: [4]}
  actual_graph = util.tf_ops_to_graph(all_ops)
  assert(util.graphs_isomorphic(actual_graph, desired_graph))


def test_articulation_points_resnet():
  """Make sure articulation points are found correctly in resnet."""
  tf.reset_default_graph()
  nodes = util.make_resnet(3)
  all_ops = ge.get_forward_walk_ops(seed_ops=nodes[0].op)
  graph = nx.Graph(util.tf_ops_to_graph(all_ops))
  assert util.set_equal(util.format_ops(nx.articulation_points(graph)),
                        ['a01_add'])
  
  tf.reset_default_graph()
  nodes = util.make_resnet(4)
  all_ops = ge.get_forward_walk_ops(seed_ops=nodes[0].op)
  graph = nx.Graph(util.tf_ops_to_graph(all_ops))
  assert util.set_equal(util.format_ops(nx.articulation_points(graph)),
                        ['a01_add', 'a02_add'])


def test_pick_n_equispaced():
  assert util.pick_n_equispaced(range(6), 4) == [0, 2, 3, 5]
  assert util.pick_n_equispaced(range(6), 6) == [0, 1, 2, 3, 4, 5]

if __name__ == '__main__':
  test_resnet_structure()
  test_articulation_points_resnet()
  test_pick_n_equispaced()
  print("%s tests succeeded"%(sys.argv[0],))
