import numpy as np
import tensorflow as tf
import unittest

from layer_MPNN import *
from utils_MPNN import *

class TestHelperFunctions(unittest.TestCase):
  def setUp(self):
    self.batch_size = 2
    self.num_nodes = 4
    self.num_node_features = 5
    self.num_edge_features = 4
    self.n_hidden_m = 32
    self.node_factor = 2
    self.edge_factor = 2
    self.testMPEU = MPEU(activation="linear", 
                        do_edge_update=True,
                        n_hidden_m=self.n_hidden_m,
                        kernel_initializer=tf.keras.initializers.Identity())
    self.h_test = tf.ones([self.batch_size, self.num_nodes, self.num_node_features], 
                          dtype=tf.dtypes.float32)*self.node_factor
    self.e_test = tf.ones([self.batch_size, self.num_nodes, self.num_nodes, self.num_edge_features], 
                          dtype=tf.dtypes.float32)*self.edge_factor
    self.a_test = cutoff_adj_batch(self.e_test[:,:,:,0], cutoff=2)
    self.testMPEU([self.h_test, self.a_test, self.e_test])

  def test_dist_matrix_batch(self):
    '''Test dist_matrix_batch, returns correct node distances for batch size 1.

    Create a 3d position matrix with a node at [1, 0, 0], another at [0, 1, 0],
    we expect the distance from one node to the other will be square root of 2
    sqrt((1-0)^2 + (0-1)^2)= sqrt(2). We expect the returned euclidian distance
    matrix to be (1, 2, 2) (batch_size, num_nodes, num_nodes).
    '''
    position_matrix = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float64)
    expected_euclidian_distance = np.array([[[0.0, np.sqrt(2)], [np.sqrt(2), 0.0]]], dtype=np.float64)
    euclidian_distance = dist_matrix_batch(position_matrix)
    #print(euclidian_distance)
    #print(expected_euclidian_distance)
    #print(np.shape(expected_euclidian_distance))
    #print(np.shape(euclidian_distance))

    self.assertEqual(np.shape(expected_euclidian_distance), np.shape(euclidian_distance))
    self.assertTrue(np.array_equal(expected_euclidian_distance, euclidian_distance.numpy()))


    #self.assertTrue(False)
  def test_dist_matrix_batch_random(self):
    '''Test dist_matrix_batch with random positions matrix'''
    batch_size = 32
    num_nodes = 10
    dimensions = 3
    position_matrix = tf.random.uniform(shape = [batch_size, num_nodes, dimensions],
                                        maxval = 1.0)
    expected_euclidian_distance = np.zeros(shape = [batch_size, num_nodes, num_nodes])
    # calculate differences manually for each entry of distance matrix
    for i in range(batch_size):
      for j in range(num_nodes):
        for k in range(num_nodes):
          difference_vector = position_matrix[i, j, :] - position_matrix[i, k, :]
          expected_euclidian_distance[i, j, k] = np.linalg.norm(difference_vector)
    euclidian_distance = dist_matrix_batch(position_matrix)
    euclidian_distance_error = np.abs(euclidian_distance - expected_euclidian_distance)
    diff_error = np.mean(euclidian_distance_error)
    print(diff_error)

    self.assertEqual(np.shape(expected_euclidian_distance), np.shape(euclidian_distance))
    self.assertTrue(diff_error < 1e-7) # expected floating point error

  def test_threshold_cutoff(self):
    batch_size = 1
    num_nodes = 3
    cutoff = 2.0
    adjacency_matrix = tf.constant([[[-1,1.9,2],[3,0,1],[2.1,4,1]]], dtype=tf.dtypes.float64)
    print(adjacency_matrix)
    threshold_adjacency_matrix = threshold_cutoff(adjacency_matrix, cutoff)
    print(threshold_adjacency_matrix)

  def test_MPEU_message(self):
    """Test MPEU message, with weight matrices initialized as identity and linear activation.
    Input to the layer are node and edge features with all ones, and every node
    is connected to every other node and itself, which is done by setting all edge features as ones.

    The output message is expected to be a tensorflow tensor with shape
    [batch_size, num_nodes, n_hidden_m]. A block strating at (0,0,0) has message values
    edge_factor*node_factor*num_nodes, with size [batch_size, num_nodes, smallest_feature_length],
    where smallest_feature length is min([num_node_features, num_edge_features]),
    as when multiplying elementwise, values outside of smallest_feature_length
    will be set to zero. All elements outside of the inner block are zero.
    """
    smallest_feature_length = min([self.num_node_features, self.num_edge_features])

    #print(testMPEU.w1)
    message_MPEU = self.testMPEU.message(self.h_test, self.a_test, self.e_test)
    #print(message_MPEU)
    message_factor = self.edge_factor*self.node_factor*self.num_nodes # this is the expected value of each message
    message_expected = tf.ones([self.batch_size, self.num_nodes, smallest_feature_length],
                               dtype=tf.dtypes.float32)*message_factor
    paddings = tf.constant([[0,0],[0,0], [0,self.n_hidden_m-smallest_feature_length]])
    message_expected = tf.pad(message_expected, paddings)
    #print(message_expected)
    message_diff = np.abs(message_MPEU - message_expected)
    diff_error = np.mean(message_diff)
    print(diff_error)
    self.assertEqual(np.shape(message_MPEU), np.shape(message_expected))
    self.assertTrue(diff_error < 1e-7) # expected floating point error

  def test_MPEU_node_update(self):
    message_MPEU = self.testMPEU.message(self.h_test, self.a_test, self.e_test)
    message_truncated = message_MPEU[:,:,:self.num_node_features]

    h_next_MPEU = self.testMPEU.node_update(self.h_test, self.a_test, self.e_test)
    #print(np.shape(message_truncated))
    h_next_expected = self.h_test + message_truncated

    h_next_diff = np.abs(h_next_MPEU - h_next_expected)
    diff_error = np.mean(h_next_diff)
    print(diff_error)
    self.assertEqual(np.shape(h_next_MPEU), np.shape(h_next_expected))
    self.assertTrue(diff_error < 1e-7) # expected floating point error

  def test_edge_update(self):
    #print(self.testMPEU.n_hidden_E1)
    #print(self.testMPEU.wE1)
    #print(self.testMPEU.wE2)
    edge_next_MPEU = self.testMPEU.edge_update(self.h_test, self.a_test, self.e_test)
    edge_next_expected = np.zeros([self.batch_size, self.num_nodes,
                                   self.num_nodes, self.num_edge_features])
    for b_i in range(self.batch_size):
      for i in range(self.num_nodes):
        for j in range(self.num_nodes):
          features_concat = tf.concat([self.h_test[b_i, i, :], self.h_test[b_i, j, :], self.e_test[b_i, i, j, :]],
                                      axis = 0)
          edge_next_expected[b_i, i, j] = features_concat[:self.num_edge_features]
    #print(edge_next_expected)
    e_next_diff = np.abs(edge_next_MPEU - edge_next_expected)
    diff_error = np.mean(e_next_diff)
    print(diff_error)
    self.assertEqual(np.shape(edge_next_MPEU), np.shape(edge_next_expected))
    self.assertTrue(diff_error < 1e-7) # expected floating point error

  def test_embedding_layer(self):
    out_dim_e = 10
    testEmbedding = MPEU_embedding_QM9(out_dim_e=out_dim_e)
    h_embed, a_embed, e_embed = testEmbedding([self.h_test, self.a_test, self.e_test])
  
  def test_edge_gradient(self):
  '''Test gradient of wE1 and wE2 in edge update function with tensorflow gradient tape
  '''
    return 0

if __name__ == '__main__':
    unittest.main()
