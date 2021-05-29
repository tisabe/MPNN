import numpy as np
import tensorflow as tf

def dist_matrix_batch(position_matrix):
  """Return the batched pairwise distance matrix of positions in euclidian space.
  
  See this link:
  https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow

  We multiply the position matrices together. And then reduce the sum by summing
  over the distance to the origin origin axis of the squared position matrix.

  Args:
    position_matrix: Position matrix given as a tensor. 
      Size: (batch size, number of positions, dimension of positions)
  
  Returns:
    Euclidian distances between nodes as a tf.Tensor.
      Size: (batch size, number of positions, number of positions)
  """

  row_norm_squared = tf.math.reduce_sum(position_matrix*position_matrix, -1)
  # Turn r into column vector
  row_norm_squared = tf.reshape(row_norm_squared, [len(position_matrix), -1, 1])
  # 2*pos*potT
  distance_matrix = 2*tf.linalg.matmul(position_matrix, position_matrix, transpose_b=True)
  # Stick this equation into our overleaf. 
  distance_matrix = row_norm_squared + tf.transpose(row_norm_squared, perm=[0,2,1]) - distance_matrix
  distance_matrix = tf.abs(distance_matrix) # to avoid negative numbers before sqrt
  return tf.sqrt(distance_matrix)

def cutoff_adj_batch(distance_matrix, cutoff):
  """Return the batched adjacency matrix from distance_matrix with cutoff, diagonal set to zero.

  All values in distance_matrix lower than cutoff are set to 1, all values higher
  are set to 0. Then the main diagonal is set to 0.

  Args:
    distance_matrix: batched pairwise distance matrix.
      Size: (batch size, number of positions, number of positions)
    
    cutoff: distance cutoff, scalar.
  
  Returns:
    batched adjacency matrix with cutoff.
      Size: (batch size, number of positions, number of positions)
  """
  adj = tf.less_equal(distance_matrix, cutoff)
  adj = tf.cast(adj, tf.int32)
  return tf.linalg.set_diag(adj, tf.zeros([adj.shape[0],adj.shape[1]], dtype=tf.int32))

def threshold_cutoff(x, cutoff):
  """Return the tensor x, with all values higher than cutoff and diagonal set to zero.

  Args: 
    x: tensor, any size
    cutoff: cutoff distance, scalar

  Returns:
    tensor, same size as x
  """
  x = tf.cast(x, tf.dtypes.float64)
  cond = tf.less(x, tf.ones(tf.shape(x), dtype=tf.dtypes.float64)*cutoff)
  out = tf.where(cond, x, tf.zeros(tf.shape(x), dtype=tf.dtypes.float64))
  out = tf.linalg.set_diag(out, tf.zeros([out.shape[0],out.shape[1]], dtype=tf.float64))
  return out






