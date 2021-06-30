import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam  # Maybe change this later.

from spektral.data import BatchLoader  # Feed batches of data.
from spektral.datasets import QM9
from spektral.layers.convolutional.conv import Conv

from utils_MPNN import *

class MPEU(Conv):
  """A message passing with edge updates layer from the paper

  [Neural Message Passing with Edge Updates for Predicting Properties
  of Molecules and Materials, Jorgensen et al, https://arxiv.org/abs/1806.03146]

  Extends the Conv layer from Spektral library.

  Mode: MPEU assumes the layer is run in batch-mode
  Node and edge feature matrices are zero padded to the highest number of nodes,
  N_max, within the batch.

  Input: tuple (h, a, e)
    - h: node feature matrix, tensor
        size: (batch_size, N_max, num_node_features)
    - a: adjacency matrix, tensor
        size: (batch_size, N_max, N_max)
    - e: edge feature matrix, tensor
        size: (batch_size, N_max, N_max, num_edge_features)

  Output: tuple (h_out, a_out, e_out)
    - h_out: same is input, but after convolution
    - a_out: same as input
    - e_out: same as input, but after convolution

  
  """
  def __init__(
      self,
      do_edge_update: bool = False,
      n_hidden_m: int = 32, n_hidden_w2=32, n_hidden_w5=32, n_hidden_E1=None,
      activation=None,
      use_bias=True,
      kernel_initializer="glorot_uniform",
      bias_initializer="zeros",
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      **kwargs
  ):
    """Constructor for MPEU class.
    
    Args:
      do_edge_update: boolean, wether to compute edge updates (for testing)
      n_m: number of hidden neurons
      n_w2: number of output neurons of the w2 weight matrix
      n_w5: number of output neurons of the w5 weight matrix
      n_E1: number of output neurons of the E2 weight matrix
      activation: Activation function.
      use_bias: Bool, add a bias vector to the output
      kernel_initializer: initializer for the weights;
      bias_initializer: initializer for the bias vector;
      kernel_regularizer: regularization applied to the weights;
      bias_regularizer: regularization applied to the bias vector;
      activity_regularizer: regularization applied to the output;
      kernel_constraint: constraint applied to the weights;
      bias_constraint: constraint applied to the bias vector.
    """
    super().__init__(
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs
    )
    self.n_hidden_m = n_hidden_m
    self.n_hidden_w2 = n_hidden_w2
    self.n_hidden_w5 = n_hidden_w5
    self.n_hidden_E1 = n_hidden_E1
    self.do_edge_update = do_edge_update

  def build(self, input_shape):
    """Build the layer on first call, when matrix dimensions are known."""
    self.batch_size = input_shape[0][0]

    # TODO: change these names.

    self.N = input_shape[0][-2] # number of nodes (maximum in batch, smaller graphs are zero padded)
    self.F = input_shape[0][-1] # number of node features
    #self.E = input_shape[2][-2] # number of edges
    self.S = input_shape[2][-1] # number of edge features
    self.n_hidden_E1 = (2*self.F + self.S) if self.n_hidden_E1==None else self.n_hidden_E1
    # add weight matrices for feed-forward neural networks
    self._w1 = self.add_weight(
        shape=(self.F, self.n_hidden_m),
        initializer=self.kernel_initializer,
        name="w1",
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self._w2 = self.add_weight(
        shape=(self.S, self.n_hidden_w2),
        initializer=self.kernel_initializer,
        name="w2",
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self._w3 = self.add_weight(
        shape=(self.n_hidden_w2, self.n_hidden_m),
        initializer=self.kernel_initializer,
        name="w2",
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self._w4 = self.add_weight(
        shape=(self.n_hidden_m, self.n_hidden_w5),
        initializer=self.kernel_initializer,
        name="w4",
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self._w5 = self.add_weight(
        shape=(self.n_hidden_w5, self.F),
        initializer=self.kernel_initializer,
        name="w5",
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)

    if self.do_edge_update:
      self._wE1 = self.add_weight(
          shape=(2*self.F + self.S, self.n_hidden_E1),
          initializer=self.kernel_initializer,
          name="wE1",
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint)
      self._wE2 = self.add_weight(
          shape=(self.n_hidden_E1, self.S),
          initializer=self.kernel_initializer,
          name="wE2",
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint)

  def message(self, x, a, e):
    """Return node-wise message as described in paper above."""
    x_w = tf.matmul(x, self._w1)
    x_w = tf.expand_dims(x_w, -3)
    # tf.tile repeats a tensor in a given dimension.
    x_w = tf.tile(x_w, [1,x_w.shape[-2],1,1]) # maybe tiling dimension needs to be changed
    e_w = tf.matmul(e, self._w2)
    e_w = self.activation(e_w)
    e_w = tf.matmul(e_w, self._w3)
    e_w = self.activation(e_w)
    message = tf.multiply(x_w, e_w) # edge-wise message
    m = tf.reduce_sum(message, axis=-2) # reduced, node-wise message
    return m

  def node_update(self, x, a, e):
    """Return updated nodes as in paper above."""
    m = self.message(x, a, e)
    h_next = tf.matmul(m, self._w4)
    h_next = self.activation(h_next)
    h_next = tf.matmul(h_next, self._w5)
    h_next = tf.add(h_next, x)
    return h_next

  def edge_update(self, x, a, e):
    """Return updated edges as in paper above."""
    h_w = tf.expand_dims(x, -3)
    h_w = tf.tile(h_w, [1,h_w.shape[-2],1,1])
    h_v = tf.expand_dims(x, -2)
    h_v = tf.tile(h_v, [1,1,h_v.shape[-3],1])
    e_next = tf.concat([h_v, h_w, e], axis=-1)
    e_next = tf.matmul(e_next, self._wE1)
    e_next = self.activation(e_next)
    e_next = tf.matmul(e_next, self._wE2)
    e_next = self.activation(e_next)
    return e_next

  def call(self, inputs):
    """Return the updated values of node, edge features and adjacency matrix.
    This function is called by calling the layer object after instantiation.
    """
    h, a, e = inputs
    h = self.node_update(h, a, e)
    if self.do_edge_update:
      e = self.edge_update(h, a, e)
    return h, a, e

class MPEU_readout(Conv):
  """A readout layer from the paper
  
  [Neural Message Passing with Edge Updates for Predicting Properties 
  of Molecules and Materials, Jorgensen et al, https://arxiv.org/abs/1806.03146]

  Extends the Conv layer from Spektral library.

  Mode: batch
  Node and edge feature matrices are zero padded to the highest number of nodes,
  N_max, within the batch.

  Input: tuple (h, a, e)
    - h: node feature matrix, tensor
        size: (batch_size, N_max, num_node_features)
    - a: adjacency matrix, tensor
        size: (batch_size, N_max, N_max)
    - e: edge feature matrix, tensor
        size: (batch_size, N_max, N_max, num_edge_features)
  
  Output: target feature vector
    size: (batch_size, out_dim)

  Args:
    - out_dim: number of features per graph in the training/testing dataset 
    - n_hidden: number of hidden neurons 
    - activation: activation function;
    - use_bias: bool, add a bias vector to the output;
    - kernel_initializer: initializer for the weights;
    - bias_initializer: initializer for the bias vector;
    - kernel_regularizer: regularization applied to the weights;
    - bias_regularizer: regularization applied to the bias vector;
    - activity_regularizer: regularization applied to the output;
    - kernel_constraint: constraint applied to the weights;
    - bias_constraint: constraint applied to the bias vector.
  """
  def __init__(
      self,
      out_dim,
      n_hidden=16,
      activation="relu",
      use_bias=True,
      kernel_initializer="glorot_uniform",
      bias_initializer="zeros",
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      **kwargs
  ):
    super().__init__(
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs
    )
    self.out_dim = out_dim
    self.n_hidden = n_hidden

  def build(self, input_shape):
    """Build the layer on first call, when matrix dimensions are known."""
    #assert len(input_shape) == 3 # assert is not supported in tf graph mode
    self.batch_size = input_shape[0][0]
    self.N = input_shape[0][-2] # number of nodes (maximum in batch, smaller graphs are zero padded)
    self.F = input_shape[0][-1] # number of node features
    #self.E = input_shape[2][-2] # number of edges
    self.S = input_shape[2][-1] # number of edge features
    self.n_hidden = self.F/2 if self.n_hidden==None else self.n_hidden
    # add weight matrices for feed-forward neural networks
    self._w6 = self.add_weight(
        shape=(self.F, self.n_hidden),
        initializer=self.kernel_initializer,
        name="w1",
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self._w7 = self.add_weight(
        shape=(self.n_hidden, self.out_dim),
        initializer=self.kernel_initializer,
        name="w2",
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)

  def call(self, inputs):
    h, a, e = inputs
    h = tf.matmul(h, self._w6)
    h = self.activation(h)
    h = tf.matmul(h, self._w7)
    y = tf.math.reduce_sum(h, axis=-2)
    # TODO: include a averaging argument, as some target properties require averaging
    return y

# Embedding is not learned. All it's doing is getting adjacency matrix with
# our distances between nodes and a cutoff distance. Expanding the radial
# distances and edges for the edge features.
class MPEU_embedding_QM9(Conv):
  """A embedding layer for Message Passing with Edge Updates from the paper
  
  [Neural Message Passing with Edge Updates for Predicting Properties 
  of Molecules and Materials, Jorgensen et al, https://arxiv.org/abs/1806.03146]

  Extends the Conv layer from Spektral library.

  Mode: batch
  Node and edge feature matrices are zero padded to the highest number of nodes,
  N_max, within the batch.

  Input: tuple (h, a, e)
    - h: node feature matrix, tensor
        size: (batch_size, N_max, num_node_features)
    - a: adjacency matrix, tensor
        size: (batch_size, N_max, N_max)
    - e: edge feature matrix, tensor
        size: (batch_size, N_max, N_max, num_edge_features)
  
  Output: target feature vector
    size: (batch_size, out_dim)

  Args:
    - cutoff: cutoff distance for atoms to be counted as neighbors
    - out_dim_e: number of edge features in the output embedding
    - activation: activation function;
    - use_bias: bool, add a bias vector to the output;
    - kernel_initializer: initializer for the weights;
    - bias_initializer: initializer for the bias vector;
    - kernel_regularizer: regularization applied to the weights;
    - bias_regularizer: regularization applied to the bias vector;
    - activity_regularizer: regularization applied to the output;
    - kernel_constraint: constraint applied to the weights;
    - bias_constraint: constraint applied to the bias vector.
  """
  def __init__(
      self,
      cutoff=20.0,
      out_dim_e=150,
      activation=None,
      use_bias=True,
      kernel_initializer="glorot_uniform",
      bias_initializer="zeros",
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      **kwargs
  ):
    super().__init__(
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs
    )
    self.out_dim_e = out_dim_e
    self.cutoff = cutoff

  def build(self, input_shape):
    """Build the layer on first call, when matrix dimensions are known."""
    #assert len(input_shape) == 3 # assert is not supported in tf graph mode
    self.batch_size = input_shape[0][0]
    self.N = input_shape[0][-2] # number of nodes (maximum in batch, smaller graphs are zero padded)
    self.F = input_shape[0][-1] # number of node features
    #self.E = input_shape[2][-2] # number of edges
    self.S = input_shape[2][-1] # number of edge features

  def call(self, inputs):
    #print("input length:", len(inputs))
    x = inputs[0]
    a = inputs[1]
    a = tf.cast(a, tf.dtypes.float32)
    e = inputs[2]
    #print("x shape:", x.shape)
    pos = x[:,:,5:8]
    D = dist_matrix_batch(pos)
    #a = threshold_cutoff(D, self.cutoff)
    # generate k's for radial basis expansion
    k_rbf = get_k_matrix(D, self.out_dim_e)
    # do radial basis expansion
    D_k = tf.expand_dims(D, -1)
    D_k = tf.tile(D_k, [1, 1, 1, self.out_dim_e])
    delta = 0.1 # TODO: make parameters accessible
    mu_min = 0.0
    #e_k = tf.cast(D_k, tf.dtypes.float64) - delta*tf.cast(k_rbf, tf.dtypes.float64) + mu_min
    e_k = D_k - delta*k_rbf 
    e_k = e_k + mu_min
    e_k = -1.0*tf.square(e_k)/delta
    e_k = tf.exp(e_k)
    # get elements, where distance is below cutoff
    cond = tf.less(D_k, tf.ones(tf.shape(D_k), dtype=tf.dtypes.float32)*self.cutoff)
    # set all other elements in e_k to zero
    e_k = tf.where(cond, e_k, tf.zeros(tf.shape(e_k), dtype=tf.dtypes.float32))

    # extract all node features except positions
    x_m = tf.concat([x[...,:5],x[...,8:]],axis=-1)
    return [x_m, a, e_k]










