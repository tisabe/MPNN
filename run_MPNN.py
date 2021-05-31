import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam  # Maybe change this later.

from layer_MPNN import *

from spektral.data import BatchLoader  # Feed batches of data.
from spektral.datasets import QM9

# Download the dataset.
dataset = QM9(amount=1000)  # Set amount=None to train on whole dataset
print(dataset[0])

# Define the dimensions of our dataset.
num_node_features = dataset.n_node_features  # Dimension of node features
num_edge_features = dataset.n_edge_features  # Dimension of edge features
n_out = dataset.n_labels  # Dimension of the target

# Train/test split
# Shuffling the indices of the dataset and holding shuffled indices in a np.array.
# Change idxs name. ids_shuffled.
#np.random.seed(0)
#tf.random.set_seed(0)

ids_shuffled = np.random.permutation(len(dataset))
split = int(0.9 * len(dataset))  # Number of training data points.
ids_train, ids_test = np.split(ids_shuffled, [split])  # Split our data into test/train.
dataset_train, dataset_test = dataset[ids_train], dataset[ids_test]

### create the model with MPNN layers
edge_updates = False # disable edge updates

class MyGNN(Model):
  def __init__(self):
    super().__init__()
    self.mp_layer1 = MPEU(activation="relu", do_edge_update=edge_updates)
    self.mp_layer2 = MPEU(activation="relu", do_edge_update=edge_updates)
    self.mp_layer3 = MPEU(activation="relu", do_edge_update=edge_updates)
    self.embedding_layer = MPEU_embedding_QM9()
    self.readout_layer = MPEU_readout(out_dim=19)
  
  def call(self, inputs):
    res = self.embedding_layer(inputs)
    res = self.mp_layer1(res)
    res = self.mp_layer2(res)
    res = self.mp_layer3(res)
    return self.readout_layer(res)

# define hyperparameters
learning_rate = 1e-3  # Learning rate
epochs = 20  # Number of training epochs
batch_size = 32  # Batch size

model = MyGNN()
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss="mse", run_eagerly=True)
#model.compile(optimizer=optimizer, loss="mse")
loader_tr = BatchLoader(dataset_train, batch_size=batch_size)

model.fit(loader_tr.load(),
          steps_per_epoch=loader_tr.steps_per_epoch,
          epochs=epochs)

print("Learning rate:")
print(learning_rate)

