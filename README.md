# MPNN
This repository was created to test Message Passing Neural Networks in Tensorflow using the Spektral library.

Main Requirements:
Numpy
Tensorflow
Spektral

We use:
- Optax for the training optimizer.
- Jraph for the graph neural network.
- Haiku for the fully connected neural networks (used to compute edge/message updates and for the readout function).
- Flax for the training loop (it keeps track of the training state and works well with Optax).
