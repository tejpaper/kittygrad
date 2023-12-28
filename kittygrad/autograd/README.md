This subpackage provides the most important difference from NumPy - the implementation of a dynamic computational graph, for finding gradients. The structure is as follows:

- **engine.py** defines the fundamental classes and functions to create a computational graph and its backward traversal,
- **interaction.py** provides advanced functionality to directly interact with the gradient calculation process itself,
- **activation.py** implements backward passes for all functions from **func/activation.py**,
- **ops.py** implements backward passes for all functions from **func/ops.py**,
- **view.py** implements backward passes for all functions from **func/view.py**.