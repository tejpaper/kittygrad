This subpackage implements all the necessary components for a forward pass. The structure is as follows:

TODO
- **handler.py** defines forward function decorators that facilitate automatic casting of data types, implement [NumPy broadcast semantics](https://numpy.org/doc/stable/user/basics.broadcasting.html), and play an important role in tensor version control,
- **activation.py** implements forward pass for all available activation functions,
- **ops.py** contains the most common frontend and backend mathematical operations on tensors that involve numerical changes,
- **view.py** contains the most common frontend and backend mathematical operations on tensors, which involve changing their shape rather than the data inside them.