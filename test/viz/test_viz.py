from test.conftest import *


def test_comp_graph():
    tensor = kitty.tensor(2., requires_grad=True)

    with pytest.raises(RuntimeError) as msg:
        with kitty.CompGraph(), kitty.CompGraph():
            tensor + 1
    assert str(msg.value) == "CompGraph context manager does not support nesting."

    non_leaf_tensor = tensor * 2
    with pytest.raises(RuntimeError) as msg:
        with kitty.CompGraph():
            non_leaf_tensor + 1
    assert str(msg.value) == ("Visualization of the computational graph "
                              "must be built starting from the leaves.")
