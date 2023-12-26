from __future__ import annotations

import typing
from functools import wraps

from graphviz import Digraph

import kittygrad.autograd.engine as engine
from kittygrad.tensor.tensor import Tensor
from kittygrad.utils.classes import DotDict


def obj_name(obj: typing.Any) -> str:
    return str(id(obj))


class CompGraph(Digraph):  # TODO: examples (plus no_grad, Function)
    def __init__(self, *,
                 leaf_color: str = '#7ba763',
                 leaf_fillcolor: str = '#d1e4cf',
                 branch_color: str = '#63473d',
                 branch_fillcolor: str = '#c5b9b7',
                 dried_leaf_color: str = '#cdad56',
                 dried_leaf_fillcolor: str = '#ffefc7',
                 backward_color: str = '#22709a',
                 backward_fillcolor: str = '#a9d8ee',
                 **digraph_kwargs) -> None:

        self.leaf_color = leaf_color
        self.leaf_fillcolor = leaf_fillcolor

        self.branch_color = branch_color
        self.branch_fillcolor = branch_fillcolor

        self.dried_leaf_color = dried_leaf_color
        self.dried_leaf_fillcolor = dried_leaf_fillcolor

        self.backward_color = backward_color
        self.backward_fillcolor = backward_fillcolor

        if 'strict' not in digraph_kwargs:
            digraph_kwargs['strict'] = True

        graph_attr = dict(margin='0.075,0.075',
                          ranksep='0.45',
                          outputorder='edgesfirst')
        graph_attr.update(digraph_kwargs.get('graph_attr', {}))
        digraph_kwargs['graph_attr'] = graph_attr

        node_attr = dict(shape='box',
                         height='0.2',
                         penwidth='1.2',
                         fontsize='12',
                         fontname='Times')
        node_attr.update(digraph_kwargs.get('node_attr', {}))
        digraph_kwargs['node_attr'] = node_attr

        edge_attr = dict(arrowsize='0.45', penwidth='1.4')
        edge_attr.update(digraph_kwargs.get('edge_attr', {}))
        digraph_kwargs['edge_attr'] = edge_attr

        super().__init__(**digraph_kwargs)

        self._level = 0
        self._stash = {}

    def __enter__(self) -> typing.Self:
        if engine.BackwardGraph.post_builder_hooks.comp_graph:
            raise RuntimeError("CompGraph context manager does not support nesting.")
        else:
            engine.BackwardGraph.post_builder_hooks.comp_graph = self._hook
            return self

    def __exit__(self, *_args, **_kwargs) -> None:
        del engine.BackwardGraph.post_builder_hooks.comp_graph
        self._stash.clear()

    def _tensor_node_cfg(self, tensor: Tensor) -> DotDict[str, str]:
        if not tensor.is_leaf:
            color = self.branch_color
            fillcolor = self.branch_fillcolor
        elif tensor.requires_grad:
            color = self.leaf_color
            fillcolor = self.leaf_fillcolor
        else:
            color = self.dried_leaf_color
            fillcolor = self.dried_leaf_fillcolor

        return DotDict(name=obj_name(tensor),
                       label=f'Tensor\n{tensor.shape}',
                       style='filled',
                       color=color,
                       fillcolor=fillcolor)

    def _forward_node_cfg(self, function: typing.Callable, out_cfg: DotDict[str, str]) -> DotDict[str, str]:
        return DotDict(name=f'forward_{self._level}',
                       label=function.__name__[1:],
                       style='rounded,filled',
                       color=out_cfg['color'],
                       fillcolor=out_cfg['fillcolor'])

    def _backward_node_cfg(self, ba: engine.BackwardAccess) -> DotDict[str, str]:
        return DotDict(name=obj_name(ba),
                       label=type(ba).__name__,
                       style='rounded,filled',
                       color=self.backward_color,
                       fillcolor=self.backward_fillcolor)

    def _stash_check(self, ba_name: str) -> None:
        if ba_name not in self._stash:
            raise RuntimeError("Visualization of the computational graph "
                               "must be built starting from the leaves.")

    def _render_forward_pass(self, builder: typing.Callable, builder_args: tuple, builder_kwargs: dict,
                             output_tensor: Tensor,
                             operation_subgraph: Digraph,
                             operands_subgraph: Digraph,
                             ) -> tuple[str, str]:

        ot_cfg = self._tensor_node_cfg(output_tensor)
        ot_name = ot_cfg.name
        self.node(**ot_cfg)

        op_cfg = self._forward_node_cfg(builder, ot_cfg)
        op_name = op_cfg.name
        operation_subgraph.node(**op_cfg)

        self.edge(op_name, ot_name, color=op_cfg.color)

        for arg in (*builder_args, *builder_kwargs.values()):
            if not isinstance(arg, Tensor):
                continue

            it_cfg = self._tensor_node_cfg(arg)
            it_name = it_cfg.name

            if it_name not in self._stash:
                self._stash[it_name] = arg  # prevents garbage collection
                operands_subgraph.node(**it_cfg)

            self.edge(it_name, op_name, color=it_cfg.color)

            if arg.grad_fn is not None:
                locked_fn = arg.grad_fn.next_functions[0] if arg is output_tensor else arg.grad_fn
                locked_fn_name = obj_name(locked_fn)

                self._stash_check(locked_fn_name)
                self.edge(it_name, locked_fn_name, constraint='false', style='invis')

        return ot_name, op_name

    def _render_backward_pass(self, ot_name: str, op_name: str,
                              output_tensor: Tensor,
                              operation_subgraph: Digraph,
                              operands_subgraph: Digraph,
                              ) -> None:
        if not output_tensor.requires_grad:
            return

        fn_cfg = self._backward_node_cfg(output_tensor.grad_fn)
        fn_name = fn_cfg.name
        operation_subgraph.node(**fn_cfg)

        self._stash[fn_name] = None  # makes _stash_check exception possible

        self.edge(ot_name, fn_name, color=self.backward_color, constraint='false', style='dashed')
        self.edge(op_name, fn_name, style='invis', weight='2')

        for next_ba in output_tensor.grad_fn.next_functions:
            if next_ba is None:
                continue

            ba_cfg = self._backward_node_cfg(next_ba)
            ba_name = ba_cfg.name

            if isinstance(next_ba, engine.AccumulateGrad):
                operands_subgraph.node(**ba_cfg)
                self.edge(tail_name=obj_name(next_ba._tensor),
                          head_name=ba_name,
                          color=self.backward_color,
                          dir='back',
                          weight='1.2')
            else:
                self._stash_check(ba_name)

            self.edge(fn_name, ba_name, color=self.backward_color, constraint='false')

    def _hook(self, builder: typing.Callable) -> typing.Callable:
        @wraps(builder)
        def construct(*args, **kwargs) -> Tensor:
            output_tensor = builder(*args, **kwargs)

            operation_subgraph = Digraph(f'operation_{self._level}', graph_attr=dict(rank='same'))
            operands_subgraph = Digraph(f'operands_{self._level}', graph_attr=dict(rank='same'))
            self._level += 1

            ot_name, op_name = self._render_forward_pass(
                builder, args, kwargs,
                output_tensor,
                operation_subgraph,
                operands_subgraph,
            )
            self._render_backward_pass(
                ot_name, op_name,
                output_tensor,
                operation_subgraph,
                operands_subgraph,
            )

            self.subgraph(operation_subgraph)
            self.subgraph(operands_subgraph)

            return output_tensor
        return construct
