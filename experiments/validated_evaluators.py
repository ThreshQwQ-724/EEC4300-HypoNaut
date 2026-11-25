"""Explicit local evaluator corrections. The pinned upstream tree stays unchanged.

These fixes are evaluation infrastructure, NOT project novelty or a model improvement.
"""
import numpy as np


def corrected_3d_class(mod):
    class SafeStructure3D(mod.Structure3D):
        def normalize(self):
            arrays = [layer.tolist() for layer in self.layers]
            while len(arrays) > 1 and not np.any(arrays[-1]):
                arrays.pop()
            return SafeStructure3D(arrays)

    class CorrectedBenchmark3D(mod.Benchmark3D):
        def parse_llm_response(self, response):
            parsed = super().parse_llm_response(response)
            if parsed is None:
                return None
            return SafeStructure3D([layer.tolist() for layer in parsed.layers])

        def validate_structure_matches_observations(self, structure, observations):
            if not structure.layers or len(structure.normalize().layers) > self.metadata['max_height']:
                return False
            size = self.metadata['grid_size']
            if any(layer.shape != (size,size) or not np.isin(layer,[0,1]).all() for layer in structure.layers):
                return False
            if not np.any(structure.layers[0]):
                return False
            return super().validate_structure_matches_observations(structure,observations)
    return CorrectedBenchmark3D


def corrected_boolean_class(mod):
    import sympy as sp
    from boolean_dataset import BooleanDiscoveryGame

    class CorrectedBooleanBenchmark(mod.BooleanBenchmarkRefined):
        def __init__(self, path):
            super().__init__(path)
            # Enumerate the PUBLIC grammar, never the instance's answer list.
            # Depth alone cannot bound flattened n-ary AND/OR expressions correctly.
            grammar = BooleanDiscoveryGame.generate_all_expressions(
                self.variables,self.operators,self.max_depth,self.mechanistic_opts)
            self.grammar_keys = {e.mechanistic_key(**self.mechanistic_opts) for e in grammar}

        def _in_space(self, expr_str):
            return (super()._in_space(expr_str) and
                    self.get_expression_mechanistic_key(expr_str) in self.grammar_keys)

        def _get_ast_depth(self, expr):
            # NOR is one allowed primitive, not two successive NOT/OR operators.
            if expr is None or isinstance(expr,(sp.Symbol,sp.logic.boolalg.BooleanTrue,sp.logic.boolalg.BooleanFalse)):
                return 0
            if isinstance(expr,sp.Not) and isinstance(expr.args[0],sp.Or):
                return 1 + max(self._get_ast_depth(arg) for arg in expr.args[0].args)
            return 1 + max((self._get_ast_depth(arg) for arg in expr.args),default=-1)

        def _check_operators_in_ast(self, expr, allowed_ops):
            # Upstream skips NOR's children, which can hide forbidden nested operators.
            if expr is None:
                return False
            if isinstance(expr,sp.Symbol):
                return str(expr) in self.variables
            if isinstance(expr,(sp.logic.boolalg.BooleanTrue,sp.logic.boolalg.BooleanFalse)):
                return False
            if isinstance(expr,sp.Not) and isinstance(expr.args[0],sp.Or):
                return ('NOR' in allowed_ops and
                        all(self._check_operators_in_ast(a,allowed_ops) for a in expr.args[0].args))
            mapping={sp.Not:'NOT',sp.And:'AND',sp.Or:'OR',sp.Xor:'XOR',sp.Implies:'IMPLIES',sp.Equivalent:'EQUIV'}
            return (mapping.get(expr.func) in allowed_ops and
                    all(self._check_operators_in_ast(a,allowed_ops) for a in expr.args))

        def validate_expression(self, expr_str, observations):
            # Never treat a parser failure as the all-zero Boolean function.
            if not self._in_space(expr_str):
                return False,None
            return super().validate_expression(expr_str,observations)
    return CorrectedBooleanBenchmark
