"""CPU-only semantic receipt gate checks; does not import Torch or start CUDA."""
import ast
import copy
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
tree = ast.parse((HERE / 'accept_joint_deployment_graph.py').read_text())
selected = [node for node in tree.body if (
    isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
        and target.id == 'PREVIOUS_SOURCE_CASES' for target in node.targets)
) or (isinstance(node, ast.FunctionDef) and node.name == 'verify_previous_source_semantics')]


def require(condition, message):
    if not condition:
        raise ValueError(message)


scope = {'require': require}
exec(compile(ast.Module(body=selected, type_ignores=[]), 'receipt_gate', 'exec'), scope)
verify = scope['verify_previous_source_semantics']


def fixture():
    checks = []
    for name in sorted(scope['PREVIOUS_SOURCE_CASES']):
        rows = [max(0, 16-int(name.rsplit('_', 1)[1]))] if '_age_' in name else [15]
        checks.append(dict(case=name, valid_rows=rows, plan_innovation_passed=True,
            proposal_reference_passed=True, source_and_queue_delta_equal=True))
    return dict(previous_plan_semantics=dict(mode='training_aligned_source_early_v1',
        passed=True, completed=False, explicit_serving_helper=True, consumer_checks=checks,
        cuda_graph_values_and_masks_checked=True, hard_prefix_unchanged=True),
        cases=[dict(name=x['case']) for x in checks],
        concurrent_slow=dict(previous_plan_helper_on_old_and_new_cache=True))


class SemanticsAcceptanceTests(unittest.TestCase):
    def test_complete_evidence(self):
        verify(fixture())

    def test_reject_completed_plan_substitute(self):
        raw = fixture()
        raw['previous_plan_semantics']['completed'] = True
        with self.assertRaises(ValueError):
            verify(raw)

    def test_reject_old_synthetic_previous_proposal_only(self):
        raw = fixture()
        raw['previous_plan_semantics']['consumer_checks'] = []
        with self.assertRaises(ValueError):
            verify(raw)

    def test_reject_expired_tail_fill(self):
        raw = fixture()
        item = next(x for x in raw['previous_plan_semantics']['consumer_checks']
            if x['case'] == 'training_previous_source_age_16')
        item['valid_rows'] = [12]
        with self.assertRaises(ValueError):
            verify(raw)

    def test_reject_unverified_second_consumer(self):
        raw = fixture()
        raw['previous_plan_semantics']['consumer_checks'][0]['proposal_reference_passed'] = False
        with self.assertRaises(ValueError):
            verify(raw)

    def test_reject_no_actual_graph_case(self):
        raw = fixture()
        raw['cases'].pop()
        with self.assertRaises(ValueError):
            verify(raw)

    def test_reject_background_bypass(self):
        raw = fixture()
        raw['concurrent_slow'] = {}
        with self.assertRaises(ValueError):
            verify(raw)

    def test_checker_syntax_and_helper_in_both_paths(self):
        source = (HERE / 'check_motionkv_graph40k.py').read_text()
        module = ast.parse(source)
        paths = {node.name: ast.unparse(node) for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef) and node.name in ('eager', 'replay')}
        self.assertEqual(set(paths), {'eager', 'replay'})
        for code in paths.values():
            self.assertIn('obs = aligned(obs, cache)', code)
        self.assertIn('graph.current.previous_plan_valid', paths['replay'])
        self.assertIn('refreshed_actual = replay(current, ready.value[0], prefix)', source)


if __name__ == '__main__':
    unittest.main()
