"""CPU checks for the new training/deployment identity; no GPU/model loading."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dynamicwam.models.onpolicy_feedback_contract import runtime_contract, validate_runtime_contract


class Schedule:
    def to_config(self):
        return {'action_steps': 10, 'video_steps': 2}


class ContractTests(unittest.TestCase):
    def test_even_age_support_and_modes(self):
        for mode in ('static', 'temporal'):
            contract = runtime_contract(Schedule(), .1, mode)
            for h in (2, 4):
                self.assertEqual(validate_runtime_contract(contract, Schedule(), .1, h), mode)
            self.assertEqual(contract['ages'], [2, 4, 6, 8, 10, 12, 14])
            with self.assertRaises(ValueError):
                validate_runtime_contract(contract, Schedule(), .1, 1)

    def test_no_silent_input_or_supervision_change(self):
        for key, value in [('state_semantics', 'measured_qpos'), ('target', 'expert_blend'),
                           ('history', 'raw_frame_lag3'), ('history_dropout', .15),
                           ('residual_limit', .2), ('execute_horizons', [1, 2, 4])]:
            contract = copy.deepcopy(runtime_contract(Schedule(), .1))
            contract[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_runtime_contract(contract, Schedule(), .1, 2)

    def test_invalid_parameters(self):
        for limit in (0, -1, float('nan'), float('inf')):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                runtime_contract(Schedule(), limit)
        with self.assertRaises(ValueError):
            runtime_contract(Schedule(), .1, 'automatic')


if __name__ == '__main__':
    unittest.main()
