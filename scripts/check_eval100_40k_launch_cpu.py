"""CPU-only launch/observer regression checks; no SSH, subprocess, or GPU."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('observer40k',
    HERE / 'observe_eval100_40k_initial_once.py')
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


class InitialAcceptanceTests(unittest.TestCase):
    def test_missing_files_are_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(observer.read_optional(Path(directory) / 'missing.json'), {})
        self.assertEqual(observer.classify({}, {}, {}, []), 'waiting')

    def test_formal_transition_before_guard_status(self):
        self.assertEqual(observer.classify({}, {'state': 'formal_running'}, {}, []), 'waiting')

    def test_both_terminal_spellings(self):
        for state in ('complete', 'completed'):
            self.assertEqual(observer.classify({}, {'state': state}, {}, []), 'terminal')

    def test_failure_precedes_acceptance(self):
        self.assertEqual(observer.classify({'state': 'failed'}, {'state': 'complete'}, {}, []), 'failed')
        self.assertEqual(observer.classify({}, {'state': 'formal_running'},
            {'state': 'failed'}, []), 'failed')

    def test_all_three_acceptance_requirements(self):
        pipeline = {'state': 'formal_running'}
        guard = dict(state='running', startup_health_window_closed=True, runtime_seconds=300)
        self.assertEqual(observer.classify({}, pipeline, guard, list(range(8))), 'ready')
        self.assertEqual(observer.classify({}, pipeline, guard, list(range(7))), 'waiting')
        self.assertEqual(observer.classify({}, pipeline, dict(guard, runtime_seconds=299),
            list(range(8))), 'waiting')
        self.assertEqual(observer.classify({}, pipeline, dict(guard, startup_health_window_closed=False),
            list(range(8))), 'waiting')

    def test_extra_saved_reports_do_not_block_readiness(self):
        self.assertEqual(observer.classify({}, {'state': 'formal_running'},
            dict(state='running', startup_health_window_closed=True, runtime_seconds=400),
            list(range(9))), 'ready')

    def test_launcher_has_no_observer_launch(self):
        source = (HERE / 'start_eval100_40k.py').read_text()
        self.assertNotIn('observe_eval100_40k_initial_once.py', source)
        self.assertEqual(source.count('subprocess.Popen('), 1)
        self.assertNotIn('8b577ef90edb35ee475c1bbe6308dd25f917bd86', source)

    def test_syntax_and_full_gate_chain(self):
        for name in ('start_eval100_40k.py', 'run_eval100_40k.py',
                     'observe_eval100_40k_initial_once.py', 'accept_eval100_40k_initial.py'):
            ast.parse((HERE / name).read_text(), filename=name)
        source = (HERE / 'run_eval100_40k.py').read_text()
        self.assertLess(source.index("run('preflight'"), source.index("run('graph'"))
        self.assertLess(source.index("run('graph_acceptance'"), source.index("run('pipeline'"))
        self.assertIn("'outputs/joint40k_35x20_01_prevsource_scaled025'", source)
        self.assertIn("'checkpoint40k/multirate_step_40000.pt'", source)


if __name__ == '__main__':
    unittest.main()
