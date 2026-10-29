import gc
import importlib.util
from pathlib import Path
import threading
import unittest


path = Path(__file__).parents[1] / 'src/dynamicwam/dualrate/loop_diagnostics.py'
spec = importlib.util.spec_from_file_location('tested_loop_diagnostics', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
LoopDiagnostics = module.LoopDiagnostics


class Clock:
    wall = 10.
    cpu = 1.
    def advance(self, wall, cpu):
        self.wall += wall
        self.cpu += cpu


class DiagnosticsTests(unittest.TestCase):
    def make(self):
        clock = Clock()
        return LoopDiagnostics(wall_clock=lambda: clock.wall, cpu_clock=lambda: clock.cpu), clock

    def test_wall_stall_distinguished_from_cpu_and_rings_bounded(self):
        diagnostics, clock = self.make()
        for tick in range(200):
            diagnostics.begin_tick(tick, tick*.004, .001)
            with diagnostics.span('physx', physical_time=tick*.004):
                clock.advance(.02, .001)
            diagnostics.finish_tick((tick+1)*.004, .017)
        result = diagnostics.snapshot(include_ticks=True)
        self.assertEqual(result['tick_count'], 200)
        self.assertEqual(len(result['recent_ticks']), 128)
        self.assertEqual(result['recent_ticks'][0]['tick'], 72)
        self.assertEqual(len(result['long_spans']), 64)
        last = result['recent_ticks'][-1]
        self.assertAlmostEqual(last['spans']['physx']['wall_seconds'], .02)
        self.assertAlmostEqual(last['spans']['physx']['thread_cpu_seconds'], .001)
        self.assertEqual(result['phase_totals']['physx']['count'], 200)
        self.assertNotIn('recent_ticks', diagnostics.snapshot())

    def test_exception_keeps_phase_and_partial_tick(self):
        diagnostics, clock = self.make()
        diagnostics.begin_tick(1, .004, .002)
        try:
            with diagnostics.span('control', physical_time=.004):
                clock.advance(.27, .26)
                raise ValueError('native control fault')
        except ValueError:
            diagnostics.finish_tick(.004, .272, complete=False)
        tick = diagnostics.snapshot(include_ticks=True)['recent_ticks'][0]
        self.assertFalse(tick['complete'])
        self.assertAlmostEqual(tick['spans']['control']['wall_seconds'], .27)
        self.assertEqual(tick['physical_before'], tick['physical_after'])

    def test_capture_history_and_video_are_separate(self):
        diagnostics, clock = self.make()
        with diagnostics.span('capture_history', physical_time=4.984):
            clock.advance(.15, .001)
        with diagnostics.span('capture_video', physical_time=4.984):
            clock.advance(.002, .002)
        result = diagnostics.snapshot()
        self.assertAlmostEqual(result['phase_totals']['capture_history']['wall_seconds'], .15)
        self.assertAlmostEqual(result['phase_totals']['capture_video']['wall_seconds'], .002)
        self.assertEqual(result['long_spans'][0]['physical_time'], 4.984)

    def test_gc_owner_only_and_callback_restoration_preserves_others(self):
        diagnostics, clock = self.make()
        original = list(gc.callbacks)
        enabled = gc.isenabled()
        other = lambda phase, info: None
        try:
            diagnostics.install()
            gc.callbacks.append(other)
            def foreign():
                diagnostics._callback('start', {'generation': 2})
                diagnostics._callback('stop', {'generation': 2})
            worker = threading.Thread(target=foreign)
            worker.start(); worker.join()
            self.assertEqual(diagnostics.gc_totals['count'], 0)
            diagnostics._callback('start', {'generation': 2})
            clock.advance(.12, .115)
            diagnostics._callback('stop', {'generation': 2, 'collected': 2, 'uncollectable': 0})
            self.assertEqual(diagnostics.gc_totals['count'], 1)
            self.assertAlmostEqual(diagnostics.gc_totals['wall_seconds'], .12)
            diagnostics.close()
            diagnostics.close()
            self.assertEqual(gc.callbacks, original + [other])
            self.assertEqual(gc.isenabled(), enabled)
        finally:
            diagnostics.close()
            gc.callbacks[:] = original


if __name__ == '__main__':
    unittest.main()
