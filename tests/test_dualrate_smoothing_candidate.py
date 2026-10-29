import math
import unittest
from dualrate_smoothing_candidate import smooth_actions


class SmoothTest(unittest.TestCase):
    def test_prefix_gripper_and_convex_bounds(self):
        actions = [[float(i)] * 14 for i in range(5)]
        actions[2][6], actions[3][13] = .2, .8
        result = smooth_actions(actions, [True, True, False, False, False], .75)
        self.assertEqual(result[:2], actions[:2])
        for i in range(2, 5):
            self.assertEqual([result[i][6], result[i][13]], [actions[i][6], actions[i][13]])
            for j in (*range(6), *range(7, 13)):
                self.assertLessEqual(min(result[i - 1][j], actions[i][j]), result[i][j])
                self.assertLessEqual(result[i][j], max(result[i - 1][j], actions[i][j]))
        self.assertEqual(actions[2][0], 2.)

    def test_identity_and_all_committed(self):
        a = [[float(i)] * 14 for i in range(4)]
        self.assertEqual(smooth_actions(a, [True, False, False, False], 1.), a)
        self.assertEqual(smooth_actions(a, [True] * 4, .85), a)

    def test_invalid_masks_values_and_alpha(self):
        a = [[0.] * 14 for _ in range(3)]
        for known in ([False] * 3, [True, False, True], [True], [1, 0, 0]):
            with self.assertRaises(ValueError): smooth_actions(a, known)
        for alpha in (0., 1.1, math.nan):
            with self.assertRaises(ValueError): smooth_actions(a, [True, False, False], alpha)
        a[1][3] = math.nan
        with self.assertRaises(ValueError): smooth_actions(a, [True, False, False])


if __name__ == '__main__': unittest.main()
