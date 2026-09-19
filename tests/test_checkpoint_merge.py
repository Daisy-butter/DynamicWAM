"""Stage-1 identity may ignore motion SHA when flow RGB is unchanged."""

from __future__ import annotations

import unittest

from dynamicwam.training.checkpoint_merge import stage1_video_dataset_identity


def _identity(**overrides: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "format": "dynamicwam_absolute_motion_dataset",
        "version": 2,
        "dataset_fingerprint": "a" * 64,
        "action_stats_sha256": "b" * 64,
        "motion_statistics_sha256": "c" * 64,
    }
    payload.update(overrides)
    return payload


class Stage1VideoIdentityTests(unittest.TestCase):
    def test_motion_sha_and_fingerprint_are_not_required(self) -> None:
        left = stage1_video_dataset_identity(_identity())
        right = stage1_video_dataset_identity(
            _identity(
                dataset_fingerprint="d" * 64,
                motion_statistics_sha256="e" * 64,
            )
        )
        self.assertEqual(left, right)
        self.assertEqual(
            left,
            {
                "format": "dynamicwam_absolute_motion_dataset",
                "version": 2,
                "action_stats_sha256": "b" * 64,
            },
        )

    def test_action_stats_mismatch_is_detected(self) -> None:
        left = stage1_video_dataset_identity(_identity())
        right = stage1_video_dataset_identity(
            _identity(action_stats_sha256="f" * 64)
        )
        self.assertNotEqual(left, right)


if __name__ == "__main__":
    unittest.main()
