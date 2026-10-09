#!/usr/bin/env python3
"""Exact CPU/CUDA regression against the unmodified vendor Euler scheduler."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import torch
    from dynamicwam.models.flow_schedule import BridgeSchedule, euler_update, scheduler_delta_product
    from dynamicwam.vendor.wan.utils.fm import FlowMatchScheduler
    torch.manual_seed(20261005)
    devices = ["cpu"] + (["cuda"] if args.cuda else [])
    cases = []
    old_device_product_differences = 0
    for device in devices:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--cuda requires the explicitly reserved CUDA GPU")
        for output_dtype in (torch.float32, torch.bfloat16, torch.float16):
            for sample_dtype in set((torch.float32, output_dtype)):
                sample = torch.randn(4096, device=device, dtype=sample_dtype)
                velocity = torch.randn(4096, device=device, dtype=output_dtype)
                pairs = [(1., 5/6), (5/6, 0.), (.2, .7), (.913, .411), (.0007, 0.)]
                native = BridgeSchedule.from_flow_match()
                pairs.extend((float(s), float(n)) for s, n, _ in (native.action_step(i) for i in range(10)))
                for index, (sigma, next_sigma) in enumerate(pairs):
                    vendor = FlowMatchScheduler()
                    vendor.sigmas = torch.tensor([sigma, next_sigma], dtype=torch.float32)
                    vendor.timesteps = torch.tensor([1000., 500.])
                    final = index in (1, 4, 14)
                    if final:
                        vendor.sigmas = vendor.sigmas[:1]
                        vendor.timesteps = vendor.timesteps[:1]
                        next_sigma = 0.
                    expected = vendor.step(velocity, torch.tensor([1000.]), sample)
                    start = vendor.sigmas[0]
                    end = torch.tensor(next_sigma, dtype=torch.float32)
                    actual = euler_update(sample, velocity, start, end)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    delta = (end - start).to(device)
                    explicit = sample + scheduler_delta_product(velocity, delta)
                    torch.testing.assert_close(explicit, expected, atol=0, rtol=0)
                    old = sample + velocity * delta
                    if not torch.equal(old, expected):
                        old_device_product_differences += 1
                    cases.append(dict(device=device, output_dtype=str(output_dtype),
                        sample_dtype=str(sample_dtype), sigma=sigma, next_sigma=next_sigma,
                        max_abs_error=float((actual - expected).abs().max())))
    if args.cuda and old_device_product_differences == 0:
        raise AssertionError("regression failed to exercise the original scalar-rounding bug")
    result = dict(passed=True, cases=len(cases), devices=devices,
                  max_abs_error=max(row["max_abs_error"] for row in cases),
                  original_device_product_mismatches=old_device_product_differences, details=cases)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
