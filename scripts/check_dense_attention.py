"""GPU parity check for the synchronization-free, unpadded attention path."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from dynamicwam.models.action_core_ops import dense_action_attention
from dynamicwam.vendor.wan.modules.attention import flash_attention

def main():
    torch.manual_seed(101)
    rows = []
    for batch, length in ((1, 20), (1, 1520), (2, 103)):
        q = torch.randn(batch, 24, 16, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(batch, length, 16, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        lens = torch.full((batch,), length, dtype=torch.int32, device='cuda')
        expected = flash_attention(q, k, v, k_lens=lens, version=2)
        actual = dense_action_attention(q, k, v)
        torch.testing.assert_close(actual, expected, atol=.005, rtol=.005)
        ga = torch.autograd.grad(actual.float().square().mean(), (q, k, v), retain_graph=True)
        ge = torch.autograd.grad(expected.float().square().mean(), (q, k, v))
        for a, e in zip(ga, ge):
            torch.testing.assert_close(a, e, atol=1e-6, rtol=.03)
        rows.append(dict(batch=batch, length=length, max_error=float((actual-expected).abs().max())))
    print(rows)

if __name__ == '__main__':
    main()
