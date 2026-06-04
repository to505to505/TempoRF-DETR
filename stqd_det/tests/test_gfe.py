"""Tests for the Global Feature Enhancement block."""

import torch

from stqd_det.gfe import GFE


def test_forward_preserves_shape():
    gfe = GFE(channels=32, heads=4, dc_kernels=2)
    x = torch.randn(2, 5, 32, 8, 8)
    y = gfe(x)
    assert y.shape == x.shape


def test_forward_t1_edge_case_runs():
    """T=1 sequence: window degenerates but must not crash."""
    gfe = GFE(channels=16, heads=4, dc_kernels=2)
    x = torch.randn(2, 1, 16, 4, 4)
    y = gfe(x)
    assert y.shape == x.shape


def test_window_uses_replicate_padding_at_edges():
    """For T=3, frame 0's previous neighbour and frame 2's next neighbour
    must repeat themselves (Eq. 5 edge clauses)."""
    feat = torch.tensor(
        [[[10.0], [20.0], [30.0]]]                # (B=1, T=3, C=1)
    )
    win = GFE._build_window(feat)
    # win shape (1, 3, 3, 1)  ->  dim2: [prev, curr, next]
    assert win.shape == (1, 3, 3, 1)
    # frame 0: prev replicated == curr == 10
    assert float(win[0, 0, 0, 0]) == 10.0
    assert float(win[0, 0, 1, 0]) == 10.0
    assert float(win[0, 0, 2, 0]) == 20.0
    # frame 1: prev=10, curr=20, next=30
    assert float(win[0, 1, 0, 0]) == 10.0
    assert float(win[0, 1, 1, 0]) == 20.0
    assert float(win[0, 1, 2, 0]) == 30.0
    # frame 2: prev=20, curr=30, next replicated == curr == 30
    assert float(win[0, 2, 0, 0]) == 20.0
    assert float(win[0, 2, 1, 0]) == 30.0
    assert float(win[0, 2, 2, 0]) == 30.0


def test_gradients_flow():
    torch.manual_seed(0)
    gfe = GFE(channels=8, heads=2, dc_kernels=2)
    x = torch.randn(1, 3, 8, 4, 4, requires_grad=True)
    loss = gfe(x).sum()
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    # Every named parameter that is trainable should receive non-zero grad.
    for name, p in gfe.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_output_changes_when_neighbour_frames_change():
    """The MHA over the 3-frame window must couple the centre frame to its
    neighbours, so changing neighbour content should change the centre's
    output."""
    torch.manual_seed(0)
    gfe = GFE(channels=8, heads=2, dc_kernels=2).eval()
    x_a = torch.randn(1, 3, 8, 4, 4)
    x_b = x_a.clone()
    x_b[:, 0] = torch.randn_like(x_b[:, 0])               # swap frame 0 only
    with torch.no_grad():
        y_a = gfe(x_a)
        y_b = gfe(x_b)
    # frame 1's output should differ between the two passes (it attends to
    # frame 0 via the window).
    assert not torch.allclose(y_a[:, 1], y_b[:, 1], atol=1e-4)
