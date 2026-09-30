"""Geometry of the log10 chart on the quaternions (imports quaternion_log10.py).

Chart: y = log10(q) = (s, w) with s = log10|q| and w = theta * q_hat / ln10.
Map:   F(y) = exp10(y) = q, so the pulled-back flat metric is G = J^T J, J = dq/dy.

Predicted (exp coordinates on S^3, theta = |w| ln10 in (0, pi)):
  G = c * [ ds^2 + dr^2 + (sin(theta)/theta)^2 * (tangential part) ],  c = (10^s ln10)^2
i.e. conformal to the cylinder R x S^3, with a factor of 100 per decade.
"""
import math
import torch
from torch.autograd.functional import jacobian

from quaternion_log10 import exp10, log10, qmul, check

LN10 = math.log(10.0)
dt = torch.float64
torch.manual_seed(0)


def pullback_metric(y):
    J = jacobian(exp10, y)          # (4, 4): dq/dy
    return J.T @ J, J


def predicted_metric(y):
    s, w = y[0], y[1:]
    r = w.norm()
    theta = r * LN10
    c = (10.0 ** s * LN10) ** 2
    f = (torch.sin(theta) / theta) ** 2
    what = (w / r).unsqueeze(1)
    P = what @ what.T                                     # radial projector
    G = torch.zeros(4, 4, dtype=dt)
    G[0, 0] = 1.0
    G[1:, 1:] = P + f * (torch.eye(3, dtype=dt) - P)
    return c * G


def left_mult(u):
    """4x4 matrix of q -> u * q."""
    basis = torch.eye(4, dtype=dt)
    return torch.stack([qmul(u.expand(4, 4), basis)[k] for k in range(4)], dim=1)


def sample_chart_points(n):
    q = torch.randn(n, 4, dtype=dt)
    y = log10(q)
    keep = (y[:, 1:].norm(dim=-1) * LN10 > 0.2) & (y[:, 1:].norm(dim=-1) * LN10 < 2.9)
    return y[keep]


if __name__ == "__main__":
    pts = sample_chart_points(40)
    print(f"Using {len(pts)} random points inside the principal branch\n")

    print("1. Pulled-back metric equals the predicted conformal-cylinder form")
    err = max((pullback_metric(y)[0] - predicted_metric(y)).abs().max().item() /
              predicted_metric(y).abs().max().item() for y in pts)
    print(f"  [{'ok ' if err < 1e-9 else 'DIFF'}] max relative error = {err:.2e}")

    print("2. One decade (s -> s+1) scales the metric by exactly 100")
    shift = torch.tensor([1.0, 0, 0, 0], dtype=dt)
    err = max((pullback_metric(y + shift)[0] - 100.0 * pullback_metric(y)[0]).abs().max().item() /
              pullback_metric(y + shift)[0].abs().max().item() for y in pts)
    print(f"  [{'ok ' if err < 1e-9 else 'DIFF'}] max relative error = {err:.2e}")

    print("3. Metric divided by 10^(2s) depends only on w (the R x S^3 cylinder structure)")
    errs = []
    for y in pts:
        y2 = y.clone()
        y2[0] += 0.7                                    # move along the s axis
        a = pullback_metric(y)[0] / 10.0 ** (2 * y[0])
        b = pullback_metric(y2)[0] / 10.0 ** (2 * y2[0])
        errs.append((a - b).abs().max().item() / a.abs().max().item())
    print(f"  [{'ok ' if max(errs) < 1e-9 else 'DIFF'}] max relative error = {max(errs):.2e}")

    print("4. Left multiplications I, J, K form a hypercomplex structure on flat H")
    e = torch.eye(4, dtype=dt)
    I, J_, K = (left_mult(e[k]) for k in (1, 2, 3))
    check("I^2 = -1, J^2 = -1, K^2 = -1",
          torch.stack([I @ I, J_ @ J_, K @ K]), -torch.stack([e, e, e]))
    check("IJ = K", I @ J_, K)
    check("I, J, K orthogonal (flat metric preserved)",
          torch.stack([I.T @ I, J_.T @ J_, K.T @ K]), torch.stack([e, e, e]))
    check("omega_I antisymmetric", I, -I.T)

    print("5. The log chart is NOT hyperholomorphic: I pulled back to y-coordinates is not constant")
    y_a, y_b = pts[0], pts[1]
    Ja, Jb = pullback_metric(y_a)[1], pullback_metric(y_b)[1]
    I_a = torch.linalg.solve(Ja, I @ Ja)
    I_b = torch.linalg.solve(Jb, I @ Jb)
    diff = (I_a - I_b).abs().max().item()
    print(f"  pulled-back I squares to -1 at a point: error {(I_a @ I_a + e).abs().max().item():.1e}")
    print(f"  [{'varies' if diff > 1e-6 else 'const '}] max difference between two points = {diff:.3f}")
    print("  (constant in the flat chart, non-constant in the log chart: the chart is a")
    print("   coordinate change, not a hyperkahler map)")
