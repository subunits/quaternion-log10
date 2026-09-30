"""Independent curvature check of the log10-chart metric.

Imports: quaternion_log10.py (exp10) and quaternion_geometry.py (sample_chart_points).

quaternion_geometry.py showed the pulled-back metric matches a formula I derived. This
script does not use that formula. It computes Christoffel symbols, the Riemann tensor and
the Ricci scalar from the metric alone, using nested autograd, and tests the predictions:

  G      = pullback of the flat metric          -> Ricci scalar 0      (it is flat)
  G/10^2s = dt^2 + g_S3  (t = s ln10)           -> Ricci scalar 6, Ricci eigenvalues (0, 2, 2, 2)
  G/c    = ds^2 + (1/ln10)^2 g_S3, c=(10^s ln10)^2 -> Ricci scalar 6 (ln10)^2

The middle row is the metric of R x S^3 with a unit S^3, i.e. the Hopf-manifold metric.
"""
import math
import torch
from torch.func import jacrev

from quaternion_log10 import exp10
from quaternion_geometry import sample_chart_points

LN10 = math.log(10.0)
dt = torch.float64


def G(y):
    J = jacrev(exp10)(y)
    return J.T @ J


def G_cyl(y):                      # G / 10^(2s)
    return G(y) / 10.0 ** (2.0 * y[0])


def G_hat(y):                      # G / (10^s ln10)^2
    return G(y) / (10.0 ** y[0] * LN10) ** 2


def christoffel(gfun, y):
    ginv = torch.linalg.inv(gfun(y))
    dg = jacrev(gfun)(y)                               # dg[i, j, k] = d_k g_ij
    T = torch.zeros(4, 4, 4, dtype=dt)
    for d in range(4):
        for b in range(4):
            for c in range(4):
                T[d, b, c] = dg[d, c, b] + dg[d, b, c] - dg[b, c, d]
    return 0.5 * torch.einsum("ad,dbc->abc", ginv, T)   # Gamma^a_{bc}


def curvature(gfun, y):
    """Returns (Ricci scalar, mixed Ricci eigenvalues) of the metric gfun at point y."""
    Gam = christoffel(gfun, y)
    dGam = jacrev(lambda yy: christoffel(gfun, yy))(y)  # dGam[a, b, c, k] = d_k Gamma^a_{bc}
    R = (torch.einsum("adbc->abcd", dGam) - torch.einsum("acbd->abcd", dGam)
         + torch.einsum("ace,edb->abcd", Gam, Gam) - torch.einsum("ade,ecb->abcd", Gam, Gam))
    ric = torch.einsum("abad->bd", R)                  # R_{bd}
    ginv = torch.linalg.inv(gfun(y))
    mixed = ginv @ ric
    return torch.trace(mixed).item(), torch.linalg.eigvals(mixed).real.sort().values


def report(tag, ok, detail):
    print(f"  [{'ok ' if ok else 'DIFF'}] {tag}: {detail}")


if __name__ == "__main__":
    pts = sample_chart_points(40)[:8]
    print(f"Using {len(pts)} random points inside the principal branch\n")

    rows = [
        ("pulled-back flat metric G", G, 0.0),
        ("G / 10^(2s)  (dt^2 + g_S3)", G_cyl, 6.0),
        ("G / (10^s ln10)^2", G_hat, 6.0 * LN10 ** 2),
    ]
    for name, fn, expected in rows:
        err = max(abs(curvature(fn, y)[0] - expected) for y in pts)
        report(f"Ricci scalar of {name}", err < 1e-6, f"expected {expected:.4f}, max error = {err:.2e}")

    want = torch.tensor([0.0, 2.0, 2.0, 2.0], dtype=dt)
    err = max((curvature(G_cyl, y)[1] - want).abs().max().item() for y in pts)
    report("Ricci eigenvalues of G / 10^(2s)", err < 1e-6, f"expected (0, 2, 2, 2), max error = {err:.2e}")

    print()
    print("A zero eigenvalue plus three equal positive ones is the R x S^3 product structure.")
    print("Scalar curvature 6 with Ricci 2g on the S^3 factor identifies the unit round S^3.")
