"""Locally conformally (hyper)Kahler check in the log10 chart.

Imports: quaternion_log10.py (exp10) and quaternion_geometry.py (left_mult, sampler).

Setup
  F(y) = exp10(y) maps log coordinates y = (s, w) to H.
  On flat H, omega_A(X, Y) = g0(A X, Y) for A in {I, J, K} is closed.
  In the chart: omega_A(y) = J^T Om0 J with J = dF/dy and Om0 = A^T.

Claims tested (Hopf rescaling, |q| = 10^s)
  1. d omega_A = 0 for the pulled-back flat forms.
  2. The rescaled form Omega_A = 10^(-2s) omega_A is NOT closed.
  3. d Omega_A = theta ^ Omega_A with theta = -2 ln10 ds, the same closed theta for I, J, K.
  4. Omega_A is invariant under the decade shift s -> s + 1, so it descends to S^1 x S^3.
  5. The period of theta around the decade loop is -2 ln10 != 0, so theta is closed but not exact.
"""
import math
import torch
from torch.func import jacrev

from quaternion_log10 import exp10
from quaternion_geometry import left_mult, sample_chart_points

LN10 = math.log(10.0)
dt = torch.float64
E = torch.eye(4, dtype=dt)
STRUCTS = {"I": left_mult(E[1]), "J": left_mult(E[2]), "K": left_mult(E[3])}


def make_forms(A):
    Om0 = A.T                                        # omega(X, Y) = (A X)^T Y = X^T A^T Y

    def omega(y):
        J = jacrev(exp10)(y)
        return J.T @ Om0 @ J

    def Omega(y):
        return 10.0 ** (-2.0 * y[0]) * omega(y)

    return omega, Omega


def exterior_d(form_fn, y):
    """(d form)_{abc} = d_a f_{bc} + d_b f_{ca} + d_c f_{ab}; also returns max |d_a f_bc|."""
    dW = jacrev(form_fn)(y)                          # dW[b, c, a] = d_a f_{bc}
    d = torch.zeros(4, 4, 4, dtype=dt)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                d[a, b, c] = dW[b, c, a] + dW[c, a, b] + dW[a, b, c]
    return d, dW.abs().max().item()


def wedge_theta(theta, form):
    """(theta ^ f)_{abc} = theta_a f_{bc} + theta_b f_{ca} + theta_c f_{ab}."""
    out = torch.zeros(4, 4, 4, dtype=dt)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                out[a, b, c] = theta[a] * form[b, c] + theta[b] * form[c, a] + theta[c] * form[a, b]
    return out


def report(tag, ok, detail):
    print(f"  [{'ok ' if ok else 'DIFF'}] {tag}: {detail}")


if __name__ == "__main__":
    pts = sample_chart_points(40)[:12]
    theta = torch.zeros(4, dtype=dt)
    theta[0] = -2.0 * LN10                           # theta = -2 ln10 ds
    shift = torch.tensor([1.0, 0, 0, 0], dtype=dt)
    print(f"Using {len(pts)} random points inside the principal branch\n")

    for name, A in STRUCTS.items():
        omega, Omega = make_forms(A)
        worst = {"closed": 0.0, "notclosed": 1e9, "lee": 0.0, "inv": 0.0}
        for y in pts:
            d_om, scale_om = exterior_d(omega, y)
            d_Om, scale_Om = exterior_d(Omega, y)
            rhs = wedge_theta(theta, Omega(y))
            worst["closed"] = max(worst["closed"], d_om.abs().max().item() / scale_om)
            worst["notclosed"] = min(worst["notclosed"], d_Om.abs().max().item() / scale_Om)
            worst["lee"] = max(worst["lee"], (d_Om - rhs).abs().max().item() / scale_Om)
            worst["inv"] = max(worst["inv"], (Omega(y + shift) - Omega(y)).abs().max().item() /
                               Omega(y).abs().max().item())

        print(f"Structure {name}")
        report("1. d omega = 0 (pulled-back flat form)", worst["closed"] < 1e-8,
               f"max relative |d omega| = {worst['closed']:.2e}")
        report("2. d Omega != 0 (rescaled form not closed)", worst["notclosed"] > 1e-3,
               f"min relative |d Omega| = {worst['notclosed']:.2e}")
        report("3. d Omega = theta ^ Omega", worst["lee"] < 1e-8,
               f"max relative error = {worst['lee']:.2e}")
        report("4. Omega invariant under s -> s + 1", worst["inv"] < 1e-9,
               f"max relative error = {worst['inv']:.2e}")
        print()

    print("5. Period of theta around one decade loop (s: 0 -> 1, w fixed)")
    n = 2000
    s = torch.linspace(0.0, 1.0, n + 1, dtype=dt)
    period = (theta[0] * (s[1:] - s[:-1])).sum().item()
    print(f"  integral of theta = {period:.6f}  (= -2 ln10 = {-2 * LN10:.6f})")
    print("  theta is closed (constant coefficients) but has nonzero period, so it is not")
    print("  the differential of any function on the quotient S^1 x S^3.")
    print()
    print("Not checked here: that S^1 x S^3 admits no global Kahler metric (b_2 = 0).")
    print("That is a topological fact, taken from the literature.")
