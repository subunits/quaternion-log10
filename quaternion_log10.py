"""10^q and log10(q) for quaternions, with numerical checks.

Quaternions are tensors of shape (..., 4) ordered (w, x, y, z).
  10^q     = 10^w * [cos(|v| ln10) + v_hat * sin(|v| ln10)]
  log10 q  = [ln|q| + v_hat * arccos(w/|q|)] / ln10
"""
import math
import torch

LN10 = math.log(10.0)
EPS = 1e-12


def qmul(p, q):
    pw, px, py, pz = p.unbind(-1)
    qw, qx, qy, qz = q.unbind(-1)
    return torch.stack([
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ], dim=-1)


def _split(q):
    w, v = q[..., :1], q[..., 1:]
    n = v.norm(dim=-1, keepdim=True)
    vhat = v / n.clamp_min(EPS)            # direction is irrelevant when n = 0
    return w, n, vhat


def exp10(q):
    w, n, vhat = _split(q)
    theta = n * LN10
    scale = 10.0 ** w
    return torch.cat([scale * torch.cos(theta), scale * torch.sin(theta) * vhat], dim=-1)


def log10(q):
    w, n, vhat = _split(q)
    mag = torch.sqrt(w ** 2 + n ** 2)
    angle = torch.atan2(n, w)              # arccos(w/|q|), stable near the axis
    return torch.cat([torch.log(mag), angle * vhat], dim=-1) / LN10


def check(name, a, b, tol=1e-9):
    err = (a - b).abs().max().item()
    print(f"  [{'ok ' if err < tol else 'DIFF'}] {name}: max error = {err:.2e}")


if __name__ == "__main__":
    torch.manual_seed(0)
    dt = torch.float64

    print("1. Scalars reduce to ordinary 10^x and log10(x)")
    x = torch.tensor([0.5, 1.0, 2.0, 3.0], dtype=dt)
    q = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)], -1)
    check("exp10 real part", exp10(q)[..., 0], 10.0 ** x)
    check("log10 real part", log10(exp10(q))[..., 0], x)

    print("2. 10^(log10 q) = q for any q != 0")
    q = torch.randn(1000, 4, dtype=dt)
    check("exp10(log10(q)) == q", exp10(log10(q)), q)

    print("3. log10(10^q) = q only when |v| ln10 < pi")
    v = torch.randn(1000, 3, dtype=dt)
    v = v / v.norm(dim=-1, keepdim=True)
    small = torch.cat([torch.randn(1000, 1, dtype=dt), 0.9 * v * (math.pi / LN10) * torch.rand(1000, 1, dtype=dt)], -1)
    big = torch.cat([small[:, :1], v * 1.5 * (math.pi / LN10)], -1)
    check("inside principal branch ", log10(exp10(small)), small)
    err = (log10(exp10(big)) - big).abs().max().item()
    print(f"  [wrap] outside branch: max error = {err:.2e} (expected, log wraps the angle)")

    print("4. 10^(p+q) = 10^p * 10^q only if p and q commute")
    p = torch.randn(4, dtype=dt)
    q_par = torch.cat([torch.randn(1, dtype=dt), p[1:] * 0.7])   # same vector direction
    q_gen = torch.randn(4, dtype=dt)
    check("parallel vector parts  ", exp10(p + q_par), qmul(exp10(p), exp10(q_par)))
    err = (exp10(p + q_gen) - qmul(exp10(p), exp10(q_gen))).abs().max().item()
    print(f"  [differ] generic p, q: max error = {err:.2e} (expected, non-commutative)")

    print("5. Gradients flow through both functions (usable in PyTorch models)")
    q = torch.randn(4, dtype=dt, requires_grad=True)
    loss = (log10(exp10(q * 0.1)) ** 2).sum()
    loss.backward()
    print(f"  grad = {q.grad.tolist()}")

    print("6. Negative reals have many logs: log10(-1) picks one axis, all are valid")
    m1 = torch.tensor([-1.0, 0.0, 0.0, 0.0], dtype=dt)
    for axis in range(1, 4):
        cand = torch.zeros(4, dtype=dt)
        cand[axis] = math.pi / LN10
        check(f"axis {axis}: exp10(pi/ln10 * unit) == -1", exp10(cand), m1)
