# Quaternion log chart: 10^q, log10(q), and the Hopf structure

Three small PyTorch scripts that extend `10^x` and `log10(x)` to the quaternions and check
what the log chart does geometrically. A decade, the step `s -> s + 1` in `s = log10|q|`,
turns out to be a scaling of flat quaternion space by 10, and the structure it induces on
the quotient is the locally conformally hyperkahler structure of the Hopf manifold
`S^1 x S^3`. Each claim below is tested numerically.

## Files

| File | What it does |
|---|---|
| `quaternion_log10.py` | `exp10(q)`, `log10(q)` and `qmul` for quaternions, with six identity checks |
| `quaternion_geometry.py` | Pulls the flat metric back through the log chart; checks the conformal cylinder form, the factor of 100 per decade, and the hypercomplex structure |
| `quaternion_lck.py` | Checks `d(Omega) = theta ^ Omega` for the rescaled Kahler forms of I, J, K, and the period of `theta` |

Import chain: `quaternion_lck.py` imports `quaternion_geometry.py`, which imports
`quaternion_log10.py`. Keep all three in one folder under these exact names.

## Setup

```
pip install torch
pip install numpy    # optional; silences a harmless "Failed to initialize NumPy" warning
```

Tested on Python 3.12 and 3.14. CPU only.

## Running

```
python quaternion_log10.py
python quaternion_geometry.py
python quaternion_lck.py
```

Each finishes in seconds. Every check prints `[ok ]` when it passes. Two lines are
intentionally different: `[wrap]` and `[differ]` in `quaternion_log10.py` mark places where
an identity is expected to fail, and `[varies]` in `quaternion_geometry.py` marks a
quantity that is expected to be non-constant.

## Conventions

Quaternions are tensors of shape `(..., 4)` ordered `(w, x, y, z)`. With `q = w + v`:

```
10^q     = 10^w * [cos(|v| ln10) + v_hat * sin(|v| ln10)]
log10 q  = [ln|q| + v_hat * atan2(|v|, w)] / ln10
```

The log chart is `y = log10(q) = (s, w')` with `s = log10|q|`.

## What has been shown

- **Inverses:** `10^(log10 q) = q` for all `q != 0`. `log10(10^q) = q` only when `|v| ln10 < pi`; outside that range the principal logarithm wraps.
- **Addition:** `10^(p+q) = 10^p * 10^q` only when `p` and `q` commute, meaning their vector parts are parallel.
- **Logarithms of -1:** there is a whole sphere of them, one per unit axis. Three axes are checked.
- **Gradients:** both functions are differentiable through PyTorch autograd.
- **Metric:** in log coordinates the pulled-back flat metric is `(10^s ln10)^2` times a cylinder-type form, and one decade scales it by exactly 100.
- **Hypercomplex structure:** left multiplication by i, j, k gives I, J, K with `I^2 = J^2 = K^2 = -1` and `IJ = K`. All three are orthogonal for the flat metric.
- **The chart is not hyperholomorphic:** the pulled-back I squares to -1 but varies from point to point.
- **Locally conformally hyperkahler:** with `Omega = 10^(-2s) * omega`, the identity `d(Omega) = theta ^ Omega` holds for I, J and K with the same `theta = -2 ln10 ds`, while `d(Omega)` itself is nonzero. `Omega` is invariant under the decade shift, so it descends to `S^1 x S^3`.
- **Period:** `theta` integrates to `-2 ln10` around one decade loop, so it is closed but not exact on the quotient.

All agreement is at floating-point level (errors near 1e-15 in float64), at random points
inside the principal branch.

## What has not been shown

- That `S^1 x S^3` admits no global Kahler metric (`b_2 = 0`). This is a topological fact taken from the literature, not computed here.
- Anything at points outside the principal branch (`|v| ln10 >= pi`), where the chart is not defined.
- A proof. These are numerical checks at sampled points, not derivations.

## Possible next steps

- Check Weyl-closedness of the conformal class in `quaternion_lck.py`.
- Test the structure near the branch boundary, where the chart degenerates.
