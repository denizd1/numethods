# numethods

A small, from-scratch, object-oriented Python package implementing classic numerical methods:

- Linear system solvers: LU, Gauss–Jordan, Jacobi, Gauss–Seidel, Cholesky (SPD)
- Root-finding: Bisection, Fixed-Point Iteration, Secant, Newton's method (for roots)
- Interpolation: Newton divided differences, Lagrange form

## Install (editable)

```bash
pip install -e /numethods
```

or just add `/numethods` to `PYTHONPATH`.

## Examples

```bash
python /numethods/examples/demo.py
```

## Notes

- All algorithms are implemented without relying on external linear algebra solvers.
- Uses plain Python floats and list-of-lists for matrices/vectors.
- Tolerances use a relative criterion `|Δ| ≤ tol (1 + |value|)`.
