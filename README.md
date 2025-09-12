# numethods

A small, from-scratch, object-oriented Python package implementing classic numerical methods.  
**No NumPy / SciPy solvers used**, algorithms are implemented transparently for learning and research.

## Features

### Linear system solvers

- **LU decomposition** (with partial pivoting): `LUDecomposition`
- **Gauss-Jordan** elimination: `GaussJordan`
- **Jacobi** iterative method: `Jacobi`
- **Gauss-Seidel** iterative method: `GaussSeidel`
- **Cholesky** factorization (SPD): `Cholesky`

### Root-finding

- **Bisection**: `Bisection`
- **Fixed-Point Iteration**: `FixedPoint`
- **Secant**: `Secant`
- **Newton's method** (for roots): `NewtonRoot`

### Interpolation

- **Newton** (divided differences): `NewtonInterpolation`
- **Lagrange** polynomials: `LagrangeInterpolation`

### Orthogonalization, QR, and Least Squares

- **Classical Gram–Schmidt**: `QRGramSchmidt`
- **Modified Gram–Schmidt**: `QRModifiedGramSchmidt`
- **Householder QR** (numerically stable): `QRHouseholder`
- **QR-based linear solver** (square systems): `QRSolver`
- **Least Squares** for overdetermined systems (via QR): `LeastSquaresSolver`

### Eigenvalue methods (NEW)

- **Power Iteration** (dominant eigenvalue/vector): `PowerIteration`
- **Inverse Power Iteration** (optionally shifted): `InversePowerIteration`
- **Rayleigh Quotient Iteration**: `RayleighQuotientIteration`
- **QR eigenvalue iteration** (unshifted, educational): `QREigenvalues`

### Singular Value Decomposition (NEW)

- **SVD** via eigen-decomposition of \(A^T A\): `SVD`

### Matrix & Vector utilities

- Minimal `Matrix` / `Vector` classes
- `@` operator for **matrix multiplication**
- `*` for **scalar**–matrix multiplication
- `.T` for transpose
- Forward / backward substitution helpers
- Norms, dot products, row/column access

---

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
