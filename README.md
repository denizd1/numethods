# numethods

A lightweight, from-scratch, object-oriented Python package implementing classic numerical methods.  
**No NumPy / SciPy solvers used**, algorithms are implemented transparently for learning and research.

## Why this might be useful

- Great for teaching/learning numerical methods step by step.
- Good reference for people writing their own solvers in C/Fortran/Julia.
- Lightweight, no dependencies (plotting helpers optionally use `matplotlib`).
- Consistent object-oriented API (.solve() etc).

## Tutorial Series

This package comes with a set of Jupyter notebooks designed as a structured tutorial series in **numerical methods**, both mathematically rigorous and hands-on with code. See [Tutorials](./tutorials/README.md).

## Features

### Linear system solvers

- **LU decomposition** (with partial pivoting, `det()`, `inverse()`): `LUDecomposition`
- **Gauss-Jordan** elimination: `GaussJordan`
- **Jacobi** iterative method: `Jacobi`
- **Gauss-Seidel** iterative method: `GaussSeidel`
- **Successive over-relaxation**: `SOR`
- **Cholesky** factorization (SPD): `Cholesky`

`LUDecomposition`, `GaussJordan` and `Cholesky` accept `record_steps=True` to keep the
intermediate matrices for `.trace()` (off by default because it costs O(n³) extra work).

### Root-finding

- **Bisection**: `Bisection`
- **Regula falsi** (Illinois variant by default): `RegulaFalsi`
- **Brent's method** (bracketing + inverse quadratic interpolation): `Brent`
- **Fixed-Point Iteration**: `FixedPoint`
- **Secant**: `Secant`
- **Newton's method** (for roots): `NewtonRoot`

### Interpolation

- **Newton** (divided differences, `add_point`): `NewtonInterpolation`
- **Lagrange** polynomials (barycentric evaluation, `add_point`): `LagrangeInterpolation`
- **Cubic splines** (natural or clamped): `CubicSpline`
- **Chebyshev nodes**: `chebyshev_nodes`

### Orthogonalization, QR, and Least Squares

- **Classical Gram–Schmidt**: `QRGramSchmidt`
- **Modified Gram–Schmidt**: `QRModifiedGramSchmidt`
- **Householder QR** (numerically stable): `QRHouseholder`
- **QR-based linear solver** (square systems, least squares for tall ones): `QRSolver`
- **Least Squares** for overdetermined systems (via QR): `LeastSquaresSolver`

### Eigenvalue methods

- **Power Iteration** (dominant eigenvalue/vector): `PowerIteration`
- **Inverse Power Iteration** (optionally shifted, factorizes once): `InversePowerIteration`
- **Rayleigh Quotient Iteration**: `RayleighQuotientIteration`
- **QR algorithm** (Hessenberg reduction + Francis double-shift steps, complex
  eigenvalues supported; `shifted=False` gives the basic unshifted iteration): `QREigenvalues`

### Singular Value Decomposition

- **Thin SVD** via one-sided Jacobi rotations (works on \(A\) directly, never forms \(A^T A\)): `SVD`

### ODE solvers

**Initial value problem solvers** for \( y'(t) = f(t,y), \; y(t_0)=y_0 \).
`y0` may be a float or a list (systems of ODEs); `t_end < t0` integrates backwards.

- **Euler's method** (explicit, first order): `Euler`
- **Heun's method** / Improved Euler (2nd order): `Heun`
- **Runge-Kutta 2** (midpoint, 2nd order): `RK2`
- **Runge-Kutta 4** (classic, 4th order): `RK4`
- **Backward Euler** (implicit, requires Newton iteration): `BackwardEuler`
- **Trapezoidal rule** (implicit, 2nd order): `ODETrapezoidal`
- **Adams-Bashforth** (2- and 3-step explicit): `AdamsBashforth`
- **Adams-Moulton** (1-step = trapezoidal, 2-step = 3rd order, implicit): `AdamsMoulton`
- **Predictor-Corrector** (AB2 predictor + trapezoidal corrector, PECE): `PredictorCorrector`
- **Adaptive Runge–Kutta–Fehlberg 4(5)**: `RK45`
- **Adaptive Dormand–Prince 5(4)** (FSAL): `DormandPrince`

### Quadrature (Numerical Integration)

- **Trapezoidal rule** (composite): `Trapezoidal`
- **Simpson's rule** (composite, even n): `Simpson`
- **Gauss-Legendre quadrature** (any number of points, optionally composite): `GaussLegendre`
- **Adaptive Simpson**: `AdaptiveSimpson`
- **Romberg integration**: `Romberg`

### Numerical Differentiation

- **Forward difference**: `ForwardDiff`
- **Backward difference**: `BackwardDiff`
- **Central difference (2nd order)**: `CentralDiff`
- **Central difference (4th order)**: `CentralDiff4th`
- **Second derivative**: `SecondDerivative`
- **Richardson extrapolation**: `RichardsonExtrap`

When `h` is omitted, each formula uses a step close to its optimum,
`h = eps**p * max(1, |x|)` (p = 1/2, 1/3, 1/5, 1/4 depending on the formula).

### Curve Fitting

- **Polynomial least squares fit** (scaled variable for good conditioning): `PolyFit`
- **Linear regression with custom basis functions**: `LinearFit`
- **Exponential fit** (via log transform): `ExpFit`
- **Nonlinear least squares (Levenberg-Marquardt)**: `NonlinearFit`
- Plot helpers `plot_fit`, `plot_residuals` (accept an existing matplotlib `ax`)

### Matrix & Vector utilities

- Minimal `Matrix` / `Vector` classes
- `@` operator for **matrix multiplication**
- `*` and `/` for **scalar**–matrix/vector operations, unary `-`, `==`
- `.T` for transpose, `A[i, j]` entries, `A[i]` rows
- Forward / backward substitution helpers
- Norms (1, 2, ∞, Frobenius), condition numbers, dot products, row/column access

---

## Install (editable)

```bash
git clone https://github.com/denizd1/numethods.git
cd numethods
pip install -e .            # the package itself (no dependencies)
pip install -e ".[test]"    # + pytest and matplotlib for the test suite
```

## Examples

```bash
python examples/demo.py
```

## Tests

```bash
python -m pytest
```

## Notes

- All algorithms are implemented without relying on external linear algebra solvers.
- Uses plain Python floats and list-of-lists for matrices/vectors.
- Iterative methods use the relative stopping test `|Δ| ≤ tol (1 + |value|)`; singularity
  and rank tests are scale-aware (`n · eps · max|a_ij|`), so they do not depend on units.
  The shifted QR algorithm deflates at machine precision.
- ODE implicit solvers use Newton’s method with a finite-difference Jacobian approximation
  and raise `ConvergenceError` if Newton fails.
- Curve fitting supports polynomial, linear basis, exponential, and general nonlinear regression.
- Visualization requires `matplotlib`.
