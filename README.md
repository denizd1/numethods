# numethods

A lightweight, from-scratch, object-oriented Python package implementing classic numerical methods.  
**No NumPy / SciPy solvers used**, algorithms are implemented transparently for learning and research.

## Why this might be useful

- Great for teaching/learning numerical methods step by step.
- Good reference for people writing their own solvers in C/Fortran/Julia.
- Lightweight, no dependencies.
- Consistent object-oriented API (.solve(), .integrate() etc).

---

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

### Eigenvalue methods

- **Power Iteration** (dominant eigenvalue/vector): `PowerIteration`
- **Inverse Power Iteration** (optionally shifted): `InversePowerIteration`
- **Rayleigh Quotient Iteration**: `RayleighQuotientIteration`
- **QR eigenvalue iteration** (unshifted, educational): `QREigenvalues`

### Singular Value Decomposition

- **SVD** via eigen-decomposition of \(A^T A\): `SVD`

### ODE solvers

**Initial value problem solvers** for \( y'(t) = f(t,y), \; y(t_0)=y_0 \)

- **Euler's method** (explicit, first order): `Euler`
- **Heun's method** / Improved Euler (2nd order): `Heun`
- **Runge-Kutta 2** (midpoint, 2nd order): `RK2`
- **Runge-Kutta 4** (classic, 4th order): `RK4`
- **Backward Euler** (implicit, requires Newton iteration): `BackwardEuler`
- **Trapezoidal rule** (implicit, 2nd order): `ODETrapezoidal`
- **Adams-Bashforth** (multistep explicit): `AdamsBashforth`
- **Adams-Moulton** (multistep implicit): `AdamsMoulton`
- **Predictor-Corrector** (AB predictor + AM corrector): `PredictorCorrector`
- **Adaptive Runge–Kutta (RK45)** (Fehlberg/Dormand–Prince, step control): `RK45`

### Quadrature (Numerical Integration)

- **Trapezoidal rule** (composite): `Trapezoidal`
- **Simpson's rule** (composite, even n): `Simpson`
- **Gauss-Legendre quadrature** (2 and 3 point): `GaussLegendre`

### Numerical Differentiation

- **Forward difference**: `ForwardDiff`
- **Backward difference**: `BackwardDiff`
- **Central difference (2nd order)**: `CentralDiff`
- **Central difference (4th order)**: `CentralDiff4th`
- **Second derivative**: `SecondDerivative`
- **Richardson extrapolation**: `RichardsonExtrap`

### Curve Fitting

- **Polynomial least squares fit**: `PolyFit`
- **Linear regression with custom basis functions**: `LinearFit`
- **Exponential fit** (via log transform): `ExpFit`
- **Nonlinear least squares (Gauss–Newton / Levenberg–Marquardt)**: `NonlinearFit`

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
- ODE implicit solvers use Newton’s method with finite-difference Jacobian approximation.
- Curve fitting supports polynomial, linear basis, exponential, and general nonlinear regression.
- Visualization requires `matplotlib`.
