"""numethods: classic numerical methods implemented from scratch in pure Python."""

__version__ = "0.2.0"

from .linalg import Matrix, Vector, forward_substitution, backward_substitution
from .orthogonal import (
    QRGramSchmidt,
    QRModifiedGramSchmidt,
    QRHouseholder,
    QRSolver,
    LeastSquaresSolver,
)
from .solvers import LUDecomposition, GaussJordan, Jacobi, GaussSeidel, SOR, Cholesky
from .roots import (
    Bisection,
    RegulaFalsi,
    Brent,
    FixedPoint,
    Secant,
    NewtonRoot,
    print_trace,
)
from .interpolation import (
    NewtonInterpolation,
    LagrangeInterpolation,
    CubicSpline,
    chebyshev_nodes,
)
from .quadrature import (
    Trapezoidal,
    Simpson,
    GaussLegendre,
    AdaptiveSimpson,
    Romberg,
    gauss_legendre_nodes,
)
from .eigen import (
    PowerIteration,
    InversePowerIteration,
    RayleighQuotientIteration,
    QREigenvalues,
    SVD,
)
from .ode import (
    Euler,
    Heun,
    RK2,
    RK4,
    BackwardEuler,
    ODETrapezoidal,
    AdamsBashforth,
    AdamsMoulton,
    PredictorCorrector,
    RK45,
    DormandPrince,
)
from .differentiation import (
    ForwardDiff,
    BackwardDiff,
    CentralDiff,
    CentralDiff4th,
    SecondDerivative,
    RichardsonExtrap,
)
from .fitting import PolyFit, LinearFit, ExpFit, NonlinearFit, plot_fit, plot_residuals

from .exceptions import (
    NumericalError,
    NonSquareMatrixError,
    SingularMatrixError,
    NotSymmetricError,
    NotPositiveDefiniteError,
    ConvergenceError,
    DomainError,
)

__all__ = [
    "__version__",
    # linear algebra
    "Matrix",
    "Vector",
    "forward_substitution",
    "backward_substitution",
    "QRGramSchmidt",
    "QRModifiedGramSchmidt",
    "QRHouseholder",
    "QRSolver",
    "LeastSquaresSolver",
    "LUDecomposition",
    "GaussJordan",
    "Jacobi",
    "GaussSeidel",
    "SOR",
    "Cholesky",
    # roots
    "Bisection",
    "RegulaFalsi",
    "Brent",
    "FixedPoint",
    "Secant",
    "NewtonRoot",
    "print_trace",
    # interpolation
    "NewtonInterpolation",
    "LagrangeInterpolation",
    "CubicSpline",
    "chebyshev_nodes",
    # quadrature
    "Trapezoidal",
    "Simpson",
    "GaussLegendre",
    "AdaptiveSimpson",
    "Romberg",
    "gauss_legendre_nodes",
    # eigenvalues / SVD
    "PowerIteration",
    "InversePowerIteration",
    "RayleighQuotientIteration",
    "QREigenvalues",
    "SVD",
    # ODEs
    "Euler",
    "Heun",
    "RK2",
    "RK4",
    "BackwardEuler",
    "ODETrapezoidal",
    "AdamsBashforth",
    "AdamsMoulton",
    "PredictorCorrector",
    "RK45",
    "DormandPrince",
    # differentiation
    "ForwardDiff",
    "BackwardDiff",
    "CentralDiff",
    "CentralDiff4th",
    "SecondDerivative",
    "RichardsonExtrap",
    # fitting
    "PolyFit",
    "LinearFit",
    "ExpFit",
    "NonlinearFit",
    "plot_fit",
    "plot_residuals",
    # exceptions
    "NumericalError",
    "NonSquareMatrixError",
    "SingularMatrixError",
    "NotSymmetricError",
    "NotPositiveDefiniteError",
    "ConvergenceError",
    "DomainError",
]
