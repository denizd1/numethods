class NumericalError(Exception):
    """Base class for numerical method errors."""
    pass

class NonSquareMatrixError(NumericalError):
    """Raised when a non-square matrix is provided where a square matrix is required."""
    pass

class SingularMatrixError(NumericalError):
    """Raised when a matrix is singular to working precision."""
    pass

class NotSymmetricError(NumericalError):
    """Raised when a matrix expected to be symmetric is not."""
    pass

class NotPositiveDefiniteError(NumericalError):
    """Raised when a matrix expected to be SPD is not."""
    pass

class ConvergenceError(NumericalError):
    """Raised when an iterative method fails to converge within limits."""
    pass

class DomainError(NumericalError):
    """Raised when inputs violate a method's domain assumptions."""
    pass
