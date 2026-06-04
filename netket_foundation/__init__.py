from netket_foundation._src.hilbert.parameter_space import (
    ParameterSpace as ParameterSpace,
)
from netket_foundation._src.vqs.state import (
    FoundationalQuantumState as FoundationalQuantumState,
)


from netket_foundation._src.driver.ngd.driver_vmc_ngd import VMC_SR as VMC_SR

from netket_foundation import operator as operator
from netket_foundation import expectation_value as expectation_value
from netket_foundation import observable as observable
from netket_foundation import vqs as vqs
from netket_foundation import model as model

from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr

_deprecations = {
    # June 2026 — VMC_NG was renamed to VMC_SR to match netket.driver.VMC_SR.
    "VMC_NG": (
        "netket_foundation.VMC_NG has been renamed to netket_foundation.VMC_SR "
        "to match netket.driver.VMC_SR. The old name is now deprecated and will be "
        "removed in a future release. Please update your code by changing occurrences "
        "of `VMC_NG` with `VMC_SR`.",
        VMC_SR,
    ),
}

__getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr
