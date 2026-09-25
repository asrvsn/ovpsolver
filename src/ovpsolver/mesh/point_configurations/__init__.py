"""Where to put equal inclusions, so that none of them overlap.

Hard-coded centres record an arrangement without recording what it was meant to
be. These build one from a rule and a seed instead, so a geometry file says which
pattern it wanted and the run can be reproduced from the spec alone.

Split by the region the rule places into, because the spacing a pattern settles
at, and where its boundary layer is, belong to the domain and not to the
inclusions. :mod:`.disk` is the only one so far.
"""

from __future__ import annotations

from .disk import (
    lloyd,
    log_gas_hard_sphere,
    thinned_ginibre,
)

__all__ = [
    "lloyd",
    "log_gas_hard_sphere",
    "thinned_ginibre",
]
