"""Flags shared by the ``mpp`` modules.

Mirrors FMS ``mpp/mpp_parameter.F90``. The update flags are bit masks so a
caller can request several edges in one call, exactly as in FMS, where
``XUPDATE`` is ``WUPDATE + EUPDATE``.
"""

from __future__ import annotations

from typing import Final

#: Update the halo on one edge of the process grid.
WUPDATE: Final = 1 << 0
EUPDATE: Final = 1 << 1
SUPDATE: Final = 1 << 2
NUPDATE: Final = 1 << 3

#: Update both edges of one axis.
XUPDATE: Final = WUPDATE | EUPDATE
YUPDATE: Final = SUPDATE | NUPDATE

#: Update every edge, corners included.
BOTH_UPDATE: Final = XUPDATE | YUPDATE

#: Where a field sits relative to its grid cell.
CENTER: Final = "center"
CORNER: Final = "corner"

#: A folded edge joins an axis to itself in reverse, which is how a tripolar
#: ocean grid closes across the north pole. The halo point beyond the fold
#: comes from the mirrored position along the other axis, not from a
#: neighbouring rank in the usual sense.
FOLD_WEST_EDGE: Final = 1 << 4
FOLD_EAST_EDGE: Final = 1 << 5
FOLD_SOUTH_EDGE: Final = 1 << 6
FOLD_NORTH_EDGE: Final = 1 << 7

#: Grid staggering. Only ``AGRID`` (cell-centred) is handled at the fold so
#: far; the staggered forms need the half-cell offset FMS applies in
#: ``mpp_do_updateV``.
AGRID: Final = "agrid"
BGRID_NE: Final = "bgrid_ne"
CGRID_NE: Final = "cgrid_ne"

#: Boundary treatment of a partitioned axis. ``CYCLIC_GLOBAL_DOMAIN`` wraps
#: the first and last ranks into neighbours; ``GLOBAL_DATA_DOMAIN`` leaves the
#: outermost halo untouched.
GLOBAL_DATA_DOMAIN: Final = "global"
CYCLIC_GLOBAL_DOMAIN: Final = "cyclic"

#: Whether a global sum is accumulated in extended fixed point, making it
#: independent of the rank count, or in plain floating point.
BITWISE_EXACT_SUM: Final = True
NON_BITWISE_EXACT_SUM: Final = False
