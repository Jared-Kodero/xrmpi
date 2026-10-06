"""Exchange several fields in one round of messages.

Mirrors FMS ``mpp/include/mpp_group_update.fh``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from .mpp_do_update import mpp_update_domains
from .mpp_domains import Domain


@dataclass
class GroupUpdate:
    """A set of fields exchanged together on one axis.

    Attributes
    ----------
    fields : dict[str, numpy.ndarray]
        Fields taking part, keyed by name.
    domain : Domain
        Domain the fields are laid out on.
    dim : str
        Axis being exchanged.
    axis : int
        Array axis corresponding to ``dim``.
    before, after : int
        Requested halo widths.
    """

    fields: dict[str, np.ndarray[Any, Any]]
    domain: Domain
    dim: str
    axis: int
    before: int
    after: int


def mpp_create_group_update(
    domain: Domain,
    dim: str,
    axis: int,
    *,
    before: int,
    after: int,
) -> GroupUpdate:
    """Open a group that several fields can be added to before exchanging.

    Exchanging fields one at a time costs one message round each. FMS groups
    them so a single round carries all of them; :func:`mpp_do_group_update`
    performs that round.

    Parameters
    ----------
    domain : Domain
        Domain the fields are laid out on.
    dim : str
        Axis to exchange.
    axis : int
        Array axis corresponding to ``dim``.
    before, after : int
        Halo widths to request.

    Returns
    -------
    GroupUpdate
        Empty group; add fields with :func:`mpp_add_to_group_update`.
    """
    return GroupUpdate({}, domain, dim, axis, before, after)


def mpp_add_to_group_update(
    group: GroupUpdate, name: str, field_data: np.ndarray[Any, Any]
) -> None:
    """Add one field to a group that has not been exchanged yet.

    Parameters
    ----------
    group : GroupUpdate
        Group to extend.
    name : str
        Name to retrieve the result under.
    field_data : numpy.ndarray
        This rank's compute-domain values.
    """
    group.fields[name] = field_data


def mpp_do_group_update(
    group: GroupUpdate,
) -> tuple[dict[str, np.ndarray[Any, Any]], int, int]:
    """Exchange every field in a group in one round of messages.

    Parameters
    ----------
    group : GroupUpdate
        Group to exchange.

    Returns
    -------
    tuple[dict[str, numpy.ndarray], int, int]
        Padded fields by name, and the halo width actually received before
        and after, which is narrower than requested at a non-cyclic edge.

    Raises
    ------
    ValueError
        If the group is empty.
    """
    if not group.fields:
        raise ValueError("mpp_do_group_update: the group has no fields.")
    padded, low, high = mpp_update_domains(
        group.fields,
        group.domain,
        group.dim,
        group.axis,
        before=group.before,
        after=group.after,
        periodic=group.domain.cyclic.get(group.dim, False),
    )
    return cast("dict[str, np.ndarray[Any, Any]]", padded), low, high
