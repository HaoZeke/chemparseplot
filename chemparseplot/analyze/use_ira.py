import dataclasses

import ira_mod
import numpy as np


@dataclasses.dataclass
class IRAComp:
    rot: np.array
    trans: np.array
    perm: np.array
    hd: float


class IncomparableStructuresError(ValueError):
    """Custom exception raised for incompatible atomistic structures."""

    pass


def _perform_ira_match(atm1, atm2, k_factor=2.8):
    """Performs IRA matching. Internal function."""
    if len(atm1) != len(atm2):
        errmsg = "Atomistic structures must have the same number of atoms."
        raise IncomparableStructuresError(errmsg)
    if not np.array_equal(atm1.symbols, atm2.symbols):
        errmsg = "Atomistic structures must have the same atom types."
        raise IncomparableStructuresError(errmsg)

    ira = ira_mod.IRA()
    return ira.match(
        len(atm1),
        atm1.get_atomic_numbers(),
        atm1.get_positions(),
        len(atm2),
        atm2.get_atomic_numbers(),
        atm2.get_positions(),
        k_factor,
    )


def is_ira_pair(atm1, atm2, hd_tol=1, k_factor=2.8):
    """Checks if two atomistic structures are an IRA pair."""
    try:
        _, _, _, hd = _perform_ira_match(atm1, atm2, k_factor)
        return hd < hd_tol
    except IncomparableStructuresError:
        return False


def do_ira(atm1, atm2, k_factor=2.8):
    """Performs IRA matching on two atomistic structures."""
    rotation, translation, perm, hd = _perform_ira_match(atm1, atm2, k_factor)
    return IRAComp(rot=rotation, trans=translation, perm=perm, hd=hd)
