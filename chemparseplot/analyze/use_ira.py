import dataclasses

import ira_mod
import numpy as np


@dataclasses.dataclass
class IRAComp:
    rot: np.array
    trans: np.array
    perm: np.array
    hd: float


def is_ira_pair(atm1, atm2, hd_tol=1, k_factor=2.8):
    ira = ira_mod.IRA()
    if len(atm1) != len(atm2):
        return False
    if np.all(atm1.symbols == atm2.symbols):
        hd = ira.match(
            len(atm1),
            atm1.get_atomic_numbers(),
            atm1.get_positions(),
            len(atm2),
            atm2.get_atomic_numbers(),
            atm2.get_positions(),
            k_factor,
        )[-1]
        if hd < hd_tol:
            return True
        else:
            return False
    else:
        return False


def do_ira(atm1, atm2, k_factor=2.8):
    ira = ira_mod.IRA()
    if len(atm1) != len(atm2):
        errmsg = "Can't match different sized fragments"
        raise ValueError(errmsg)
    if np.all(atm1.symbols == atm2.symbols):
        rotation, translation, perm, hd = ira.match(
            len(atm1),
            atm1.get_atomic_numbers(),
            atm1.get_positions(),
            len(atm2),
            atm2.get_atomic_numbers(),
            atm2.get_positions(),
            k_factor,
        )
        return IRAComp(rot=rotation, trans=translation, perm=perm, hd=hd)
