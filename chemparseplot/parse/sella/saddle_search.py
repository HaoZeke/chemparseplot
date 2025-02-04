from pathlib import Path
from chemparseplot.basetypes import SaddleMeasure, SpinID


def parse_sella_saddle(eresp: Path, rloc: SpinID):
    respth = eresp / "npes.txt"
    if respth.exists():
        rdat = respth.read_text()
    else:
        return SaddleMeasure()  # default is a failure state
    npes = int(respth.read_text().split()[-1])
    logdat = (
        [f for f in eresp.iterdir() if "log" in str(f)][0]
        .read_text()
        .strip()
        .split("\n")
    )
    _, _, tstart, init_energy, _, _, _, _ = logdat[1].split()
    _, iter_steps, tend, saddle_energy, saddle_fmax, _, _, _ = logdat[-1].split()

    return SaddleMeasure(
        pes_calls=npes,
        iter_steps=int(iter_steps),
        tot_time=one_day_tdelta(tend, tstart),
        saddle_energy=float(saddle_energy),
        saddle_fmax=float(saddle_fmax),
        init_energy=float(init_energy),
        barrier=float(saddle_energy) - float(init_energy),
        success=True,
        method="Sella",
        mol_id=rloc.mol_id,
        spin=rloc.spin,
    )
