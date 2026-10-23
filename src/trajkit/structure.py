import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from MDAnalysis.analysis.rms import RMSF
from pathlib import Path
import pandas as pd

def average_structure(topology: str, trajectory: Path) -> mda.Universe:
    """
    Compute the mean structure of a trajectory as a reference universe.

    Frames are aligned to frame 0 (backbone) on-the-fly and positions are
    accumulated incrementally, so memory usage is O(n_atoms) rather than
    O(n_frames × n_atoms). This avoids loading the full trajectory into RAM.
    """
    u = mda.Universe(topology, str(trajectory))

    # Store reference positions from frame 0
    u.trajectory[0]
    bb        = u.select_atoms("backbone")
    ref_bb    = bb.positions.copy()
    ref_com   = ref_bb.mean(axis=0)
    ref_centered = ref_bb - ref_com

    mean_pos = np.zeros((u.atoms.n_atoms, 3))

    for ts in u.trajectory:
        mobile_bb      = bb.positions.copy()
        mobile_com     = mobile_bb.mean(axis=0)
        mobile_centered = mobile_bb - mobile_com

        R, _ = align.rotation_matrix(mobile_centered, ref_centered)

        # Align all atoms using the backbone rotation
        aligned = np.dot(u.atoms.positions - mobile_com, R.T) + ref_com
        mean_pos += aligned

    mean_pos /= len(u.trajectory)

    tmp = u.copy()
    tmp.atoms.positions = mean_pos
    ref_uni = mda.Merge(tmp.atoms)
    
    return ref_uni


def build_reference(
    topology: str,
    trajectory: Path,
    reference: str | int,
    ) -> mda.Universe:
    
    if reference == 'average':
        ref_pdb = trajectory.parent / f'traj_average.pdb'
        if not ref_pdb.exists():
            ref_uni =  average_structure(topology, trajectory)
            ref_uni.atoms.write(str(ref_pdb))
            return ref_uni
        else: 
            return mda.Universe(str(ref_pdb))

    elif isinstance(reference, int):
        tmp = mda.Universe(topology, str(trajectory))
        tmp.trajectory[reference]
        return mda.Merge(tmp.atoms)

    elif isinstance(reference, (str, Path)):
        return mda.Universe(topology, str(reference))

    else:
        ref = mda.Universe(topology, str(trajectory))
        ref.trajectory[0]
        return ref


def compute_rmsd(
    topology: str,
    trajectory: Path,
    reference: mda.Universe,
    selection: str,
    total_time_ns: float,
    align: str = 'backbone'
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute backbone RMSD relative to the given reference Universe.

    DCD files do not store reliable time metadata, so the time axis is
    constructed as a linear span from 0 to total_time_ns.
    Returns time in ns and RMSD in Å.
    """
    u = mda.Universe(topology, str(trajectory))

    R = rms.RMSD(u, reference, select='backbone', groupselections=selection)
    R.run()

    n_frames = len(R.results.rmsd)
    times    = np.linspace(0, total_time_ns, n_frames)
    rmsd     = R.results.rmsd.T
    rmsd[1]     = times
    rmsd    = rmsd.T
    return rmsd

def rmsd(SIMULATIONS):
    results: dict[str, dict[str, list]] = {}

    for sim_name, cfg in SIMULATIONS.items():
        trajectories = cfg["trajectories"]
        results[sim_name] = {}

        for replica, traj in enumerate(trajectories):
            results[sim_name][traj] = []
            print(f"  {traj}")

            if isinstance(cfg["reference"], list):
                reference = cfg["reference"][replica]
            else:
                reference = cfg["reference"]

            ref = build_reference(
                cfg["topology"], traj, reference
            )

            for _, sett in cfg['rmsd'].items():
                rmsd_data = compute_rmsd(
                    cfg["topology"], traj, ref, 
                    list(sett["selections"].values()), cfg["total_time_ns"], align=sett["align"]
                )

                column_names = ['Frame', 'Time (ns)', 'backbone']
                column_names.extend(list(sett["selections"].keys()))
                df = pd.DataFrame(rmsd_data, columns=column_names)
                results[sim_name][traj].append(df)

    return results

def rmsf(
    topology: str,
    trajectory: Path,
    total_time_ns: float,
    start_ns: int | bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-residue RMSF using C-alpha atoms.
    """
    u  = mda.Universe(topology, str(trajectory))
    ca = u.select_atoms("name CA and protein")

    if start_ns:
        start = int(u.trajectory.n_frames * (1 - start_ns / total_time_ns))
    else:
        start = 0
    
    ref = mda.Universe(topology)
    ref.atoms.positions = u.trajectory[start].positions.copy()
 
    align.AlignTraj(
        u, ref, select="backbone").run()
 
    rmsf = RMSF(ca).run(start=start)
 
    return ca.resids, rmsf.results.rmsf