#!/usr/bin/env python
"""Compare monofonIC (generator = FILE) with Simbelmyne/pysbmy on the same white noise.

Steps (all in --workdir):
  1. x = numpy default_rng(seed).standard_normal((N,N,N), float32), written as a pysbmy Field (wn.h5).
  2. monofonIC reads wn.h5 (random/generator = FILE), output format simbelmyne -> mono_DM_{phi,phi2,delta}.h5.
  3. Simbelmyne gets the *same* P(k): monofonIC's input_powerspec.txt, rescaled to z=0, evaluated on
     pysbmy's k modes. Then pysbmy (ICsMode=1, sCOLA init step) writes lpt_phi1.h5, lpt_phi2.h5, ics.h5.
  4. Report max|a-b|/rms(b) and rms(a-b)/rms(b) for delta, phi1, phi2, plus
     - phi1 with monofonIC's exact -1/k^2 replaced by Simbelmyne's FD2 Green function (operator removed:
       what is left is sign, normalisation, Fourier convention, axis order, offsets, P(k) interpolation),
     - per-shell cross-correlation and amplitude ratio (an offset shows up as r<1 growing with k,
       a transpose as r~0, a sign error as ratio<0).

Usage (needs pysbmy importable and a monofonIC built with the FILE plugin):
  python compare_external_noise_sbmy.py --monofonic /path/to/monofonIC --workdir /tmp/cmp --n 64 --box 250
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def read_field(path):
    with h5py.File(path, "r") as f:
        return f["scalars/field"][...].astype(np.float64)


def write_noise(path, x, box):
    from pysbmy.field import Field
    n = x.shape[0]
    Field(L0=box, L1=box, L2=box, corner0=0.0, corner1=0.0, corner2=0.0, rank=1, N0=n, N1=n, N2=n,
          time=1e-3, data=np.ascontiguousarray(x, np.float32)).write(str(path))


def run_monofonic(exe, work, n, box, zstart, transfer, nranks, threads):
    conf = work / "mono.conf"
    conf.write_text(f"""[setup]
GridRes = {n}
BoxLength = {box}
zstart = {zstart}
LPTorder = 2
DoBaryons = no
DoBaryonVrel = no
DoFixing = no
DoInversion = no
ParticleLoad = sc

[cosmology]
ParameterSet = none
H0 = 67.7
Omega_m = 0.311
Omega_b = 0.049
Omega_L = 0.689
n_s = 0.967
sigma_8 = 0.815
m_nu1 = 0.0
m_nu2 = 0.0
m_nu3 = 0.0
transfer = {transfer}
ztarget = {zstart}
ZeroRadiation = true

[random]
generator = FILE
noise_file = {work / 'wn.h5'}
noise_dataset = /scalars/field

[execution]
NumThreads = {threads}

[output]
format = simbelmyne
filename = {work / 'mono_'}
""")
    cmd = ([os.environ.get("MPIEXEC", "mpirun"), "-np", str(nranks)] if nranks > 1 else []) + [str(exe), str(conf)]
    proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    (work / "mono.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        sys.exit(f"monofonIC failed, see {work / 'mono.log'}")
    dstart = float(re.search(r"D\+_start = ([0-9.eE+-]+)", proc.stdout + proc.stderr).group(1))
    return dstart


def power_from_monofonic(work, box, n, dstart):
    """pysbmy PowerSpectrum on its own k modes with monofonIC's P(k, z=0).

    monofonIC: delta_k(unitary FFT) = w_k * A(k) / (dx/2pi)^1.5, so P(k) = (2 pi)^3 A(k)^2; the file
    column 2 is (A(k) D+_start)^2.
    """
    from pysbmy.power import PowerSpectrum
    tab = np.loadtxt(next(work.glob("*input_powerspec.txt")))
    k_tab, p_tab = tab[:, 0], tab[:, 1] / dstart**2 * (2 * np.pi) ** 3
    grid = PowerSpectrum(box, box, box, n, n, n, PowerSpectrum.cosmo_default)
    k = np.asarray(grid.k_modes, np.float64)
    pk = np.zeros_like(k)
    ok = k > 0
    pk[ok] = np.exp(np.interp(np.log(k[ok]), np.log(k_tab), np.log(p_tab)))
    out = PowerSpectrum(box, box, box, n, n, n, PowerSpectrum.cosmo_default, k_keys=grid.k_keys,
                        k_modes=grid.k_modes, k_nmodes=grid.k_nmodes, powerspectrum=pk)
    out.write(str(work / "power.h5"))


def run_simbelmyne(work, n, box, zstart, threads):
    from pysbmy.pysbmy import param_file
    from pysbmy.timestepping import StandardTimeStepping
    ai = 1.0 / (1.0 + zstart)
    try:   # newer pysbmy (CC); the time stepping does not affect the potentials
        ts = StandardTimeStepping(ai, 1.0, 2, 3, TimeStepDistribution=0, ModifiedDiscretization=True)
    except TypeError:
        ts = StandardTimeStepping(ai, 1.0, 2, 3, TimeStepDistribution=0)
    ts.write(str(work / "ts.h5"))
    card = param_file(
        SnapFormat=3, ModuleLPT=1, Particles=n, Mesh=n, BoxSize=box, corner0=0.0, corner1=0.0, corner2=0.0,
        ICsMode=1, InputWhiteNoise=str(work / "wn.h5"), InputPowerSpectrum=str(work / "power.h5"),
        WriteInitialConditions=1, OutputInitialConditions=str(work / "ics.h5"),
        RedshiftLPT=zstart, WriteLPTSnapshot=0, WriteLPTDensity=0, WriteLPTDisplacements=0,
        ModulePMCOLA=1, EvolutionMode=3, ParticleMesh=n, TimeStepDistribution=str(work / "ts.h5"),
        NumberOfTilesPerDimension=2, NumberOfParticlesInBuffer=8, NumberOfParticlesInLPTBuffer=8,
        OutputLPTPotential1=str(work / "lpt_phi1.h5"), OutputLPTPotential2=str(work / "lpt_phi2.h5"),
        OutputTilesBase=str(work / "tile_"), OutputTilesExt=".h5",
        RedshiftFCs=0.0, WriteFinalSnapshot=0, GenerateLightcone=0,
        WriteFinalDensity=1, OutputFinalDensity=str(work / "final_density.h5"),  # 0 makes sbmy read a density
        ModuleHalos=0, ModuleRSD=0, ModuleMocks=0)
    card.write(str(work / "card.sbmy"))
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    proc = subprocess.run([sys.executable, "-m", "pysbmy", str(work / "card.sbmy"), str(work / "sbmy_log.txt")],
                          cwd=work, env=env, capture_output=True, text=True)
    if proc.returncode != 0 or not (work / "lpt_phi2.h5").exists():
        sys.exit(f"pysbmy failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")


def diff(a, b):
    s = np.sqrt(np.mean(b**2))
    return np.max(np.abs(a - b)) / s, np.sqrt(np.mean((a - b) ** 2)) / s


def kgrid(n, box):
    k1 = 2 * np.pi * np.fft.fftfreq(n, d=box / n)
    kz = 2 * np.pi * np.fft.rfftfreq(n, d=box / n)
    return k1[:, None, None], k1[None, :, None], kz[None, None, :]


def shells(a, b, n, box, nbins=8):
    """Cross-correlation r and amplitude ratio <Re(A B*)>/<|B|^2> in |k| shells (up to k_Ny)."""
    A, B = np.fft.rfftn(a), np.fft.rfftn(b)
    kx, ky, kz = kgrid(n, box)
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    kny = np.pi * n / box
    edges = np.linspace(0, kny, nbins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (k > lo) & (k <= hi)
        ab = np.sum((A * np.conj(B))[m]).real
        rows.append(((lo + hi) / 2 / kny, ab / np.sqrt(np.sum(np.abs(A[m]) ** 2) * np.sum(np.abs(B[m]) ** 2)),
                     ab / np.sum(np.abs(B[m]) ** 2)))
    return rows


def to_fd2(phi, n, box):
    """Replace the exact Green function -1/k^2 by Simbelmyne's FD2 one, 1/sum(-4 sin^2(k dx/2)/dx^2)."""
    dx = box / n
    kx, ky, kz = kgrid(n, box)
    k2 = kx**2 + ky**2 + kz**2
    lam = sum(-4 * np.sin(q * dx / 2) ** 2 / dx**2 for q in (kx, ky, kz))
    k2[0, 0, 0], lam[0, 0, 0] = 1.0, -1.0
    return np.fft.irfftn(np.fft.rfftn(phi) * (k2 / -lam), s=phi.shape, axes=(0, 1, 2))


def phi2_from_phi1(phi, box, fd2):
    """phi2 = L^-1 sum_{a<b} (phi_aa phi_bb - phi_ab^2).

    fd2=True: Simbelmyne's operators (FD2 gradient sin(k dx)/dx per axis, products on the N grid, FD2
    Poisson). fd2=False: exact ik operators with 3/2 zero padding, as monofonIC's Orszag convolver.
    """
    n = phi.shape[0]
    dx = box / n
    F = np.fft.rfftn(phi)
    m = n if fd2 else 3 * n // 2
    if not fd2:   # zero-pad to m^3 (Nyquist planes dropped)
        G = np.zeros((m, m, m // 2 + 1), complex)
        h = n // 2
        for si, di in ((slice(0, h), slice(0, h)), (slice(n - h + 1, n), slice(m - h + 1, m))):
            for sj, dj in ((slice(0, h), slice(0, h)), (slice(n - h + 1, n), slice(m - h + 1, m))):
                G[di, dj, :h] = F[si, sj, :h]
        F = G * (m / n) ** 3
    kx, ky, kz = kgrid(m, box)
    g = [np.sin(q * dx) / dx for q in (kx, ky, kz)] if fd2 else [kx, ky, kz]
    H = {(a, b): np.fft.irfftn(-g[a] * g[b] * F, s=(m,) * 3, axes=(0, 1, 2)) for a in range(3) for b in range(a, 3)}
    src = (H[0, 0] * H[1, 1] + H[0, 0] * H[2, 2] + H[1, 1] * H[2, 2]
           - H[0, 1] ** 2 - H[0, 2] ** 2 - H[1, 2] ** 2)
    S = np.fft.rfftn(src)
    if not fd2:
        S = S[np.r_[0:n // 2, m - n // 2:m]][:, np.r_[0:n // 2, m - n // 2:m]][:, :, :n // 2 + 1] * (n / m) ** 3
        S[n // 2], S[:, n // 2], S[:, :, n // 2] = 0, 0, 0
    kx, ky, kz = kgrid(n, box)
    lam = sum(-4 * np.sin(q * dx / 2) ** 2 / dx**2 for q in (kx, ky, kz)) if fd2 else -(kx**2 + ky**2 + kz**2)
    lam[0, 0, 0] = 1.0
    S /= lam
    S[0, 0, 0] = 0
    return np.fft.irfftn(S, s=(n,) * 3, axes=(0, 1, 2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--monofonic", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--box", type=float, default=250.0)
    p.add_argument("--zstart", type=float, default=24.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--transfer", default="eisenstein", help="monofonIC transfer plugin (eisenstein, CLASS)")
    p.add_argument("--nranks", type=int, default=1, help="MPI ranks for monofonIC")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()

    work = Path(a.workdir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    n, box = a.n, a.box
    x = np.random.default_rng(a.seed).standard_normal((n, n, n), dtype=np.float32)
    write_noise(work / "wn.h5", x, box)
    dstart = run_monofonic(Path(a.monofonic).resolve(), work, n, box, a.zstart, a.transfer, a.nranks, a.threads)
    power_from_monofonic(work, box, n, dstart)
    run_simbelmyne(work, n, box, a.zstart, a.threads)

    m = {k: read_field(work / f"mono_DM_{k}.h5") for k in ("delta", "phi", "phi2")}
    s = dict(delta=read_field(work / "ics.h5"), phi=read_field(work / "lpt_phi1.h5"), phi2=read_field(work / "lpt_phi2.h5"))
    print(f"N={n} L={box} seed={a.seed} transfer={a.transfer} ranks={a.nranks}")
    print("field            max|m-s|/rms(s)  rms(m-s)/rms(s)")
    for name, mm, ss in (("delta", m["delta"], s["delta"]), ("phi1", m["phi"], s["phi"]),
                         ("phi1 (FD2 op)", to_fd2(m["phi"], n, box), s["phi"]), ("phi2", m["phi2"], s["phi2"]),
                         ("phi2 (FD2 ops)", phi2_from_phi1(to_fd2(m["phi"], n, box), box, True), s["phi2"]),
                         ("phi2 mono/exact", m["phi2"], phi2_from_phi1(m["phi"], box, False))):
        mx, rms = diff(mm, ss)
        print(f"{name:16s} {mx:15.3e}  {rms:15.3e}")
    for name, key in (("delta", "delta"), ("phi1", "phi"), ("phi2", "phi2")):
        print(f"{name}: k/k_Ny, cross-correlation r, amplitude ratio mono/sbmy")
        for kk, r, ratio in shells(m[key], s[key], n, box):
            print(f"   {kk:5.3f}  {r: .6f}  {ratio: .6f}")


if __name__ == "__main__":
    main()
