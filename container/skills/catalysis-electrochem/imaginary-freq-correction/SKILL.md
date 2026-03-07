# Imaginary Frequency Correction

## When to Use

- Validate a transition state: a true TS should have exactly one imaginary vibrational mode along the reaction coordinate
- Fix spurious imaginary frequencies in adsorbate vibrational analysis from numerical noise or incomplete relaxation
- Correct thermodynamic quantities (ZPE, entropy, free energy) when small imaginary frequencies appear
- Corresponds to VASPKIT task 507 (Imaginary Frequency Correction)

## Method Selection

| Criterion | ASE + MACE (Method A) | QE DFT (Method B) | VASP (Method C) |
|---|---|---|---|
| Speed | Minutes | Hours | Hours |
| Accuracy | Good for screening | DFT quality | DFT quality |
| TS validation | Displace + re-optimize with MACE | PHonon at Gamma + dynmat | IBRION=5, parse OUTCAR |
| Adsorbate fix | Replace small imag with 12 cm-1 | Same correction on DFT freqs | VASPKIT 507 automated |
| When to use | Rapid screening, large systems | Publication quality | VASP license available |

```
Found imaginary frequencies in a transition state?
  --> Exactly 1 imaginary mode along reaction coordinate?
      YES --> TS is valid. Exclude that mode from ZPE sum.
      NO  --> Not a proper TS. Displace + re-optimize.

Found imaginary frequencies in an adsorbate or minimum?
  --> Small (< ~50 cm-1)? Likely numerical noise.
      Replace with 12 cm-1 (or |freq|) and recalculate thermodynamics.
  --> Large (> 50 cm-1)? Structure not at a minimum.
      Re-relax with tighter convergence (fmax < 0.001 eV/A).
```

## Prerequisites

- ASE with thermochemistry module (`ase.thermochemistry`)
- MACE-MP-0 (Method A)
- Quantum ESPRESSO pw.x, ph.x, dynmat.x (Method B)
- numpy, scipy, matplotlib
- A structure suspected to be a TS or a minimum with spurious imaginary modes

## Detailed Steps

### Method A: ASE + MACE -- Vibrational Analysis and Imaginary Frequency Correction

```python
#!/usr/bin/env python3
"""
Compute vibrational frequencies of an adsorbate/TS using ASE Vibrations +
MACE calculator, identify imaginary modes, correct thermodynamic quantities,
and verify TS by displacing along the imaginary mode.
Covers VASPKIT 507 functionality.
"""
import numpy as np
from ase.build import fcc111, molecule
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo
from ase.io import write, read
from ase import Atoms
from mace.calculators import mace_mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

# ============================================================
# 1. Build or load the system
# ============================================================
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")

# Example: CO on Cu(111). In practice: atoms = read("relaxed_ts.xyz")
slab = fcc111("Cu", size=(3, 3, 4), vacuum=15.0)
n_slab = len(slab)
co = molecule("CO")
slab_pos = slab.get_positions()
z_top = np.max(slab_pos[:, 2])
x_c, y_c = np.mean(slab_pos[:, 0]), np.mean(slab_pos[:, 1])
co.set_positions([[x_c, y_c, z_top + 1.9], [x_c, y_c, z_top + 3.028]])

system = slab + co
n_total = len(system)
n_adsorbate = n_total - n_slab
fix_slab = list(range(n_slab))
system.set_constraint(FixAtoms(indices=fix_slab))
system.calc = calc

print(f"System: {n_slab} slab + {n_adsorbate} adsorbate atoms")
opt = BFGS(system, logfile="relax_system.log")
opt.run(fmax=0.01, steps=300)
e_system = system.get_potential_energy()
print(f"E_system = {e_system:.6f} eV")
write("relaxed_system.xyz", system)

# ============================================================
# 2. Compute vibrational frequencies (adsorbate atoms only)
# ============================================================
adsorbate_indices = list(range(n_slab, n_total))
vib = Vibrations(system, indices=adsorbate_indices, name="vib_imag_check", delta=0.01)
vib.run()
freq_cm = vib.get_frequencies()
freq_ev = vib.get_energies()

print(f"\n{'Mode':<6} {'Freq (cm-1)':<16} {'Type':<12}")
print("-" * 36)
imaginary_modes = []
for i, (fcm, fev) in enumerate(zip(freq_cm, freq_ev)):
    if np.iscomplex(fcm) and fcm.imag > 1.0:
        imaginary_modes.append(i)
        print(f"  {i+1:<4} {fcm.imag:>10.2f}i       IMAGINARY")
    else:
        print(f"  {i+1:<4} {fcm.real:>10.2f}        real")

n_imag = len(imaginary_modes)
print(f"\nFound {n_imag} imaginary mode(s)")

# ============================================================
# 3. Classify and apply corrections
# ============================================================
IMAG_THRESHOLD_CM = 50.0   # below: numerical noise
REPLACE_FREQ_CM = 12.0     # replacement for small imaginary
is_transition_state = False # set True if expecting a TS

if is_transition_state:
    print("\n--- Transition State Analysis ---")
    if n_imag == 1:
        print("PASS: Exactly 1 imaginary mode (valid TS).")
    elif n_imag == 0:
        print("WARNING: No imaginary modes. This is a minimum, not a TS.")
    else:
        print(f"WARNING: {n_imag} imaginary modes. TS should have exactly 1.")

# ============================================================
# 4. Build corrected frequency list
# ============================================================
corrected_freqs_ev = []
correction_log = []

for i, (fcm, fev) in enumerate(zip(freq_cm, freq_ev)):
    is_imag = np.iscomplex(fcm) and fcm.imag > 1.0
    if is_imag:
        imag_val_cm = fcm.imag
        if is_transition_state and i == imaginary_modes[0]:
            correction_log.append(f"Mode {i+1}: {imag_val_cm:.2f}i -> EXCLUDED (TS)")
            continue
        if imag_val_cm < IMAG_THRESHOLD_CM:
            corrected_freqs_ev.append(REPLACE_FREQ_CM / 8065.54)
            correction_log.append(
                f"Mode {i+1}: {imag_val_cm:.2f}i -> {REPLACE_FREQ_CM:.1f} cm-1 (replaced)")
        else:
            corrected_freqs_ev.append(imag_val_cm / 8065.54)
            correction_log.append(
                f"Mode {i+1}: {imag_val_cm:.2f}i -> {imag_val_cm:.1f} cm-1 (|val|, re-relax!)")
    else:
        real_ev = fev.real if np.isreal(fev) else fev.real
        if real_ev > 0.001:
            corrected_freqs_ev.append(real_ev)

print("\n--- Corrections Applied ---")
for entry in correction_log:
    print(f"  {entry}")

# ============================================================
# 5. Corrected thermodynamics
# ============================================================
thermo = HarmonicThermo(vib_energies=corrected_freqs_ev)
zpe = thermo.get_ZPE_correction()
print(f"\nCorrected ZPE = {zpe:.4f} eV")

temperatures = [298.15, 400, 500, 600, 700, 800]
print(f"\n{'T (K)':<10} {'U_vib (eV)':<12} {'S_vib (eV/K)':<14} {'F_vib (eV)':<12}")
print("-" * 50)
results_T = []
for T in temperatures:
    u = thermo.get_internal_energy(T, verbose=False)
    s = thermo.get_entropy(T, verbose=False)
    f = thermo.get_helmholtz_energy(T, verbose=False)
    print(f"{T:<10.2f} {u:<12.6f} {s:<14.8f} {f:<12.6f}")
    results_T.append({"T_K": T, "U_vib_eV": float(u),
                       "S_vib_eV_per_K": float(s), "F_vib_eV": float(f)})

# ============================================================
# 6. Displace along imaginary mode to verify TS
# ============================================================
if n_imag >= 1:
    imag_idx = imaginary_modes[0]
    mode_vector = vib.get_mode(imag_idx)
    norm = np.linalg.norm(mode_vector)
    if norm > 1e-10:
        mode_vector = mode_vector / norm
    disp_mag = 0.1  # Angstrom

    for sign, label in [(+1, "forward"), (-1, "backward")]:
        displaced = system.copy()
        pos = displaced.get_positions()
        for j, aidx in enumerate(adsorbate_indices):
            pos[aidx] += sign * disp_mag * mode_vector[j]
        displaced.set_positions(pos)
        write(f"displaced_{label}.xyz", displaced)

    print("\nRe-optimizing displaced structures...")
    for label in ["forward", "backward"]:
        disp = read(f"displaced_{label}.xyz")
        disp.set_constraint(FixAtoms(indices=fix_slab))
        disp.calc = calc
        BFGS(disp, logfile=f"reopt_{label}.log").run(fmax=0.01, steps=200)
        e_disp = disp.get_potential_energy()
        write(f"reoptimized_{label}.xyz", disp)
        print(f"  {label}: E = {e_disp:.6f} eV (dE = {e_disp - e_system:.4f} eV)")
    print("  Valid TS: both directions should yield different, lower-energy minima.")

# ============================================================
# 7. Save results and plot
# ============================================================
results = {
    "system": system.get_chemical_formula(),
    "e_system_eV": float(e_system),
    "is_transition_state": is_transition_state,
    "n_imaginary_modes": n_imag,
    "imaginary_freqs_cm-1": [float(freq_cm[i].imag) for i in imaginary_modes],
    "corrected_frequencies_eV": [float(f) for f in corrected_freqs_ev],
    "zpe_corrected_eV": float(zpe),
    "corrections_applied": correction_log,
    "thermal_corrections": results_T,
}
with open("imaginary_freq_correction.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved imaginary_freq_correction.json")

fig, ax = plt.subplots(figsize=(10, 5))
vals, colors = [], []
for fcm in freq_cm:
    if np.iscomplex(fcm) and fcm.imag > 1.0:
        vals.append(-fcm.imag); colors.append("#d62728")
    else:
        vals.append(fcm.real); colors.append("#2ca02c")
ax.bar(range(1, len(vals)+1), vals, color=colors, edgecolor="black", linewidth=0.5)
ax.axhline(0, color="black", linewidth=1.0)
ax.axhline(-IMAG_THRESHOLD_CM, color="orange", linestyle="--",
           label=f"Noise threshold ({IMAG_THRESHOLD_CM} cm-1)")
ax.set_xlabel("Mode"); ax.set_ylabel("Frequency (cm-1)")
ax.set_title("Vibrational Spectrum (negative = imaginary)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig("frequency_spectrum.png", dpi=150); plt.close()
print("Saved frequency_spectrum.png")
vib.clean()
```

### Method B: QE DFT -- PHonon at Gamma and Correction

```python
#!/usr/bin/env python3
"""
Compute vibrational frequencies using QE PHonon (ph.x) at Gamma,
parse dynmat.x output, identify imaginary frequencies, and apply corrections.
DFT-quality frequencies for publication-grade work.
"""
import numpy as np
import subprocess
import os
import re
import json
from ase.thermochemistry import HarmonicThermo

PREFIX = "adsorbate"
WORK_DIR = os.path.abspath("./qe_phonon")
PSEUDO_DIR = os.path.abspath("./pseudo")
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, "tmp"), exist_ok=True)
NPROC = os.cpu_count() or 4
IMAG_THRESHOLD_CM = 50.0
REPLACE_FREQ_CM = 12.0

# 1. SCF input (example adsorbate on slab -- adjust for your system)
scf_input = f"""&CONTROL
    calculation='scf', prefix='{PREFIX}', outdir='./tmp',
    pseudo_dir='{PSEUDO_DIR}', tprnfor=.true.
/
&SYSTEM
    ibrav=0, nat=6, ntyp=2, ecutwfc=50.0, ecutrho=400.0,
    occupations='smearing', smearing='gaussian', degauss=0.005
/
&ELECTRONS
    conv_thr=1.0d-10
/
ATOMIC_SPECIES
  Cu 63.546 Cu.pbe-dn-kjpaw_psl.1.0.0.UPF
  C  12.011 C.pbe-n-kjpaw_psl.1.0.0.UPF
CELL_PARAMETERS angstrom
  7.669 0.000 0.000
  3.835 6.641 0.000
  0.000 0.000 25.00
ATOMIC_POSITIONS angstrom
  Cu 0.000 0.000 10.000
  Cu 2.556 0.000 10.000
  Cu 1.278 2.214 10.000
  Cu 1.278 0.738 12.084
  C  1.278 0.738 13.984
  C  1.278 0.738 15.112
K_POINTS gamma
"""
with open(os.path.join(WORK_DIR, "scf.in"), "w") as f:
    f.write(scf_input)

print("Running SCF...")
r = subprocess.run(["mpirun","--allow-run-as-root","-np",str(NPROC),
    "pw.x","-in","scf.in"], capture_output=True, text=True, timeout=1800, cwd=WORK_DIR)
with open(os.path.join(WORK_DIR,"scf.out"),"w") as f: f.write(r.stdout)

# 2. PHonon at Gamma (displace only adsorbate atoms 5,6)
ph_input = f"""Phonon at Gamma
&INPUTPH
    prefix='{PREFIX}', outdir='./tmp', tr2_ph=1.0d-14,
    fildyn='{PREFIX}.dyn', ldisp=.false., nat_todo=2, asr=.true.
/
0.0 0.0 0.0
5 6
"""
with open(os.path.join(WORK_DIR,"ph.in"),"w") as f: f.write(ph_input)
print("Running PHonon...")
r = subprocess.run(["mpirun","--allow-run-as-root","-np",str(NPROC),
    "ph.x","-in","ph.in"], capture_output=True, text=True, timeout=7200, cwd=WORK_DIR)
with open(os.path.join(WORK_DIR,"ph.out"),"w") as f: f.write(r.stdout)

# 3. dynmat.x to extract frequencies
dynmat_input = f"&INPUT\n    fildyn='{PREFIX}.dyn', asr='crystal'\n/\n"
with open(os.path.join(WORK_DIR,"dynmat.in"),"w") as f: f.write(dynmat_input)
r = subprocess.run(["dynmat.x","-in","dynmat.in"],
    capture_output=True, text=True, timeout=60, cwd=WORK_DIR)
with open(os.path.join(WORK_DIR,"dynmat.out"),"w") as f: f.write(r.stdout)

# 4. Parse frequencies from dynmat output
def parse_dynmat(outfile):
    freqs_cm = []
    with open(outfile) as f:
        for line in f:
            m = re.search(r'freq\s*\(\s*\d+\)\s*=\s*([-\d.]+)\s*\[THz\]\s*=\s*([-\d.]+)\s*\[cm-1\]', line)
            if m:
                freqs_cm.append(float(m.group(2)))
    return freqs_cm

dynmat_out = os.path.join(WORK_DIR, "dynmat.out")
if os.path.exists(dynmat_out):
    freq_cm_list = parse_dynmat(dynmat_out)
else:
    print("WARNING: Using example frequencies.")
    freq_cm_list = [1850.5, 420.3, 380.1, 290.7, -35.2, -12.8]

print(f"\n{'Mode':<6} {'Freq (cm-1)':<14} {'Type':<10}")
print("-" * 32)
for i, fcm in enumerate(freq_cm_list):
    tag = "IMAGINARY" if fcm < 0 else "real"
    print(f"  {i+1:<4} {fcm:>10.2f}     {tag}")

# 5. Apply corrections and compute thermodynamics
corrected_ev, corrections = [], []
for i, fcm in enumerate(freq_cm_list):
    if fcm < 0:
        abs_cm = abs(fcm)
        if abs_cm < IMAG_THRESHOLD_CM:
            corrected_ev.append(REPLACE_FREQ_CM / 8065.54)
            corrections.append(f"Mode {i+1}: {fcm:.1f} -> {REPLACE_FREQ_CM:.0f} cm-1 (noise)")
        else:
            corrected_ev.append(abs_cm / 8065.54)
            corrections.append(f"Mode {i+1}: {fcm:.1f} -> {abs_cm:.1f} cm-1 (|val|, re-relax!)")
    elif fcm > 1.0:
        corrected_ev.append(fcm / 8065.54)

print("\nCorrections:")
for c in corrections: print(f"  {c}")

thermo = HarmonicThermo(vib_energies=corrected_ev)
T = 298.15
zpe = thermo.get_ZPE_correction()
f_vib = thermo.get_helmholtz_energy(T, verbose=False)
print(f"\nCorrected ZPE = {zpe:.4f} eV")
print(f"F_vib({T:.0f} K) = {f_vib:.4f} eV")

with open(os.path.join(WORK_DIR,"imag_freq_correction_qe.json"),"w") as f:
    json.dump({"frequencies_cm-1": freq_cm_list, "corrections": corrections,
               "zpe_eV": float(zpe), "F_vib_eV": float(f_vib)}, f, indent=2)
print(f"Saved imag_freq_correction_qe.json")
```

### Method C: VASP -- Parse OUTCAR Frequencies and Correct (VASPKIT 507)

```python
#!/usr/bin/env python3
"""
Parse vibrational frequencies from VASP OUTCAR, identify imaginary modes,
apply corrections, and recalculate thermodynamic quantities.
Replicates VASPKIT task 507.

VASP INCAR settings for frequency calculation:
  IBRION=5, NSW=1, NFREE=2, POTIM=0.015, EDIFF=1E-7
  For adsorbates: selective dynamics to freeze slab.
"""
import numpy as np
import re
import json
from ase.thermochemistry import HarmonicThermo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAG_THRESHOLD_CM = 50.0
REPLACE_FREQ_CM = 12.0
IS_TRANSITION_STATE = False

def parse_vasp_frequencies(outcar="OUTCAR"):
    """Parse OUTCAR for lines like:
       1 f  = ... cm-1 ... meV     (real)
       2 f/i= ... cm-1 ... meV     (imaginary)
    """
    freqs = []
    with open(outcar) as f:
        in_section = False
        for line in f:
            if "Eigenvectors and eigenvalues" in line:
                in_section = True; freqs = []; continue
            if not in_section: continue
            m_real = re.match(r'\s*(\d+)\s+f\s+=.*?([\d.]+)\s+cm-1\s+([\d.]+)\s+meV', line)
            m_imag = re.match(r'\s*(\d+)\s+f/i\s*=.*?([\d.]+)\s+cm-1\s+([\d.]+)\s+meV', line)
            if m_real:
                freqs.append({"mode": int(m_real.group(1)), "freq_cm": float(m_real.group(2)),
                    "freq_mev": float(m_real.group(3)), "imaginary": False})
            elif m_imag:
                freqs.append({"mode": int(m_imag.group(1)), "freq_cm": float(m_imag.group(2)),
                    "freq_mev": float(m_imag.group(3)), "imaginary": True})
    return freqs

try:
    frequencies = parse_vasp_frequencies("OUTCAR")
except FileNotFoundError:
    print("WARNING: OUTCAR not found. Using example data.")
    frequencies = [
        {"mode":1,"freq_cm":1856.3,"freq_mev":230.1,"imaginary":False},
        {"mode":2,"freq_cm":425.8,"freq_mev":52.8,"imaginary":False},
        {"mode":3,"freq_cm":398.2,"freq_mev":49.4,"imaginary":False},
        {"mode":4,"freq_cm":285.6,"freq_mev":35.4,"imaginary":False},
        {"mode":5,"freq_cm":42.3,"freq_mev":5.2,"imaginary":True},
        {"mode":6,"freq_cm":15.1,"freq_mev":1.9,"imaginary":True},
    ]

print(f"\n{'Mode':<6} {'Freq (cm-1)':<14} {'Type':<10}")
print("-" * 32)
imag_modes = []
for fr in frequencies:
    tag = "IMAGINARY" if fr["imaginary"] else "real"
    sign = "-" if fr["imaginary"] else " "
    print(f"  {fr['mode']:<4} {sign}{fr['freq_cm']:>10.2f}    {tag}")
    if fr["imaginary"]: imag_modes.append(fr)

# Apply corrections
corrected_mev, corrections = [], []
for fr in frequencies:
    if fr["imaginary"]:
        if IS_TRANSITION_STATE and fr == imag_modes[0]:
            corrections.append(f"Mode {fr['mode']}: {fr['freq_cm']:.1f}i -> EXCLUDED (TS)")
            continue
        if fr["freq_cm"] < IMAG_THRESHOLD_CM:
            corrected_mev.append(REPLACE_FREQ_CM / 8065.54 * 1000)
            corrections.append(f"Mode {fr['mode']}: {fr['freq_cm']:.1f}i -> {REPLACE_FREQ_CM:.0f} cm-1")
        else:
            corrected_mev.append(fr["freq_mev"])
            corrections.append(f"Mode {fr['mode']}: {fr['freq_cm']:.1f}i -> {fr['freq_cm']:.1f} cm-1 (|val|)")
    elif fr["freq_mev"] > 0.1:
        corrected_mev.append(fr["freq_mev"])

print("\n--- VASPKIT 507: Corrections ---")
for c in corrections: print(f"  {c}")

corrected_ev = [m / 1000.0 for m in corrected_mev]
thermo = HarmonicThermo(vib_energies=corrected_ev)
zpe = thermo.get_ZPE_correction()
print(f"\nCorrected ZPE = {zpe:.4f} eV")

temperatures = [298.15, 400, 500, 600, 700, 800]
print(f"\n{'T (K)':<10} {'U_vib':<12} {'S_vib':<14} {'F_vib':<12}")
print("-" * 50)
results_T = []
for T in temperatures:
    u = thermo.get_internal_energy(T, verbose=False)
    s = thermo.get_entropy(T, verbose=False)
    fv = thermo.get_helmholtz_energy(T, verbose=False)
    print(f"{T:<10.2f} {u:<12.4f} {s:<14.8f} {fv:<12.4f}")
    results_T.append({"T_K":T,"U_vib_eV":float(u),"S_vib_eV_per_K":float(s),"F_vib_eV":float(fv)})

# Plot original vs corrected
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
orig_vals = [-fr["freq_cm"] if fr["imaginary"] else fr["freq_cm"] for fr in frequencies]
orig_cols = ["#d62728" if fr["imaginary"] else "#2ca02c" for fr in frequencies]
ax1.bar(range(1,len(orig_vals)+1), orig_vals, color=orig_cols, edgecolor="black", linewidth=0.5)
ax1.axhline(0, color="black"); ax1.axhline(-IMAG_THRESHOLD_CM, color="orange", linestyle="--")
ax1.set_xlabel("Mode"); ax1.set_ylabel("Freq (cm-1)"); ax1.set_title("Original")

corr_cm = [e*8065.54 for e in corrected_ev]
ax2.bar(range(1,len(corr_cm)+1), corr_cm, color="#2ca02c", edgecolor="black", linewidth=0.5)
ax2.axhline(0, color="black"); ax2.set_xlabel("Mode"); ax2.set_title("Corrected")
fig.suptitle("Imaginary Frequency Correction (VASPKIT 507)", fontsize=13)
fig.tight_layout(); fig.savefig("imag_freq_correction_vasp.png", dpi=150); plt.close()
print("Saved imag_freq_correction_vasp.png")

with open("imag_freq_correction_vasp.json","w") as f:
    json.dump({"original": frequencies, "corrections": corrections,
               "corrected_eV": [float(e) for e in corrected_ev],
               "zpe_eV": float(zpe), "thermal": results_T}, f, indent=2)
print("Saved imag_freq_correction_vasp.json")
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| Imaginary threshold | 50 cm-1 | Below: likely noise. Above: structural issue. |
| Replacement frequency | 12 cm-1 | Common approximation. Some use 50 cm-1 or |freq|. |
| Displacement delta | 0.01 A (ASE), 0.015 A (VASP POTIM) | Too small: noisy forces. Too large: anharmonic. |
| fmax for relaxation | < 0.001 eV/A | Must be tight before frequency calculation. |
| TS imaginary freq | > 100 cm-1 typical | Valid TS mode is 100-2000 cm-1 depending on process. |
| ASR (acoustic sum rule) | 'crystal' in dynmat.x | Removes 3 acoustic modes at Gamma. |
| TS verification displacement | 0.05-0.2 A | Along imaginary eigenvector, then re-optimize both ways. |

## Interpreting Results

1. **TS validation**: Valid TS has exactly 1 imaginary mode whose eigenvector corresponds to the reaction coordinate (bond breaking/forming). Animate the mode to verify.
2. **TS with 0 imaginary modes**: Structure is a minimum. Use NEB or dimer method to find the saddle point.
3. **TS with 2+ imaginary modes**: Higher-order saddle. Displace along extra modes and re-optimize.
4. **Small imaginary in adsorbate (< 50 cm-1)**: Frustrated translations/rotations. Replace with 12 cm-1; ZPE changes < 0.01 eV.
5. **Large imaginary in adsorbate (> 50 cm-1)**: Geometry not relaxed. Re-optimize with fmax < 0.001 eV/A.
6. **TS displacement test**: Both forward and backward re-optimizations should yield different, lower-energy minima.

## Common Issues

| Issue | Solution |
|---|---|
| All frequencies imaginary | Structure far from stationary point. Relax first with proper constraints. |
| TS has 0 imaginary modes | Not a TS. Use NEB or dimer method. |
| TS has 2+ imaginary modes | Higher-order saddle. Displace along extra modes, re-optimize. |
| Small imaginary in adsorbate | Replace with 12 cm-1 or |freq|. Tighten relaxation if > 50 cm-1. |
| MACE vs DFT frequency disagreement | Expected for unusual bonding. Use QE/VASP for publications. |
| VASP OUTCAR parsing fails | Check IBRION=5, NSW=1, NFREE=2 were set. |
| dynmat.x crashes | Ensure ph.x completed and produced .dyn file. Check ASR setting. |
| ZPE barely changes after correction | Expected for small imaginary modes. More important for entropy. |
