#!/usr/bin/env python3
"""
Docker container smoke test.
Verifies the computation stack is functional without requiring an LLM API key.
Exit 0 = all checks passed, Exit 1 = something is broken.
"""

import sys
import subprocess

passed = 0
failed = 0

def check(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        failed += 1

# ── Python packages ──

def test_mace():
    from mace.calculators import mace_mp
    from ase.build import bulk
    calc = mace_mp(model="medium", default_dtype="float64")
    si = bulk("Si", "diamond", 5.43)
    si.calc = calc
    e = si.get_potential_energy()
    assert -12 < e < -10, f"Si energy {e:.4f} eV out of range [-12, -10]"

def test_pymatgen():
    from pymatgen.core import Structure, Lattice
    s = Structure(Lattice.cubic(5.43), ["Si", "Si"],
                  [[0, 0, 0], [0.25, 0.25, 0.25]])
    assert s.num_sites == 2
    assert "Si" in s.formula

def test_ase():
    from ase.build import bulk
    from ase.optimize import BFGS
    cu = bulk("Cu", "fcc", 3.6)
    assert len(cu) == 1

def test_torch():
    import torch
    x = torch.randn(3, 3)
    assert x.shape == (3, 3)

def test_numpy_scipy():
    import numpy as np
    from scipy.optimize import minimize
    result = minimize(lambda x: x[0]**2 + x[1]**2, [1.0, 1.0])
    assert result.success

def test_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    plt.close(fig)

# ── Computation engines ──

def test_qe():
    r = subprocess.run(["pw.x", "--version"], capture_output=True, text=True, timeout=10)
    # pw.x --version exits non-zero but prints version to stderr
    output = r.stdout + r.stderr
    assert "7." in output or "Quantum ESPRESSO" in output, f"Unexpected QE output: {output[:200]}"

def test_lammps():
    r = subprocess.run(["lmp", "-h"], capture_output=True, text=True, timeout=10)
    output = r.stdout + r.stderr
    assert "LAMMPS" in output, f"LAMMPS not found in output: {output[:200]}"

def test_raspa3():
    r = subprocess.run(["raspa3", "--help"], capture_output=True, text=True, timeout=10)
    output = r.stdout + r.stderr
    assert r.returncode == 0 or "raspa" in output.lower(), f"RASPA3 issue: {output[:200]}"

# ── Node.js agent runner ──

def test_node():
    r = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=10)
    assert r.returncode == 0
    assert r.stdout.strip().startswith("v")

# ── Run all checks ──

print("MatClaw container smoke test")
print("=" * 40)

print("\nPython packages:")
check("MACE-MP-0 (Si energy)", test_mace)
check("pymatgen (structure)", test_pymatgen)
check("ASE (bulk builder)", test_ase)
check("PyTorch (tensor ops)", test_torch)
check("numpy + scipy", test_numpy_scipy)
check("matplotlib (Agg)", test_matplotlib)

print("\nComputation engines:")
check("Quantum ESPRESSO (pw.x)", test_qe)
check("LAMMPS (lmp)", test_lammps)
check("RASPA3 (raspa3)", test_raspa3)

print("\nRuntime:")
check("Node.js", test_node)

print(f"\n{'=' * 40}")
print(f"Results: {passed} passed, {failed} failed")

if failed > 0:
    print("\nSMOKE TEST FAILED")
    sys.exit(1)
else:
    print("\nSMOKE TEST PASSED")
    sys.exit(0)
