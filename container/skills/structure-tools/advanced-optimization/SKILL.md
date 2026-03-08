# Advanced Structure Optimization

## When to Use

- Standard ASE BFGS/FIRE fails to converge or oscillates for a difficult structure.
- You need simultaneous cell + position (volume) relaxation with fine control over strain degrees of freedom.
- You are optimizing amorphous, disordered, or heavily defected structures where the potential energy surface is rugged.
- You want to compare Hessian update schemes (BFGS, SR1, PSB) for convergence behavior.
- You need scipy-based minimizers (CG, L-BFGS-B, TNC) applied to atomic structures for benchmarking or special constraints.
- The system is large (>1000 atoms) and you need memory-efficient limited-memory methods.
- You are searching for saddle points or transition states where indefinite Hessians are acceptable (SR1, PSB).

## Method Selection

| Criterion | ASE Built-in (BFGS/FIRE/LBFGS) | Custom Quasi-Newton (BFGS/SR1/PSB) | Scipy Minimizers (CG/L-BFGS-B/TNC) | QE vc-relax |
|---|---|---|---|---|
| Ease of use | Simplest, 3 lines | Moderate, explicit loop | Moderate, callback-based | QE input file |
| Hessian control | Fixed scheme per optimizer | Full control over update formula | Handled internally by scipy | N/A (BFGS inside QE) |
| Volume relaxation | Via FrechetCellFilter | Manual strain parameterization | Manual strain parameterization | Built-in (`vc-relax`) |
| Saddle point search | Not designed for this | SR1/PSB allow indefinite Hessian | Not designed for this | Not supported |
| Large systems (>1000 atoms) | LBFGS (limited memory) | Full Hessian is O(N^2) memory | L-BFGS-B is limited memory | DFT cost-prohibitive |
| DFT accuracy | MLIP only | MLIP only | MLIP only | Full DFT |
| Best for | Routine relaxations | Difficult convergence, research | Constrained optimization, benchmarking | Publication-quality DFT relaxation |

### When to Choose What

- **Routine relaxation of a crystal or molecule**: Use ASE BFGS or LBFGS with FrechetCellFilter. Simple and reliable.
- **Oscillating or slow convergence**: Switch to custom Quasi-Newton with PSB or SR1 update -- these can handle ill-conditioned Hessians.
- **Amorphous/glassy/disordered systems**: Use FIRE (good for rugged landscapes) or L-BFGS-B (memory efficient). If FIRE oscillates, try custom Quasi-Newton with eigenvalue softening.
- **Simultaneous cell + positions**: ASE FrechetCellFilter is the simplest route. For finer control over which strain components relax, use the manual strain-parameterized approach (Scripts 3-4).
- **Saddle point vicinity**: Only SR1 and PSB allow indefinite Hessians. BFGS enforces positive definiteness and will fail near saddle points.
- **DFT-level accuracy for publication**: Use QE `vc-relax` (Script 6). MLIP relaxation is fast but approximate.

## Prerequisites

- Python packages: `ase`, `numpy`, `scipy`, `matplotlib`, `mace-torch` (install via `pip install mace-torch`).
- For QE scripts: Quantum ESPRESSO installed, pseudopotentials in `./pseudo/`.

---

## Detailed Steps

### Script 1: Quasi-Newton Optimization with Hessian Updates (BFGS, SR1, PSB)

This script implements the three Hessian update formulas from scratch and runs a structure optimization loop. It compares convergence behavior across update schemes.

```python
#!/usr/bin/env python3
"""
Quasi-Newton structure optimization with explicit Hessian updates.

Implements three Hessian update schemes:
  - BFGS: Broyden-Fletcher-Goldfarb-Shanno (enforces positive definite Hessian)
  - SR1:  Symmetric Rank-1 (allows indefinite Hessian, good near saddle points)
  - PSB:  Powell-Symmetric-Broyden (allows indefinite Hessian)

Uses MACE calculator. No pyiron dependency.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.build import bulk
from ase.io import write
from mace.calculators import mace_mp


# ── Hessian update formulas ─────────────────────────────────────────
def hessian_update_BFGS(dx, dg, H):
    """
    BFGS update. Maintains positive definiteness if H starts positive definite.
    dx: change in positions (flattened)
    dg: change in gradients (flattened)
    H:  current Hessian (N x N)
    """
    Hx = H.dot(dx)
    dg_dx = dg.dot(dx)
    if abs(dg_dx) < 1e-12:
        return H  # skip update if denominator too small
    return np.outer(dg, dg) / dg_dx - np.outer(Hx, Hx) / dx.dot(Hx) + H


def hessian_update_SR1(dx, dg, H, threshold=1e-4):
    """
    Symmetric Rank-1 update. Does NOT enforce positive definiteness.
    Can capture negative curvature -- useful near saddle points.
    """
    r = dg - H.dot(dx)
    denominator = r.dot(dx)
    if abs(denominator) < threshold:
        denominator += threshold  # regularize to prevent explosion
    return np.outer(r, r) / denominator + H


def hessian_update_PSB(dx, dg, H):
    """
    Powell-Symmetric-Broyden update. Does NOT enforce positive definiteness.
    Symmetric rank-2 update that preserves symmetry of H.
    """
    r = dg - H.dot(dx)
    dxdx = dx.dot(dx)
    if abs(dxdx) < 1e-12:
        return H
    dH = np.outer(r, dx)
    dH = (dH + dH.T) / dxdx
    correction = np.outer(dx, dx) * r.dot(dx) / dxdx**2
    return dH - correction + H


# ── Quasi-Newton optimizer class ────────────────────────────────────
class QuasiNewtonOptimizer:
    """
    Quasi-Newton optimizer for atomic structures.

    Args:
        n_dof: number of degrees of freedom (3 * n_atoms)
        starting_h: initial diagonal Hessian value (eV/Ang^2)
        max_displacement: maximum allowed displacement per atom per step (Ang)
        use_eigenvalue_softening: use eigenvalue-based regularization (True)
            or Tikhonov regularization (False)
    """

    def __init__(self, n_dof, starting_h=10.0, max_displacement=0.1,
                 use_eigenvalue_softening=True):
        self.hessian = starting_h * np.eye(n_dof)
        self.max_displacement = max_displacement
        self.use_eigenvalue_softening = use_eigenvalue_softening
        self.g_old = None
        self.dx = None

    def _regularized_inverse(self, reg_param):
        """Compute regularized inverse Hessian."""
        if self.use_eigenvalue_softening:
            eigvals, eigvecs = np.linalg.eigh(self.hessian)
            # Eigenvalue softening: x = V * (d / (d^2 + lambda)) * V^T * f
            soft = eigvals / (eigvals**2 + np.exp(reg_param))
            return eigvecs @ np.diag(soft) @ eigvecs.T
        else:
            # Tikhonov: x = (H + lambda * I)^{-1} * f
            return np.linalg.inv(
                self.hessian + np.eye(len(self.hessian)) * np.exp(reg_param)
            )

    def get_step(self, gradient, mode="PSB"):
        """
        Compute displacement step given current gradient (negative forces).

        Args:
            gradient: forces with sign flip, shape (n_atoms, 3)
            mode: "BFGS", "SR1", or "PSB"

        Returns:
            displacement: shape (n_atoms, 3)
        """
        g_flat = gradient.flatten()

        # Update Hessian using previous step info
        if self.g_old is not None and self.dx is not None:
            dg = g_flat - self.g_old
            dx_flat = self.dx.flatten()
            if mode == "BFGS":
                self.hessian = hessian_update_BFGS(dx_flat, dg, self.hessian)
            elif mode == "SR1":
                self.hessian = hessian_update_SR1(dx_flat, dg, self.hessian)
            elif mode == "PSB":
                self.hessian = hessian_update_PSB(dx_flat, dg, self.hessian)
            else:
                raise ValueError(f"Unknown mode: {mode}. Use BFGS, SR1, or PSB.")

        self.g_old = g_flat.copy()

        # Compute step: dx = -H^{-1} * g
        # First try without regularization
        try:
            H_inv = np.linalg.inv(self.hessian)
        except np.linalg.LinAlgError:
            H_inv = self._regularized_inverse(0.0)

        step = -H_inv.dot(g_flat).reshape(-1, 3)

        # Check if max displacement is exceeded; if so, apply regularization
        max_disp = np.linalg.norm(step, axis=-1).max()
        if max_disp > self.max_displacement:
            reg = -2.0
            for _ in range(30):
                H_inv = self._regularized_inverse(reg)
                step = -H_inv.dot(g_flat).reshape(-1, 3)
                if np.linalg.norm(step, axis=-1).max() < self.max_displacement:
                    break
                reg += 1.0
                if reg > 20:
                    reg = 20.0
                    break

        self.dx = step.copy()
        return step


def run_quasi_newton(atoms, mode="PSB", fmax=0.01, max_steps=200,
                     starting_h=10.0, max_displacement=0.1):
    """
    Run Quasi-Newton optimization on an ASE Atoms object.

    Args:
        atoms: ASE Atoms with calculator attached
        mode: Hessian update scheme ("BFGS", "SR1", or "PSB")
        fmax: force convergence criterion (eV/Ang)
        max_steps: maximum optimization steps
        starting_h: initial Hessian diagonal value
        max_displacement: max atomic displacement per step

    Returns:
        dict with optimization history
    """
    n_atoms = len(atoms)
    n_dof = 3 * n_atoms
    optimizer = QuasiNewtonOptimizer(
        n_dof=n_dof,
        starting_h=starting_h,
        max_displacement=max_displacement,
    )

    energies = []
    max_forces = []
    converged = False

    for step in range(max_steps):
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        fmax_current = np.linalg.norm(forces, axis=-1).max()

        energies.append(energy)
        max_forces.append(fmax_current)

        if step % 10 == 0 or fmax_current < fmax:
            print(f"  Step {step:4d}: E = {energy:.6f} eV, "
                  f"Fmax = {fmax_current:.6f} eV/Ang")

        if fmax_current < fmax:
            print(f"  Converged at step {step} with Fmax = {fmax_current:.6f} eV/Ang")
            converged = True
            break

        # gradient = -forces (we minimize energy, forces point downhill)
        gradient = -forces
        dx = optimizer.get_step(gradient, mode=mode)
        atoms.positions += dx

    if not converged:
        print(f"  WARNING: Not converged after {max_steps} steps. "
              f"Fmax = {max_forces[-1]:.6f} eV/Ang")

    return {
        "energies": np.array(energies),
        "max_forces": np.array(max_forces),
        "converged": converged,
        "n_steps": len(energies),
    }


# ── Main: compare BFGS, SR1, PSB on a distorted structure ──────────
print("=" * 70)
print("Quasi-Newton optimization with Hessian updates: BFGS vs SR1 vs PSB")
print("=" * 70)

# Create a distorted FCC Al structure
atoms_ref = bulk("Al", "fcc", a=4.05, cubic=True)  # 4 atoms
# Add random displacement to make it non-trivial
np.random.seed(42)
atoms_ref.positions += np.random.randn(*atoms_ref.positions.shape) * 0.15

calc = mace_mp(model="medium", dispersion=False, device="cpu")

results = {}
for mode in ["BFGS", "SR1", "PSB"]:
    print(f"\n--- Mode: {mode} ---")
    atoms = atoms_ref.copy()
    atoms.calc = calc
    results[mode] = run_quasi_newton(
        atoms, mode=mode, fmax=0.005, max_steps=100, starting_h=10.0
    )

# ── Plot convergence comparison ─────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

colors = {"BFGS": "#1f77b4", "SR1": "#ff7f0e", "PSB": "#2ca02c"}

for mode, res in results.items():
    steps = np.arange(res["n_steps"])
    ax1.plot(steps, res["energies"] - res["energies"][-1],
             label=f"{mode} ({res['n_steps']} steps)", color=colors[mode], linewidth=1.5)
    ax2.semilogy(steps, res["max_forces"],
                 label=f"{mode} ({res['n_steps']} steps)", color=colors[mode], linewidth=1.5)

ax1.set_xlabel("Step", fontsize=12)
ax1.set_ylabel("Energy - E_final (eV)", fontsize=12)
ax1.set_title("Energy Convergence", fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

ax2.set_xlabel("Step", fontsize=12)
ax2.set_ylabel("Max Force (eV/Ang)", fontsize=12)
ax2.set_title("Force Convergence", fontsize=13)
ax2.axhline(0.005, color="red", linestyle="--", linewidth=0.8, label="Tolerance")
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.suptitle("Quasi-Newton: Hessian Update Scheme Comparison", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("quasi_newton_comparison.png", dpi=200, bbox_inches="tight")
print("\nPlot saved: quasi_newton_comparison.png")
```

### Script 2: Eigenvalue Softening vs Tikhonov Regularization

Demonstrates the two regularization strategies that prevent unphysically large displacements when the Hessian has small or negative eigenvalues.

```python
#!/usr/bin/env python3
"""
Compare eigenvalue softening vs Tikhonov regularization for
Quasi-Newton structure optimization.

Eigenvalue softening:
    x = V * diag(d / (d^2 + lambda)) * V^T * f

Tikhonov regularization:
    x = (H + lambda * I)^{-1} * f

Eigenvalue softening handles negative eigenvalues more gracefully,
making it suitable for saddle-point searches.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.build import bulk, make_supercell
from ase.io import write
from mace.calculators import mace_mp


class QuasiNewtonWithRegularization:
    """Quasi-Newton with selectable regularization strategy."""

    def __init__(self, n_dof, starting_h=10.0, max_displacement=0.1,
                 regularization="eigenvalue"):
        """
        regularization: "eigenvalue" for eigenvalue softening,
                        "tikhonov" for Tikhonov regularization
        """
        self.hessian = starting_h * np.eye(n_dof)
        self.max_displacement = max_displacement
        self.regularization = regularization
        self.g_old = None
        self.dx = None
        self.reg_values = []  # track regularization parameter over steps

    def _get_inv_hessian(self, reg_param):
        if self.regularization == "eigenvalue":
            eigvals, eigvecs = np.linalg.eigh(self.hessian)
            soft = eigvals / (eigvals**2 + np.exp(reg_param))
            return eigvecs @ np.diag(soft) @ eigvecs.T
        else:
            return np.linalg.inv(
                self.hessian + np.eye(len(self.hessian)) * np.exp(reg_param)
            )

    def get_step(self, forces, mode="PSB"):
        g = -forces.flatten()

        # Update Hessian
        if self.g_old is not None and self.dx is not None:
            dg = g - self.g_old
            dx_flat = self.dx.flatten()
            r = dg - self.hessian.dot(dx_flat)
            dxdx = dx_flat.dot(dx_flat)
            if dxdx > 1e-12:
                if mode == "PSB":
                    dH = np.outer(r, dx_flat)
                    dH = (dH + dH.T) / dxdx
                    self.hessian = dH - np.outer(dx_flat, dx_flat) * r.dot(dx_flat) / dxdx**2 + self.hessian
                elif mode == "BFGS":
                    Hx = self.hessian.dot(dx_flat)
                    dg_dx = dg.dot(dx_flat)
                    if abs(dg_dx) > 1e-12:
                        self.hessian = (np.outer(dg, dg) / dg_dx
                                        - np.outer(Hx, Hx) / dx_flat.dot(Hx)
                                        + self.hessian)

        self.g_old = g.copy()

        # Compute step with regularization if needed
        try:
            H_inv = np.linalg.inv(self.hessian)
            step = -H_inv.dot(g).reshape(-1, 3)
            reg_used = 0.0
        except np.linalg.LinAlgError:
            step = np.zeros_like(forces)
            reg_used = 0.0

        if np.linalg.norm(step, axis=-1).max() > self.max_displacement:
            reg = -2.0
            for _ in range(30):
                H_inv = self._get_inv_hessian(reg)
                step = -H_inv.dot(g).reshape(-1, 3)
                if np.linalg.norm(step, axis=-1).max() < self.max_displacement:
                    break
                reg += 1.0
            reg_used = reg

        self.reg_values.append(reg_used)
        self.dx = step.copy()
        return step


def optimize_with_regularization(atoms, reg_type, fmax=0.005, max_steps=150):
    """Run optimization and return history."""
    n_dof = 3 * len(atoms)
    opt = QuasiNewtonWithRegularization(
        n_dof=n_dof, starting_h=10.0, max_displacement=0.1,
        regularization=reg_type,
    )
    energies, max_forces = [], []
    for step in range(max_steps):
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        fm = np.linalg.norm(f, axis=-1).max()
        energies.append(e)
        max_forces.append(fm)
        if fm < fmax:
            break
        dx = opt.get_step(f, mode="PSB")
        atoms.positions += dx
    return {
        "energies": np.array(energies),
        "max_forces": np.array(max_forces),
        "reg_values": np.array(opt.reg_values),
        "n_steps": len(energies),
    }


# ── Main ────────────────────────────────────────────────────────────
print("=" * 70)
print("Eigenvalue Softening vs Tikhonov Regularization")
print("=" * 70)

atoms_ref = bulk("Cu", "fcc", a=3.61, cubic=True)
np.random.seed(123)
atoms_ref.positions += np.random.randn(*atoms_ref.positions.shape) * 0.2

calc = mace_mp(model="medium", dispersion=False, device="cpu")

results = {}
for reg_type in ["eigenvalue", "tikhonov"]:
    print(f"\n--- Regularization: {reg_type} ---")
    atoms = atoms_ref.copy()
    atoms.calc = calc
    results[reg_type] = optimize_with_regularization(atoms, reg_type, fmax=0.005)
    print(f"  Converged in {results[reg_type]['n_steps']} steps")

# ── Plot ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

labels = {"eigenvalue": "Eigenvalue Softening", "tikhonov": "Tikhonov"}
colors = {"eigenvalue": "#1f77b4", "tikhonov": "#ff7f0e"}

for reg_type, res in results.items():
    steps = np.arange(res["n_steps"])
    axes[0].plot(steps, res["energies"] - res["energies"][-1],
                 label=labels[reg_type], color=colors[reg_type], linewidth=1.5)
    axes[1].semilogy(steps, res["max_forces"],
                     label=labels[reg_type], color=colors[reg_type], linewidth=1.5)
    axes[2].plot(np.arange(len(res["reg_values"])), res["reg_values"],
                 label=labels[reg_type], color=colors[reg_type], linewidth=1.5)

axes[0].set_xlabel("Step")
axes[0].set_ylabel("E - E_final (eV)")
axes[0].set_title("Energy Convergence")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_xlabel("Step")
axes[1].set_ylabel("Max Force (eV/Ang)")
axes[1].set_title("Force Convergence")
axes[1].axhline(0.005, color="red", linestyle="--", linewidth=0.8, label="Tolerance")
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].set_xlabel("Step")
axes[2].set_ylabel("Regularization Parameter")
axes[2].set_title("Regularization History")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.suptitle("Regularization Strategy Comparison (PSB Hessian Update)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("regularization_comparison.png", dpi=200, bbox_inches="tight")
print("\nPlot saved: regularization_comparison.png")
```

### Script 3: Simultaneous Cell + Position Optimization (Volume Relaxation)

Implements volume relaxation via strain parameterization, optimizing both atomic positions and cell shape simultaneously. This is the approach used by pyiron's ScipyMinimizer for pressure-targeted relaxation.

```python
#!/usr/bin/env python3
"""
Simultaneous cell + position optimization via strain parameterization.

The cell is parameterized via Voigt strain components:
    cell_new = (I + epsilon) @ cell_original
where epsilon is the symmetric strain tensor built from up to 6 Voigt
components (eps_xx, eps_yy, eps_zz, eps_yz, eps_xz, eps_xy).

Positions are expressed in fractional (scaled) coordinates so that
cell changes automatically move atoms proportionally.

Target pressure can be specified. Volume-only mode keeps positions fixed
and only relaxes cell shape/volume.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scipy.constants
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.build import bulk
from ase.io import write, Trajectory
from mace.calculators import mace_mp

GPa_to_eV_per_A3 = (
    1e21 / scipy.constants.physical_constants["joule-electron volt relationship"][0]
)


def voigt_to_tensor(voigt, is_strain=False):
    """Convert 6-component Voigt vector to 3x3 symmetric tensor."""
    v = np.array(voigt, dtype=float).copy()
    if not is_strain:
        v[:3] /= 2
    tensor = np.array([
        [v[0], v[5], v[4]],
        [0.0,  v[1], v[3]],
        [0.0,  0.0,  v[2]],
    ])
    tensor = tensor + tensor.T
    return tensor


def tensor_to_voigt(tensor, is_strain=False):
    """Convert 3x3 symmetric tensor to 6-component Voigt vector."""
    s = 0.5 * (tensor + tensor.T)
    voigt = np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
    if is_strain:
        voigt[3:] *= 2
    return voigt


class CellPositionOptimizer:
    """
    Optimizer for simultaneous cell + position relaxation.

    Args:
        atoms: ASE Atoms with calculator
        target_pressure: target pressure in GPa (scalar for hydrostatic,
            or 6-component Voigt for anisotropic)
        volume_only: if True, only relax cell, keep fractional coords fixed
        fmax: force convergence (eV/Ang)
        pressure_tol: pressure convergence (GPa)
        max_steps: maximum optimization steps
    """

    def __init__(self, atoms, target_pressure=0.0, volume_only=False,
                 fmax=0.01, pressure_tol=0.5, max_steps=200):
        self.atoms = atoms
        self.original_cell = atoms.cell.array.copy()
        self.current_strain = np.zeros(6)
        self.volume_only = volume_only
        self.fmax = fmax
        self.pressure_tol = pressure_tol  # GPa
        self.max_steps = max_steps

        # Handle pressure specification
        if np.isscalar(target_pressure):
            # Hydrostatic: only optimize volumetric strain
            self.target_pressure = np.array([target_pressure])
            self.n_strain_dof = 1  # single volumetric parameter
        else:
            self.target_pressure = np.array(target_pressure).flatten()
            if len(self.target_pressure) == 6:
                self.n_strain_dof = 6
            elif len(self.target_pressure) == 3:
                self.target_pressure = np.append(self.target_pressure, [0, 0, 0])
                self.n_strain_dof = 6
            else:
                raise ValueError("Pressure must be scalar, 3-, or 6-component")

    def _apply_strain(self, strain_voigt):
        """Apply strain to the original cell."""
        if self.n_strain_dof == 1:
            # Hydrostatic: apply isotropic strain
            eps = np.zeros(6)
            eps[:3] = strain_voigt[0]
        else:
            eps = strain_voigt.copy()
        eps_tensor = voigt_to_tensor(eps, is_strain=True)
        new_cell = (np.eye(3) + eps_tensor) @ self.original_cell
        self.atoms.set_cell(new_cell, scale_atoms=True)

    def _get_pressure(self):
        """Get current stress as pressure in GPa."""
        # ASE stress: [xx, yy, zz, yz, xz, xy] in eV/Ang^3, positive = compressive
        stress_voigt = self.atoms.get_stress()  # Voigt, eV/Ang^3
        pressure_GPa = -stress_voigt / GPa_to_eV_per_A3  # convert and flip sign
        if self.n_strain_dof == 1:
            return np.array([np.mean(pressure_GPa[:3])])
        return pressure_GPa[:self.n_strain_dof]

    def run(self):
        """Run the optimization."""
        energies = []
        max_forces_history = []
        pressures_history = []
        volumes = []

        strain_x = np.zeros(self.n_strain_dof)

        print(f"  Starting cell+position optimization")
        print(f"  Target pressure: {self.target_pressure} GPa")
        print(f"  Volume only: {self.volume_only}")
        print(f"  Strain DOFs: {self.n_strain_dof}")

        for step in range(self.max_steps):
            # Apply current strain
            self._apply_strain(strain_x)

            # Evaluate
            energy = self.atoms.get_potential_energy()
            forces = self.atoms.get_forces()
            pressure = self._get_pressure()
            fmax_val = np.linalg.norm(forces, axis=-1).max()
            volume = self.atoms.get_volume()

            energies.append(energy)
            max_forces_history.append(fmax_val)
            pressures_history.append(pressure.copy())
            volumes.append(volume)

            if step % 10 == 0:
                p_str = f"{pressure[0]:.3f}" if len(pressure) == 1 else str(np.round(pressure, 3))
                print(f"  Step {step:4d}: E={energy:.6f} eV, Fmax={fmax_val:.4f} eV/A, "
                      f"P={p_str} GPa, V={volume:.2f} A^3")

            # Check convergence
            pressure_converged = np.all(
                np.abs(pressure - self.target_pressure[:len(pressure)]) < self.pressure_tol
            )
            force_converged = self.volume_only or fmax_val < self.fmax

            if pressure_converged and force_converged and step > 0:
                print(f"  Converged at step {step}")
                break

            # Update strain using pressure gradient (steepest descent on strain)
            dp = pressure - self.target_pressure[:len(pressure)]
            strain_step = 0.002  # small step for stability
            strain_x -= strain_step * dp

            # Update positions using force-based steepest descent (in fractional coords)
            if not self.volume_only:
                # Convert forces to fractional coordinate forces
                cell_inv = np.linalg.inv(self.atoms.cell.array)
                frac_forces = forces @ cell_inv.T
                # Simple steepest descent on positions
                scaled_pos = self.atoms.get_scaled_positions()
                scaled_pos += 0.01 * frac_forces
                self.atoms.set_scaled_positions(scaled_pos)

        return {
            "energies": np.array(energies),
            "max_forces": np.array(max_forces_history),
            "pressures": np.array(pressures_history),
            "volumes": np.array(volumes),
            "n_steps": len(energies),
        }


# ── Main ────────────────────────────────────────────────────────────
print("=" * 70)
print("Simultaneous Cell + Position Optimization (Volume Relaxation)")
print("=" * 70)

# Create Al with slightly wrong lattice constant
atoms = bulk("Al", "fcc", a=4.20, cubic=True)  # 4 atoms, a is too large
np.random.seed(42)
atoms.positions += np.random.randn(*atoms.positions.shape) * 0.05

calc = mace_mp(model="medium", dispersion=False, device="cpu")
atoms.calc = calc

print(f"\nInitial volume: {atoms.get_volume():.2f} A^3")
print(f"Initial lattice parameter: {atoms.cell.cellpar()[0]:.4f} A")

# Volume-only optimization (fix positions, relax cell)
print("\n--- Volume-only relaxation ---")
atoms_vol = atoms.copy()
atoms_vol.calc = calc
opt_vol = CellPositionOptimizer(
    atoms_vol, target_pressure=0.0, volume_only=True,
    pressure_tol=0.1, max_steps=100,
)
res_vol = opt_vol.run()

# Full cell + position optimization
print("\n--- Full cell + position relaxation ---")
atoms_full = atoms.copy()
atoms_full.calc = calc
opt_full = CellPositionOptimizer(
    atoms_full, target_pressure=0.0, volume_only=False,
    fmax=0.005, pressure_tol=0.1, max_steps=200,
)
res_full = opt_full.run()

# ── Plot results ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Energy
for label, res, c in [("Volume only", res_vol, "#1f77b4"),
                       ("Cell + positions", res_full, "#ff7f0e")]:
    s = np.arange(res["n_steps"])
    axes[0, 0].plot(s, res["energies"], label=label, color=c, linewidth=1.5)
axes[0, 0].set_xlabel("Step")
axes[0, 0].set_ylabel("Energy (eV)")
axes[0, 0].set_title("Energy vs Step")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Max force
for label, res, c in [("Volume only", res_vol, "#1f77b4"),
                       ("Cell + positions", res_full, "#ff7f0e")]:
    s = np.arange(res["n_steps"])
    axes[0, 1].semilogy(s, res["max_forces"], label=label, color=c, linewidth=1.5)
axes[0, 1].set_xlabel("Step")
axes[0, 1].set_ylabel("Max Force (eV/Ang)")
axes[0, 1].set_title("Force Convergence")
axes[0, 1].axhline(0.005, color="red", linestyle="--", linewidth=0.8, label="Tolerance")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Pressure
for label, res, c in [("Volume only", res_vol, "#1f77b4"),
                       ("Cell + positions", res_full, "#ff7f0e")]:
    s = np.arange(res["n_steps"])
    axes[1, 0].plot(s, res["pressures"][:, 0], label=label, color=c, linewidth=1.5)
axes[1, 0].set_xlabel("Step")
axes[1, 0].set_ylabel("Pressure (GPa)")
axes[1, 0].set_title("Pressure vs Step")
axes[1, 0].axhline(0, color="red", linestyle="--", linewidth=0.8)
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Volume
for label, res, c in [("Volume only", res_vol, "#1f77b4"),
                       ("Cell + positions", res_full, "#ff7f0e")]:
    s = np.arange(res["n_steps"])
    axes[1, 1].plot(s, res["volumes"], label=label, color=c, linewidth=1.5)
axes[1, 1].set_xlabel("Step")
axes[1, 1].set_ylabel("Volume (A^3)")
axes[1, 1].set_title("Volume vs Step")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.suptitle("Cell + Position Optimization: Al (fcc)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("cell_position_optimization.png", dpi=200, bbox_inches="tight")
print("\nPlot saved: cell_position_optimization.png")

print(f"\nFinal results:")
print(f"  Volume-only:      V = {res_vol['volumes'][-1]:.2f} A^3, "
      f"a = {(res_vol['volumes'][-1])**(1/3) * (4/4)**(1/3):.4f} A (approx)")
print(f"  Cell + positions: V = {res_full['volumes'][-1]:.2f} A^3, "
      f"a = {(res_full['volumes'][-1])**(1/3) * (4/4)**(1/3):.4f} A (approx)")
```

### Script 4: Scipy Minimizers Applied to Atomic Structures (CG, L-BFGS-B, TNC)

Uses `scipy.optimize.minimize` to drive atomic structure optimization. The ASE calculator provides energy and forces; scipy handles the optimization algorithm.

```python
#!/usr/bin/env python3
"""
Scipy-based minimizers applied to atomic structure optimization.

Wraps ASE calculators with scipy.optimize.minimize to use:
  - CG (Conjugate Gradient): good general-purpose, moderate memory
  - L-BFGS-B (Limited-memory BFGS with bounds): memory efficient for large systems
  - TNC (Truncated Newton Conjugate): good for constrained problems

The atomic positions (and optionally strain DOFs) are the optimization
variables. Energy is the objective; negative forces are the gradient.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.build import bulk, make_supercell
from ase.io import write
from mace.calculators import mace_mp


class ScipyAtomisticMinimizer:
    """
    Wrapper that uses scipy.optimize.minimize on an ASE Atoms object.

    The optimization variables are the flattened positions (3*N values).
    Energy is the objective function; negative forces are the Jacobian.
    """

    def __init__(self, atoms, method="CG", fmax=0.01, max_steps=200):
        self.atoms = atoms
        self.method = method
        self.fmax = fmax
        self.max_steps = max_steps
        self.history = {"energies": [], "max_forces": [], "positions": []}
        self._n_calls = 0

    def _objective(self, x):
        """Objective function: returns energy."""
        self.atoms.positions = x.reshape(-1, 3)
        energy = self.atoms.get_potential_energy()
        forces = self.atoms.get_forces()
        fmax_val = np.linalg.norm(forces, axis=-1).max()

        self.history["energies"].append(energy)
        self.history["max_forces"].append(fmax_val)
        self.history["positions"].append(x.copy())
        self._n_calls += 1

        if self._n_calls % 20 == 0:
            print(f"    Call {self._n_calls:4d}: E = {energy:.6f} eV, "
                  f"Fmax = {fmax_val:.6f} eV/Ang")

        return energy

    def _jacobian(self, x):
        """Jacobian: returns negative forces (gradient of energy)."""
        self.atoms.positions = x.reshape(-1, 3)
        forces = self.atoms.get_forces()
        return -forces.flatten()

    def run(self):
        """Run scipy minimization."""
        x0 = self.atoms.positions.flatten()

        print(f"  Running scipy.optimize.minimize with method={self.method}")
        print(f"  DOFs: {len(x0)}, max_steps: {self.max_steps}")

        result = minimize(
            fun=self._objective,
            x0=x0,
            jac=self._jacobian,
            method=self.method,
            options={"maxiter": self.max_steps, "disp": False},
            tol=1e-10,  # use fmax for convergence, not scipy tol
        )

        # Set final positions
        self.atoms.positions = result.x.reshape(-1, 3)

        return {
            "energies": np.array(self.history["energies"]),
            "max_forces": np.array(self.history["max_forces"]),
            "n_calls": self._n_calls,
            "scipy_success": result.success,
            "scipy_message": result.message,
        }


class ScipyCellMinimizer:
    """
    Scipy-based simultaneous cell + position optimizer.

    Optimization variables: [strain_xx, strain_yy, strain_zz, scaled_positions...]
    Uses hydrostatic strain for cell and scaled positions for atoms.
    """

    def __init__(self, atoms, method="L-BFGS-B", target_pressure_GPa=0.0,
                 fmax=0.01, max_steps=300):
        self.atoms = atoms
        self.method = method
        self.target_pressure = target_pressure_GPa
        self.original_cell = atoms.cell.array.copy()
        self.fmax = fmax
        self.max_steps = max_steps
        self.history = {"energies": [], "max_forces": [], "pressures": [], "volumes": []}
        self._n_calls = 0

        import scipy.constants
        self._GPa_to_eV_A3 = (
            1e21 / scipy.constants.physical_constants[
                "joule-electron volt relationship"
            ][0]
        )

    def _objective(self, x):
        """x = [eps_xx, eps_yy, eps_zz, scaled_pos_flat...]"""
        eps = x[:3]
        strain_tensor = np.diag(1.0 + eps)
        new_cell = strain_tensor @ self.original_cell
        self.atoms.set_cell(new_cell, scale_atoms=True)

        scaled_pos = x[3:].reshape(-1, 3)
        self.atoms.set_scaled_positions(scaled_pos)

        energy = self.atoms.get_potential_energy()
        forces = self.atoms.get_forces()
        fmax_val = np.linalg.norm(forces, axis=-1).max()

        stress = self.atoms.get_stress()[:3]  # xx, yy, zz only
        pressure_GPa = -np.mean(stress) / self._GPa_to_eV_A3
        volume = self.atoms.get_volume()

        self.history["energies"].append(energy)
        self.history["max_forces"].append(fmax_val)
        self.history["pressures"].append(pressure_GPa)
        self.history["volumes"].append(volume)
        self._n_calls += 1

        if self._n_calls % 20 == 0:
            print(f"    Call {self._n_calls:4d}: E={energy:.6f} eV, "
                  f"Fmax={fmax_val:.4f}, P={pressure_GPa:.2f} GPa, V={volume:.2f} A^3")

        return energy

    def _jacobian(self, x):
        """Gradient w.r.t. [strain, scaled_positions]."""
        eps = x[:3]
        strain_tensor = np.diag(1.0 + eps)
        new_cell = strain_tensor @ self.original_cell
        self.atoms.set_cell(new_cell, scale_atoms=True)
        self.atoms.set_scaled_positions(x[3:].reshape(-1, 3))

        # Strain gradient: -volume * (stress - target_stress)
        stress = self.atoms.get_stress()[:3]
        target_stress = -self.target_pressure * self._GPa_to_eV_A3
        volume = self.atoms.get_volume()
        strain_grad = volume * (stress - target_stress) * 0.01  # scale factor

        # Position gradient: negative forces in fractional coords
        forces = self.atoms.get_forces()
        cell_inv = np.linalg.inv(self.atoms.cell.array)
        frac_forces = forces @ cell_inv.T
        pos_grad = -frac_forces.flatten()

        return np.concatenate([strain_grad, pos_grad])

    def run(self):
        """Run scipy minimization."""
        scaled_pos = self.atoms.get_scaled_positions()
        x0 = np.concatenate([np.zeros(3), scaled_pos.flatten()])

        print(f"  Running scipy cell+position optimization: method={self.method}")
        print(f"  DOFs: 3 strain + {3 * len(self.atoms)} positions = {len(x0)}")

        result = minimize(
            fun=self._objective,
            x0=x0,
            jac=self._jacobian,
            method=self.method,
            options={"maxiter": self.max_steps, "disp": False},
        )

        return {
            "energies": np.array(self.history["energies"]),
            "max_forces": np.array(self.history["max_forces"]),
            "pressures": np.array(self.history["pressures"]),
            "volumes": np.array(self.history["volumes"]),
            "n_calls": self._n_calls,
            "scipy_success": result.success,
        }


# ── Main: compare CG, L-BFGS-B, TNC for position optimization ─────
print("=" * 70)
print("Scipy Minimizers for Atomic Structure Optimization")
print("=" * 70)

# Build a distorted Al supercell
atoms_ref = bulk("Al", "fcc", a=4.05) * (2, 2, 2)  # 32 atoms
np.random.seed(42)
atoms_ref.positions += np.random.randn(*atoms_ref.positions.shape) * 0.15

calc = mace_mp(model="medium", dispersion=False, device="cpu")

print(f"\nSystem: Al fcc 2x2x2 supercell ({len(atoms_ref)} atoms)")
print(f"DOFs: {3 * len(atoms_ref)}")

# Compare methods for position-only optimization
methods = ["CG", "L-BFGS-B", "TNC"]
results = {}

for method in methods:
    print(f"\n--- Method: {method} ---")
    atoms = atoms_ref.copy()
    atoms.calc = calc
    opt = ScipyAtomisticMinimizer(atoms, method=method, fmax=0.005, max_steps=300)
    results[method] = opt.run()
    print(f"  Calls: {results[method]['n_calls']}, "
          f"Success: {results[method]['scipy_success']}, "
          f"Message: {results[method]['scipy_message']}")

# ── Cell + position optimization with L-BFGS-B ─────────────────────
print(f"\n--- Cell + Position optimization (L-BFGS-B) ---")
atoms_cell = bulk("Al", "fcc", a=4.20) * (2, 2, 2)  # wrong lattice param
np.random.seed(42)
atoms_cell.positions += np.random.randn(*atoms_cell.positions.shape) * 0.05
atoms_cell.calc = calc

cell_opt = ScipyCellMinimizer(
    atoms_cell, method="L-BFGS-B", target_pressure_GPa=0.0,
    fmax=0.005, max_steps=300,
)
res_cell = cell_opt.run()

# ── Plot ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

colors = {"CG": "#1f77b4", "L-BFGS-B": "#ff7f0e", "TNC": "#2ca02c"}

# Position optimization comparison
for method, res in results.items():
    n = res["n_calls"]
    axes[0, 0].plot(np.arange(n), res["energies"] - res["energies"][-1],
                    label=f"{method} ({n} calls)", color=colors[method], linewidth=1.5)
    axes[0, 1].semilogy(np.arange(n), res["max_forces"],
                        label=f"{method} ({n} calls)", color=colors[method], linewidth=1.5)

axes[0, 0].set_xlabel("Function Calls")
axes[0, 0].set_ylabel("E - E_final (eV)")
axes[0, 0].set_title("Position Optimization: Energy")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].set_xlabel("Function Calls")
axes[0, 1].set_ylabel("Max Force (eV/Ang)")
axes[0, 1].set_title("Position Optimization: Forces")
axes[0, 1].axhline(0.005, color="red", linestyle="--", linewidth=0.8, label="Tolerance")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Cell + position optimization
n_cell = res_cell["n_calls"]
axes[1, 0].plot(np.arange(n_cell), res_cell["volumes"], color="#d62728", linewidth=1.5)
axes[1, 0].set_xlabel("Function Calls")
axes[1, 0].set_ylabel("Volume (A^3)")
axes[1, 0].set_title("Cell+Position Opt: Volume")
axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(np.arange(n_cell), res_cell["pressures"], color="#9467bd", linewidth=1.5)
axes[1, 1].axhline(0, color="red", linestyle="--", linewidth=0.8, label="Target")
axes[1, 1].set_xlabel("Function Calls")
axes[1, 1].set_ylabel("Pressure (GPa)")
axes[1, 1].set_title("Cell+Position Opt: Pressure")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.suptitle("Scipy Minimizers for Atomic Structure Optimization", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("scipy_minimizers.png", dpi=200, bbox_inches="tight")
print("\nPlot saved: scipy_minimizers.png")
```

### Script 5: ASE Built-in vs Advanced Methods -- Full Benchmark

Head-to-head comparison of ASE BFGS, ASE FIRE, ASE LBFGS, custom Quasi-Newton (PSB), and scipy L-BFGS-B on an amorphous structure.

```python
#!/usr/bin/env python3
"""
Full benchmark: ASE built-in optimizers vs advanced methods.

Tests on a deliberately difficult structure (distorted + large displacement)
to show when advanced methods outperform standard ones.

Optimizers tested:
  1. ASE BFGS
  2. ASE FIRE
  3. ASE LBFGS
  4. Custom Quasi-Newton (PSB with eigenvalue softening)
  5. Scipy L-BFGS-B

Metrics: number of steps, number of force calls, wall time, final energy.
"""
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.build import bulk, make_supercell
from ase.optimize import BFGS, FIRE, LBFGS
from ase.io import write
from mace.calculators import mace_mp
from scipy.optimize import minimize


# ── Reuse Quasi-Newton from Script 1 ───────────────────────────────
def hessian_update_PSB(dx, dg, H):
    r = dg - H.dot(dx)
    dxdx = dx.dot(dx)
    if abs(dxdx) < 1e-12:
        return H
    dH = np.outer(r, dx)
    dH = (dH + dH.T) / dxdx
    return dH - np.outer(dx, dx) * r.dot(dx) / dxdx**2 + H


class QuasiNewtonPSB:
    def __init__(self, n_dof, starting_h=10.0, max_disp=0.1):
        self.H = starting_h * np.eye(n_dof)
        self.max_disp = max_disp
        self.g_old = None
        self.dx = None

    def step(self, forces):
        g = -forces.flatten()
        if self.g_old is not None and self.dx is not None:
            dg = g - self.g_old
            dx_f = self.dx.flatten()
            self.H = hessian_update_PSB(dx_f, dg, self.H)
        self.g_old = g.copy()

        eigvals, eigvecs = np.linalg.eigh(self.H)
        reg = -2.0
        for _ in range(30):
            soft = eigvals / (eigvals**2 + np.exp(reg))
            H_inv = eigvecs @ np.diag(soft) @ eigvecs.T
            s = -H_inv.dot(g).reshape(-1, 3)
            if np.linalg.norm(s, axis=-1).max() < self.max_disp:
                break
            reg += 1.0

        self.dx = s.copy()
        return s


def run_custom_qn(atoms, fmax=0.005, max_steps=300):
    n_dof = 3 * len(atoms)
    qn = QuasiNewtonPSB(n_dof)
    energies, forces_hist = [], []
    for i in range(max_steps):
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        fm = np.linalg.norm(f, axis=-1).max()
        energies.append(e)
        forces_hist.append(fm)
        if fm < fmax:
            break
        dx = qn.step(f)
        atoms.positions += dx
    return np.array(energies), np.array(forces_hist)


def run_scipy_lbfgsb(atoms, fmax=0.005, max_steps=300):
    energies, forces_hist = [], []

    def obj(x):
        atoms.positions = x.reshape(-1, 3)
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        fm = np.linalg.norm(f, axis=-1).max()
        energies.append(e)
        forces_hist.append(fm)
        return e

    def jac(x):
        atoms.positions = x.reshape(-1, 3)
        return -atoms.get_forces().flatten()

    minimize(fun=obj, x0=atoms.positions.flatten(), jac=jac,
             method="L-BFGS-B", options={"maxiter": max_steps})
    return np.array(energies), np.array(forces_hist)


# ── Build test structure ────────────────────────────────────────────
print("=" * 70)
print("Benchmark: ASE Built-in vs Advanced Optimizers")
print("=" * 70)

atoms_ref = bulk("Cu", "fcc", a=3.61) * (3, 3, 3)  # 108 atoms
np.random.seed(99)
atoms_ref.positions += np.random.randn(*atoms_ref.positions.shape) * 0.25

calc = mace_mp(model="medium", dispersion=False, device="cpu")

print(f"System: Cu fcc 3x3x3 supercell ({len(atoms_ref)} atoms)")
print(f"Random displacement: 0.25 Ang RMS")

FMAX = 0.01
results = {}

# ASE BFGS
print("\n--- ASE BFGS ---")
atoms = atoms_ref.copy()
atoms.calc = calc
t0 = time.time()
opt = BFGS(atoms, logfile=None)
energies_bfgs, forces_bfgs = [], []

class Callback:
    def __init__(self, atoms, elist, flist):
        self.atoms = atoms
        self.elist = elist
        self.flist = flist
    def __call__(self):
        self.elist.append(self.atoms.get_potential_energy())
        self.flist.append(np.linalg.norm(self.atoms.get_forces(), axis=-1).max())

cb_bfgs = Callback(atoms, energies_bfgs, forces_bfgs)
opt.attach(cb_bfgs)
opt.run(fmax=FMAX, steps=300)
t_bfgs = time.time() - t0
results["ASE BFGS"] = {
    "energies": np.array(energies_bfgs),
    "forces": np.array(forces_bfgs),
    "time": t_bfgs,
}
print(f"  Steps: {len(energies_bfgs)}, Time: {t_bfgs:.1f} s")

# ASE FIRE
print("\n--- ASE FIRE ---")
atoms = atoms_ref.copy()
atoms.calc = calc
t0 = time.time()
opt = FIRE(atoms, logfile=None)
energies_fire, forces_fire = [], []
cb_fire = Callback(atoms, energies_fire, forces_fire)
opt.attach(cb_fire)
opt.run(fmax=FMAX, steps=300)
t_fire = time.time() - t0
results["ASE FIRE"] = {
    "energies": np.array(energies_fire),
    "forces": np.array(forces_fire),
    "time": t_fire,
}
print(f"  Steps: {len(energies_fire)}, Time: {t_fire:.1f} s")

# ASE LBFGS
print("\n--- ASE LBFGS ---")
atoms = atoms_ref.copy()
atoms.calc = calc
t0 = time.time()
opt = LBFGS(atoms, logfile=None)
energies_lbfgs, forces_lbfgs = [], []
cb_lbfgs = Callback(atoms, energies_lbfgs, forces_lbfgs)
opt.attach(cb_lbfgs)
opt.run(fmax=FMAX, steps=300)
t_lbfgs = time.time() - t0
results["ASE LBFGS"] = {
    "energies": np.array(energies_lbfgs),
    "forces": np.array(forces_lbfgs),
    "time": t_lbfgs,
}
print(f"  Steps: {len(energies_lbfgs)}, Time: {t_lbfgs:.1f} s")

# Custom Quasi-Newton PSB
print("\n--- Custom QN (PSB) ---")
atoms = atoms_ref.copy()
atoms.calc = calc
t0 = time.time()
e_qn, f_qn = run_custom_qn(atoms, fmax=FMAX, max_steps=300)
t_qn = time.time() - t0
results["QN-PSB"] = {"energies": e_qn, "forces": f_qn, "time": t_qn}
print(f"  Steps: {len(e_qn)}, Time: {t_qn:.1f} s")

# Scipy L-BFGS-B
print("\n--- Scipy L-BFGS-B ---")
atoms = atoms_ref.copy()
atoms.calc = calc
t0 = time.time()
e_sp, f_sp = run_scipy_lbfgsb(atoms, fmax=FMAX, max_steps=300)
t_sp = time.time() - t0
results["Scipy L-BFGS-B"] = {"energies": e_sp, "forces": f_sp, "time": t_sp}
print(f"  Steps: {len(e_sp)}, Time: {t_sp:.1f} s")

# ── Summary table ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'Method':<20} {'Steps':>8} {'Final Fmax':>12} {'Time (s)':>10}")
print("-" * 50)
for name, r in results.items():
    n = len(r["forces"])
    fm = r["forces"][-1] if len(r["forces"]) > 0 else float("nan")
    print(f"{name:<20} {n:>8d} {fm:>12.6f} {r['time']:>10.1f}")

# ── Plot ────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

colors_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
for i, (name, r) in enumerate(results.items()):
    n = len(r["forces"])
    steps = np.arange(n)
    ax1.plot(steps, r["energies"] - r["energies"][-1],
             label=f"{name} ({n})", color=colors_list[i], linewidth=1.5)
    ax2.semilogy(steps, r["forces"],
                 label=f"{name} ({n})", color=colors_list[i], linewidth=1.5)

ax1.set_xlabel("Step / Function Call", fontsize=12)
ax1.set_ylabel("E - E_final (eV)", fontsize=12)
ax1.set_title("Energy Convergence", fontsize=13)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

ax2.set_xlabel("Step / Function Call", fontsize=12)
ax2.set_ylabel("Max Force (eV/Ang)", fontsize=12)
ax2.set_title("Force Convergence", fontsize=13)
ax2.axhline(FMAX, color="red", linestyle="--", linewidth=0.8, label="Tolerance")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.suptitle(f"Optimizer Benchmark: Cu fcc 3x3x3 ({len(atoms_ref)} atoms)",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("optimizer_benchmark.png", dpi=200, bbox_inches="tight")
print("\nPlot saved: optimizer_benchmark.png")
```

### Script 6: QE Variable-Cell Relaxation (vc-relax) for DFT Accuracy

When MLIP-level accuracy is insufficient, use Quantum ESPRESSO `vc-relax` for publication-quality simultaneous cell + position relaxation.

```python
#!/usr/bin/env python3
"""
QE variable-cell relaxation (vc-relax) for DFT-level accuracy.

This performs simultaneous cell + position optimization within QE's
own BFGS optimizer. Use this when MACE/MLIP accuracy is not sufficient
(e.g., for computing equations of state, elastic constants, or
structures where the MLIP training set may be sparse).

After QE vc-relax, the script extracts final structure, energy,
pressure, and convergence info.
"""
import os
import subprocess
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ───────────────────────────────────────────────────
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_vcrelax")
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "al_vcrelax"
NPROC = 4

# ── Write vc-relax input ───────────────────────────────────────────
vc_relax_input = f"""&CONTROL
    calculation  = 'vc-relax'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
    tprnfor      = .true.
    tstress      = .true.
    forc_conv_thr = 1.0d-4
    etot_conv_thr = 1.0d-6
/

&SYSTEM
    ibrav        = 0
    nat          = 4
    ntyp         = 1
    ecutwfc      = 60.0
    ecutrho      = 480.0
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.01
/

&ELECTRONS
    conv_thr     = 1.0d-8
    mixing_beta  = 0.7
/

&IONS
    ion_dynamics = 'bfgs'
/

&CELL
    cell_dynamics = 'bfgs'
    press         = 0.0
    press_conv_thr = 0.1
    cell_dofree   = 'all'
/

CELL_PARAMETERS (angstrom)
  4.10   0.00   0.00
  0.00   4.10   0.00
  0.00   0.00   4.10

ATOMIC_SPECIES
  Al  26.9815  Al.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Al  0.00  0.00  0.00
  Al  0.50  0.50  0.00
  Al  0.50  0.00  0.50
  Al  0.00  0.50  0.50

K_POINTS (automatic)
  8 8 8  0 0 0
"""

input_file = f"{PREFIX}.in"
output_file = f"{PREFIX}.out"

with open(input_file, "w") as f:
    f.write(vc_relax_input)

print("=" * 70)
print("QE vc-relax: Variable-Cell Relaxation")
print("=" * 70)
print(f"Input: {input_file}")
print(f"Running with {NPROC} MPI processes...")

# ── Run QE ──────────────────────────────────────────────────────────
result = subprocess.run(
    ["mpirun", "-np", str(NPROC), "pw.x", "-in", input_file],
    capture_output=True, text=True, timeout=1800,
)

with open(output_file, "w") as f:
    f.write(result.stdout)

if result.returncode != 0:
    print(f"QE failed! Check {output_file}")
    print(result.stderr[-500:] if result.stderr else "No stderr")
else:
    print(f"QE completed. Output: {output_file}")

# ── Parse results ───────────────────────────────────────────────────
output_text = result.stdout

# Extract energies at each ionic step
energies = []
for m in re.finditer(r"!\s+total energy\s+=\s+([-\d.]+)\s+Ry", output_text):
    energies.append(float(m.group(1)) * 13.6057)  # Ry to eV

# Extract pressures
pressures = []
for m in re.finditer(r"P=\s+([-\d.]+)", output_text):
    pressures.append(float(m.group(1)))  # kbar

# Extract total forces
total_forces = []
for m in re.finditer(r"Total force\s+=\s+([\d.]+)", output_text):
    total_forces.append(float(m.group(1)))

# Check convergence
converged = "Final enthalpy" in output_text or "bfgs converged" in output_text

# Extract final cell
final_cell_match = re.findall(
    r"CELL_PARAMETERS.*?\n(.*?\n.*?\n.*?\n)",
    output_text, re.DOTALL
)

# Extract final positions
final_pos_match = re.findall(
    r"ATOMIC_POSITIONS.*?\n((?:.*\n)*?)(?:\n\n|\Z)",
    output_text
)

print(f"\n{'=' * 50}")
print(f"Results Summary")
print(f"{'=' * 50}")
print(f"Converged: {converged}")
print(f"Ionic steps: {len(energies)}")
if energies:
    print(f"Initial energy: {energies[0]:.6f} eV")
    print(f"Final energy:   {energies[-1]:.6f} eV")
    print(f"Energy change:  {energies[-1] - energies[0]:.6f} eV")
if pressures:
    print(f"Initial pressure: {pressures[0]:.2f} kbar")
    print(f"Final pressure:   {pressures[-1]:.2f} kbar")
if total_forces:
    print(f"Initial total force: {total_forces[0]:.6f} Ry/bohr")
    print(f"Final total force:   {total_forces[-1]:.6f} Ry/bohr")
if final_cell_match:
    print(f"\nFinal cell parameters:")
    print(final_cell_match[-1].strip())
if final_pos_match:
    print(f"\nFinal atomic positions:")
    print(final_pos_match[-1].strip())

# ── Plot convergence ────────────────────────────────────────────────
if len(energies) > 1:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    steps = np.arange(len(energies))
    axes[0].plot(steps, np.array(energies) - energies[-1], "o-",
                 color="#1f77b4", markersize=4)
    axes[0].set_xlabel("Ionic Step")
    axes[0].set_ylabel("E - E_final (eV)")
    axes[0].set_title("Energy Convergence")
    axes[0].grid(alpha=0.3)

    if total_forces:
        axes[1].semilogy(np.arange(len(total_forces)), total_forces, "o-",
                         color="#ff7f0e", markersize=4)
        axes[1].set_xlabel("Ionic Step")
        axes[1].set_ylabel("Total Force (Ry/bohr)")
        axes[1].set_title("Force Convergence")
        axes[1].grid(alpha=0.3)

    if pressures:
        axes[2].plot(np.arange(len(pressures)), pressures, "o-",
                     color="#2ca02c", markersize=4)
        axes[2].axhline(0, color="red", linestyle="--", linewidth=0.8)
        axes[2].set_xlabel("Ionic Step")
        axes[2].set_ylabel("Pressure (kbar)")
        axes[2].set_title("Pressure Convergence")
        axes[2].grid(alpha=0.3)

    plt.suptitle("QE vc-relax Convergence: Al (fcc)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("qe_vcrelax_convergence.png", dpi=200, bbox_inches="tight")
    print("\nPlot saved: qe_vcrelax_convergence.png")
```

### Script 7: Difficult Structure Optimization -- Amorphous/Disordered Materials

Practical workflow for optimizing amorphous or heavily disordered structures where standard optimizers often struggle.

```python
#!/usr/bin/env python3
"""
Optimization workflow for amorphous/disordered materials.

Strategy:
  1. Start with FIRE (robust for rugged PES, no Hessian needed)
  2. Switch to LBFGS once forces are moderate (fast quadratic convergence)
  3. If still not converged, use custom Quasi-Newton with PSB

This staged approach combines the robustness of FIRE for the initial
rough optimization with the fast convergence of quasi-Newton methods
for the final tightening.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase import Atoms
from ase.build import bulk
from ase.optimize import FIRE, LBFGS, BFGS
from ase.io import write
from mace.calculators import mace_mp


def make_amorphous_structure(element="Si", n_atoms=64, density_g_cm3=2.33,
                             seed=42):
    """
    Create a random amorphous structure by placing atoms randomly
    in a cubic box with the specified density, then applying
    minimum distance constraints.

    Args:
        element: chemical symbol
        n_atoms: number of atoms
        density_g_cm3: target density in g/cm^3
        seed: random seed

    Returns:
        ASE Atoms object
    """
    from ase.data import atomic_masses, atomic_numbers

    np.random.seed(seed)
    Z = atomic_numbers[element]
    mass_amu = atomic_masses[Z]

    # Calculate box size from density
    # density = n_atoms * mass / volume
    # mass in grams: mass_amu * 1.66054e-24 g
    # volume in cm^3 -> Ang^3: 1 cm^3 = 1e24 Ang^3
    mass_total_g = n_atoms * mass_amu * 1.66054e-24
    volume_cm3 = mass_total_g / density_g_cm3
    volume_A3 = volume_cm3 * 1e24
    L = volume_A3 ** (1.0 / 3.0)

    print(f"  Box size: {L:.2f} Ang, Volume: {volume_A3:.1f} Ang^3")

    # Place atoms randomly with minimum distance constraint
    min_dist = 1.8  # Angstrom
    positions = []
    max_attempts = n_atoms * 1000

    for _ in range(max_attempts):
        if len(positions) >= n_atoms:
            break
        pos = np.random.rand(3) * L
        # Check minimum distance with periodic boundary conditions
        ok = True
        for existing in positions:
            diff = pos - existing
            diff -= L * np.round(diff / L)  # PBC
            if np.linalg.norm(diff) < min_dist:
                ok = False
                break
        if ok:
            positions.append(pos)

    if len(positions) < n_atoms:
        raise RuntimeError(f"Could only place {len(positions)}/{n_atoms} atoms "
                           f"with min_dist={min_dist} Ang")

    positions = np.array(positions)
    atoms = Atoms(
        symbols=[element] * n_atoms,
        positions=positions,
        cell=[L, L, L],
        pbc=True,
    )
    return atoms


def staged_optimization(atoms, fmax_final=0.01, verbose=True):
    """
    Three-stage optimization:
      Stage 1: FIRE with loose tolerance (robust for rugged PES)
      Stage 2: LBFGS to tighten (fast quadratic convergence)
      Stage 3: If needed, custom QN with PSB for stubborn cases
    """
    calc = atoms.calc
    all_energies = []
    all_forces = []
    stage_boundaries = []

    # Callback to track history
    class Tracker:
        def __init__(self, atoms, elist, flist):
            self.atoms = atoms
            self.elist = elist
            self.flist = flist
        def __call__(self):
            self.elist.append(self.atoms.get_potential_energy())
            self.flist.append(np.linalg.norm(self.atoms.get_forces(), axis=-1).max())

    tracker = Tracker(atoms, all_energies, all_forces)

    # Stage 1: FIRE (robust, handles large forces well)
    fmax_stage1 = max(fmax_final * 10, 0.1)  # loose tolerance
    if verbose:
        print(f"\n  Stage 1: FIRE (fmax target = {fmax_stage1:.3f} eV/Ang)")
    opt1 = FIRE(atoms, logfile=None)
    opt1.attach(tracker)
    try:
        opt1.run(fmax=fmax_stage1, steps=500)
    except Exception as e:
        if verbose:
            print(f"    FIRE stopped: {e}")
    stage_boundaries.append(len(all_energies))
    if verbose:
        fm = all_forces[-1] if all_forces else float("nan")
        print(f"    After FIRE: {len(all_energies)} steps, Fmax = {fm:.4f} eV/Ang")

    # Stage 2: LBFGS (fast convergence in smooth region)
    fmax_stage2 = fmax_final
    if verbose:
        print(f"\n  Stage 2: LBFGS (fmax target = {fmax_stage2:.3f} eV/Ang)")
    opt2 = LBFGS(atoms, logfile=None)
    opt2.attach(tracker)
    try:
        opt2.run(fmax=fmax_stage2, steps=300)
    except Exception as e:
        if verbose:
            print(f"    LBFGS stopped: {e}")
    stage_boundaries.append(len(all_energies))
    if verbose:
        fm = all_forces[-1] if all_forces else float("nan")
        print(f"    After LBFGS: {len(all_energies)} total steps, Fmax = {fm:.4f}")

    # Check if we need Stage 3
    if all_forces and all_forces[-1] > fmax_final:
        if verbose:
            print(f"\n  Stage 3: Custom QN-PSB (final tightening)")

        # Inline PSB optimizer
        n_dof = 3 * len(atoms)
        H = 10.0 * np.eye(n_dof)
        g_old = None
        dx = None

        for step in range(200):
            e = atoms.get_potential_energy()
            f = atoms.get_forces()
            fm = np.linalg.norm(f, axis=-1).max()
            all_energies.append(e)
            all_forces.append(fm)
            if fm < fmax_final:
                break
            g = -f.flatten()
            if g_old is not None and dx is not None:
                dg = g - g_old
                dx_f = dx.flatten()
                r = dg - H.dot(dx_f)
                dxdx = dx_f.dot(dx_f)
                if dxdx > 1e-12:
                    dH = np.outer(r, dx_f)
                    dH = (dH + dH.T) / dxdx
                    H = dH - np.outer(dx_f, dx_f) * r.dot(dx_f) / dxdx**2 + H
            g_old = g.copy()
            eigv, eigvc = np.linalg.eigh(H)
            reg = -2.0
            for _ in range(30):
                soft = eigv / (eigv**2 + np.exp(reg))
                H_inv = eigvc @ np.diag(soft) @ eigvc.T
                s = -H_inv.dot(g).reshape(-1, 3)
                if np.linalg.norm(s, axis=-1).max() < 0.1:
                    break
                reg += 1.0
            dx = s.copy()
            atoms.positions += dx

        stage_boundaries.append(len(all_energies))
        if verbose:
            fm = all_forces[-1] if all_forces else float("nan")
            print(f"    After QN-PSB: {len(all_energies)} total steps, Fmax = {fm:.4f}")

    return {
        "energies": np.array(all_energies),
        "max_forces": np.array(all_forces),
        "stage_boundaries": stage_boundaries,
    }


# ── Main ────────────────────────────────────────────────────────────
print("=" * 70)
print("Amorphous Structure Optimization: Staged Approach")
print("=" * 70)

# Create amorphous Si
print("\nCreating amorphous Si (64 atoms)...")
atoms = make_amorphous_structure("Si", n_atoms=64, density_g_cm3=2.33, seed=42)
print(f"  Created {len(atoms)} atoms in {atoms.cell.cellpar()[0]:.2f} Ang box")

calc = mace_mp(model="medium", dispersion=False, device="cpu")
atoms.calc = calc

e0 = atoms.get_potential_energy()
f0 = np.linalg.norm(atoms.get_forces(), axis=-1).max()
print(f"  Initial energy: {e0:.4f} eV")
print(f"  Initial max force: {f0:.4f} eV/Ang")

# Run staged optimization
result = staged_optimization(atoms, fmax_final=0.01, verbose=True)

# Save optimized structure
write("amorphous_Si_optimized.xyz", atoms)
print(f"\nOptimized structure saved: amorphous_Si_optimized.xyz")

# ── Plot ────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

steps = np.arange(len(result["energies"]))
ax1.plot(steps, result["energies"] - result["energies"][-1],
         color="#1f77b4", linewidth=1.2)
ax2.semilogy(steps, result["max_forces"], color="#1f77b4", linewidth=1.2)

# Mark stage boundaries
stage_colors = ["#ff7f0e", "#2ca02c", "#d62728"]
stage_names = ["FIRE", "LBFGS", "QN-PSB"]
for i, boundary in enumerate(result["stage_boundaries"]):
    if boundary < len(steps):
        ax1.axvline(boundary, color=stage_colors[i], linestyle="--",
                    linewidth=1, alpha=0.7)
        ax2.axvline(boundary, color=stage_colors[i], linestyle="--",
                    linewidth=1, alpha=0.7, label=f"Stage {i+1} end: {stage_names[i]}")

ax1.set_xlabel("Step", fontsize=12)
ax1.set_ylabel("E - E_final (eV)", fontsize=12)
ax1.set_title("Energy Convergence", fontsize=13)
ax1.grid(alpha=0.3)

ax2.set_xlabel("Step", fontsize=12)
ax2.set_ylabel("Max Force (eV/Ang)", fontsize=12)
ax2.set_title("Force Convergence", fontsize=13)
ax2.axhline(0.01, color="red", linestyle="--", linewidth=0.8, label="Tolerance")
ax2.legend(fontsize=9, loc="upper right")
ax2.grid(alpha=0.3)

plt.suptitle(f"Staged Optimization: Amorphous Si ({len(atoms)} atoms)",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("amorphous_optimization.png", dpi=200, bbox_inches="tight")
print("Plot saved: amorphous_optimization.png")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `starting_h` | 5--20 eV/Ang^2 | Initial diagonal Hessian. Higher = more conservative steps. Start with 10. |
| `max_displacement` | 0.05--0.2 Ang | Maximum per-atom displacement per step. Prevents unphysical jumps. |
| `fmax` | 0.01--0.001 eV/Ang | Force convergence criterion. 0.01 for screening, 0.001 for production. |
| `ionic_steps` | 100--500 | Maximum optimization steps. Increase for large/difficult systems. |
| `mode` | PSB, SR1, BFGS | Hessian update scheme. PSB is a good default. BFGS for guaranteed positive definite. SR1 near saddle points. |
| `use_eigenvalue_softening` | True | Eigenvalue-based regularization. More stable than Tikhonov near saddle points. |
| `target_pressure` | 0.0 GPa | Target pressure for cell relaxation. Set to 0 for zero-pressure ground state. |
| `pressure_tol` | 0.1--1.0 GPa | Pressure convergence tolerance. 0.5 GPa is reasonable for MLIP. |
| `cell_dofree` | all / shape / volume | QE cell DOFs: `all` = full, `shape` = fixed volume, `volume` = isotropic only. |

## Interpreting Results

- **Monotonic energy decrease**: Normal for BFGS and CG. FIRE may oscillate initially but should decrease on average.
- **Force oscillation without convergence**: Hessian may be ill-conditioned. Try increasing `starting_h`, reducing `max_displacement`, or switching update scheme.
- **Very slow convergence**: The PES may be very flat (soft modes). Check for rattling atoms or near-degenerate configurations.
- **Negative eigenvalues in Hessian**: The structure is near a saddle point. Use SR1 or PSB (not BFGS). Consider whether you want the saddle point or a minimum.
- **Pressure not converging**: Cell optimization step size may be too large or too small. Adjust the strain step factor.
- **Energy increases during optimization**: With CG/L-BFGS-B, this suggests the line search is failing. Try FIRE or reduce `max_displacement`.

## Common Issues

| Problem | Solution |
|---|---|
| Hessian becomes singular/NaN | Reduce `max_displacement`. Increase `starting_h`. Check for overlapping atoms. |
| BFGS oscillates, won't converge | Switch to FIRE for initial relaxation, then BFGS for final tightening (staged approach). |
| Out of memory for large systems | Use L-BFGS-B or ASE LBFGS instead of full Hessian methods. Full Hessian scales as O(9N^2). |
| Cell optimization diverges | Simultaneous cell+position optimization is ill-posed. Use volume-only first, then relax positions, or use ASE FrechetCellFilter. |
| FIRE takes too many steps | FIRE is intentionally conservative. It will converge but may need 3--5x more steps than BFGS. |
| Scipy minimizer ignores force tolerance | Scipy uses `tol` for its own convergence criteria. Implement force checking in the callback instead. |
| QE vc-relax not converging | Increase `nstep` in `&CONTROL`. Check that `press_conv_thr` is not too tight. Try `cell_dofree='volume'` first. |
| Wrong cell shape after vc-relax | Symmetry may be broken. Set `cell_dofree='shape'` to preserve volume, or use `nosym=.true.` if intended. |
