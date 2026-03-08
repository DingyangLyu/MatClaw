# Activation Relaxation Technique (ART nouveau)

## When to Use

- Find transition states and saddle points **without knowing the final state** in advance.
- Explore energy landscapes around defects (vacancies, interstitials, impurities) to discover all possible migration pathways.
- Study diffusion mechanisms in amorphous, disordered, or complex crystalline materials where the final state is not obvious.
- Surface diffusion and reconstruction events where multiple competing pathways exist.
- Build a catalog of saddle points and activation energies for kinetic Monte Carlo (KMC) input.
- Complement NEB calculations: use ART to discover unknown pathways, then refine with NEB for publication-quality barriers.

## Method Selection (ART vs NEB)

| Criterion | ART nouveau | NEB / CI-NEB |
|---|---|---|
| Final state required? | No -- discovers it automatically | Yes -- both endpoints must be known |
| Best for | Pathway exploration, amorphous systems, complex defects | Known reaction coordinates, well-defined A->B transitions |
| Output | Saddle point + new minimum (final state) | Minimum energy path with all images |
| Computational cost | One atom pushed at a time; moderate | All images relaxed simultaneously; scales with n_images |
| Accuracy | Finds first-order saddle points; barrier from E_saddle - E_initial | CI-NEB gives accurate saddle point on the MEP |
| Limitation | May miss concerted multi-atom pathways | Requires good guess for endpoints |
| Typical use | Discovery phase: "what transitions are possible?" | Refinement phase: "what is the exact barrier for this path?" |

**Decision flow:**

```
Do you know the initial AND final state?
  YES --> Use NEB (see neb-transition-state/ skill)
  NO  --> Use ART nouveau (this skill)

Exploring all possible transitions from a defect?
  --> ART nouveau with random initial directions

Want to build a KMC event catalog?
  --> ART nouveau: systematic sampling of saddle points

Amorphous or highly disordered system?
  --> ART nouveau: NEB endpoints are hard to guess

Need publication-quality barrier for a specific path?
  --> Discover path with ART, then refine with CI-NEB
```

## Prerequisites

- ASE (structure manipulation, geometry optimization)
- MACE (`mace-torch`) for fast energy/force evaluation, or LAMMPS with an interatomic potential
- numpy, scipy (Lanczos algorithm, linear algebra)
- matplotlib (visualization)
- pymatgen (structure building, optional)
- All packages are pre-installed in the container.

## Background: The ART nouveau Algorithm

The Activation Relaxation Technique (ART nouveau) finds first-order saddle points on a potential energy surface through three phases:

1. **Activation (push)**: Displace a selected atom along a random direction. Apply a modified force that reverses the component along the push direction, driving the system uphill in energy while allowing perpendicular relaxation. The modified force is:

   `f_art = f - (1 + gamma) * dot(n, f) * n`

   where `n` is the push direction and `gamma > 0` controls the push strength.

2. **Convergence to saddle point**: Track the lowest eigenvalue of the Hessian (via the Lanczos algorithm). Once the lowest eigenvalue becomes negative (the system has crossed a ridge), follow the eigenvector corresponding to this negative eigenvalue while minimizing forces in all perpendicular directions. This converges the system to a first-order saddle point.

3. **Relaxation to new minimum**: From the saddle point, push slightly along the unstable eigenvector (the direction of negative curvature) and minimize energy to find the new local minimum (the final state).

The activation energy is: `E_a = E_saddle - E_initial`.

Reference: Barkema & Mousseau, Phys. Rev. Lett. 77, 4358 (1996); Mousseau & Barkema, Phys. Rev. E 57, 2419 (1998).

---

## Detailed Steps

### Method A: Complete ART nouveau with ASE + MACE

#### Full Standalone ART Search for Vacancy Diffusion in BCC Fe

```python
#!/usr/bin/env python3
"""
Activation Relaxation Technique (ART nouveau) for saddle point searching.

Standalone implementation using ASE + MACE -- no pyiron dependency.
Discovers transition states and activation energies without knowing
the final state.

Example: vacancy diffusion pathways in BCC Fe.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from ase.build import bulk
from ase.optimize import BFGS, FIRE
from ase.io import write, Trajectory
from ase.constraints import FixAtoms
from mace.calculators import mace_mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os
import copy

# ============================================================
# 1. Configuration
# ============================================================
SYSTEM_NAME = "BCC Fe vacancy"
SUPERCELL_SIZE = [4, 4, 4]         # 4x4x4 supercell
VACANCY_INDEX = 0                   # atom to remove for vacancy
ART_ATOM_ID = None                  # set below: nearest neighbor to vacancy
PUSH_STEP = 0.15                    # Angstrom, initial push magnitude
GAMMA = 0.15                        # force reversal strength (0.05-0.3 typical)
EIGENVALUE_THRESHOLD = -0.5         # eV/A^2, switch to saddle convergence
FMAX_PERPENDICULAR = 0.05           # eV/A, perpendicular force convergence
FMAX_RELAX = 0.01                   # eV/A, for endpoint relaxation
MAX_ART_STEPS = 150                 # max iterations in ART search
MAX_PUSH_STEPS = 30                 # max steps in initial push phase
N_LANCZOS = 12                      # Lanczos vectors for eigenvalue estimation
LANCZOS_DR = 0.01                   # finite difference step for Hessian-vector product
SADDLE_FORCE_TOL = 0.05             # eV/A, saddle point convergence
N_RANDOM_SEARCHES = 5               # number of ART searches with random directions
SEED = 42

np.random.seed(SEED)

# ============================================================
# 2. Build structure with vacancy
# ============================================================
print("=" * 60)
print("ART nouveau: Saddle Point Search")
print("=" * 60)

# BCC Fe
atoms_bulk = bulk("Fe", "bcc", a=2.87, cubic=True)
atoms_bulk *= SUPERCELL_SIZE
n_atoms_bulk = len(atoms_bulk)
print(f"Bulk supercell: {n_atoms_bulk} atoms")

# Create vacancy
atoms = atoms_bulk.copy()
vacancy_pos = atoms.positions[VACANCY_INDEX].copy()
del atoms[VACANCY_INDEX]
n_atoms = len(atoms)
print(f"Vacancy created at {vacancy_pos}, {n_atoms} atoms remaining")

# Set up calculator
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
atoms.calc = calc

# Relax the vacancy structure
print("\nRelaxing initial vacancy structure...")
opt = BFGS(atoms, logfile="art_initial_relax.log")
opt.run(fmax=FMAX_RELAX, steps=500)
e_initial = atoms.get_potential_energy()
pos_initial = atoms.positions.copy()
print(f"  E_initial = {e_initial:.6f} eV")

# Find nearest neighbor to vacancy (the atom most likely to hop)
distances = np.linalg.norm(atoms.positions - vacancy_pos, axis=1)
sorted_indices = np.argsort(distances)
nearest_neighbors = sorted_indices[:8]  # first few nearest neighbors
print(f"\nNearest neighbors to vacancy:")
for nn in nearest_neighbors[:4]:
    print(f"  Atom {nn}: distance = {distances[nn]:.3f} A")

# Default: use the nearest neighbor as the ART atom
if ART_ATOM_ID is None:
    ART_ATOM_ID = nearest_neighbors[0]
print(f"\nART atom (pushed atom): {ART_ATOM_ID}, distance to vacancy = {distances[ART_ATOM_ID]:.3f} A")


# ============================================================
# 3. Core ART routines
# ============================================================
def get_energy_and_forces(atoms_work):
    """Get energy and forces from the calculator."""
    e = atoms_work.get_potential_energy()
    f = atoms_work.get_forces()
    return e, f


def hessian_vector_product(atoms_work, direction, dr=LANCZOS_DR):
    """
    Compute Hessian-vector product H*v using finite differences.
    H*v ~ [f(x - dr*v) - f(x + dr*v)] / (2*dr)

    Parameters
    ----------
    atoms_work : ase.Atoms
        Current configuration.
    direction : np.ndarray, shape (n_atoms, 3)
        Direction vector (will be normalized).
    dr : float
        Finite difference step size.

    Returns
    -------
    Hv : np.ndarray, shape (n_atoms, 3)
        Hessian-vector product.
    """
    v = direction / np.linalg.norm(direction)
    pos0 = atoms_work.positions.copy()

    # Forward step
    atoms_work.positions = pos0 + dr * v
    f_plus = atoms_work.get_forces()

    # Backward step
    atoms_work.positions = pos0 - dr * v
    f_minus = atoms_work.get_forces()

    # Restore
    atoms_work.positions = pos0

    # H*v = -(f+ - f-) / (2*dr)  (negative because f = -grad E)
    Hv = -(f_plus - f_minus) / (2.0 * dr)
    return Hv


def lanczos_lowest_eigenvalue(atoms_work, n_lanczos=N_LANCZOS, dr=LANCZOS_DR):
    """
    Find the lowest eigenvalue and eigenvector of the Hessian using
    the Lanczos algorithm via scipy's eigsh with a LinearOperator.

    This avoids building or storing the full Hessian matrix.

    Returns
    -------
    eigenvalue : float
        Lowest eigenvalue of the Hessian.
    eigenvector : np.ndarray, shape (n_atoms, 3)
        Corresponding eigenvector, reshaped to (n_atoms, 3).
    """
    n = len(atoms_work) * 3  # total degrees of freedom

    def matvec(x):
        v = x.reshape(-1, 3)
        Hv = hessian_vector_product(atoms_work, v, dr=dr)
        return Hv.flatten()

    H_op = LinearOperator((n, n), matvec=matvec, dtype=float)

    # Find the smallest eigenvalue
    n_eigs = min(n_lanczos, n - 2)
    if n_eigs < 1:
        n_eigs = 1

    try:
        eigenvalues, eigenvectors = eigsh(H_op, k=1, which="SA",
                                          maxiter=n * 10, tol=1e-4)
        eigval = eigenvalues[0]
        eigvec = eigenvectors[:, 0].reshape(-1, 3)
    except Exception:
        # Fallback: use a random vector and power iteration
        eigval = 0.0
        eigvec = np.random.randn(len(atoms_work), 3)
        eigvec /= np.linalg.norm(eigvec)

    return eigval, eigvec


def art_modified_force(forces, art_atom_id, direction, gamma=GAMMA):
    """
    Apply the ART force modification.

    The force on the ART atom along the push direction is reversed:
      f_art = f - (1 + gamma) * dot(n, f_art_atom) * n

    The reaction force is distributed among all other atoms to
    conserve total force.

    Parameters
    ----------
    forces : np.ndarray, shape (n_atoms, 3)
        Original forces.
    art_atom_id : int
        Index of the atom being pushed.
    direction : np.ndarray, shape (3,)
        Push direction (will be normalized).
    gamma : float
        Strength of force reversal (> 0).

    Returns
    -------
    f_modified : np.ndarray, shape (n_atoms, 3)
        Modified forces.
    """
    f = forces.copy()
    n = direction / np.linalg.norm(direction)

    # Force component along push direction on ART atom
    f_along = np.dot(f[art_atom_id], n) * n

    # Reverse and amplify this component
    f_art_correction = -(1.0 + gamma) * f_along

    # Apply to ART atom
    f[art_atom_id] += f_art_correction

    # Distribute reaction force among other atoms
    n_other = len(f) - 1
    if n_other > 0:
        f_reaction = -f_art_correction / n_other
        for i in range(len(f)):
            if i != art_atom_id:
                f[i] += f_reaction

    return f


def project_perpendicular(vector, direction):
    """Project vector perpendicular to direction."""
    n = direction / np.linalg.norm(direction)
    return vector - np.dot(vector.flatten(), n.flatten()) * n.reshape(vector.shape)


def project_parallel(vector, direction):
    """Project vector parallel to direction."""
    n = direction / np.linalg.norm(direction)
    return np.dot(vector.flatten(), n.flatten()) * n.reshape(vector.shape)


def run_art_search(atoms_start, art_atom_id, initial_direction, search_id=0):
    """
    Run a single ART nouveau search from a given initial state.

    Phases:
      1. Push phase: displace art atom, use ART-modified forces
      2. Saddle convergence: follow lowest eigenvector, minimize perp forces
      3. Relaxation: push past saddle, minimize to find new basin

    Parameters
    ----------
    atoms_start : ase.Atoms
        Relaxed initial configuration.
    art_atom_id : int
        Atom index to push.
    initial_direction : np.ndarray, shape (3,)
        Initial push direction.
    search_id : int
        Identifier for this search.

    Returns
    -------
    result : dict or None
        Dictionary with saddle point info, or None if search failed.
    """
    atoms_work = atoms_start.copy()
    atoms_work.calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")

    e_start = atoms_work.get_potential_energy()
    pos_start = atoms_work.positions.copy()

    direction = np.array(initial_direction, dtype=float)
    direction /= np.linalg.norm(direction)

    trajectory_energies = [e_start]
    trajectory_eigenvalues = []
    trajectory_positions = [pos_start.copy()]

    print(f"\n--- ART Search #{search_id} ---")
    print(f"  Push direction: [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]")

    # ----------------------------------------------------------
    # Phase 1: Push phase -- activate the system
    # ----------------------------------------------------------
    print("  Phase 1: Activation (push)...")
    lowest_eigenvalue = 0.0

    for step in range(MAX_PUSH_STEPS):
        # Displace ART atom along push direction
        atoms_work.positions[art_atom_id] += PUSH_STEP * direction

        # Get forces and apply ART modification
        e, forces = get_energy_and_forces(atoms_work)
        f_mod = art_modified_force(forces, art_atom_id, direction, gamma=GAMMA)

        # Relax perpendicular degrees of freedom (a few steps)
        f_perp = project_perpendicular(f_mod, direction.reshape(1, 3).repeat(len(atoms_work), axis=0))
        atoms_work.positions += 0.05 * f_perp  # small step in perp direction

        # Check eigenvalue periodically
        if step % 3 == 0 and step > 0:
            lowest_eigenvalue, eigvec = lanczos_lowest_eigenvalue(atoms_work)
            trajectory_eigenvalues.append((step, lowest_eigenvalue))
            if lowest_eigenvalue < EIGENVALUE_THRESHOLD:
                print(f"    Step {step}: lambda_min = {lowest_eigenvalue:.4f} eV/A^2 "
                      f"< threshold ({EIGENVALUE_THRESHOLD})")
                print(f"    Entering saddle convergence phase.")
                direction = eigvec[art_atom_id].copy()
                direction /= np.linalg.norm(direction)
                break

        trajectory_energies.append(atoms_work.get_potential_energy())
        trajectory_positions.append(atoms_work.positions.copy())

    else:
        print(f"    Push phase did not find negative eigenvalue in {MAX_PUSH_STEPS} steps.")
        print(f"    Last eigenvalue: {lowest_eigenvalue:.4f} eV/A^2")
        return None

    # ----------------------------------------------------------
    # Phase 2: Converge to saddle point
    # ----------------------------------------------------------
    print("  Phase 2: Saddle point convergence...")

    for step in range(MAX_ART_STEPS):
        e, forces = get_energy_and_forces(atoms_work)

        # Get lowest eigenvalue and eigenvector
        eigval, eigvec = lanczos_lowest_eigenvalue(atoms_work)

        # Ensure eigenvector points "uphill" (same as push direction)
        eigvec_flat = eigvec.flatten()
        if np.dot(eigvec_flat, direction.repeat(len(atoms_work) // 1)) < 0:
            # Use a simpler check: project on ART atom direction
            pass
        art_eigvec = eigvec[art_atom_id]
        if np.dot(art_eigvec, direction) < 0:
            eigvec = -eigvec

        # Force components
        f_parallel = project_parallel(forces, eigvec)
        f_perp = forces - f_parallel

        # Saddle point convergence criterion
        f_total = np.sqrt(np.sum(forces**2, axis=1))
        fmax_current = np.max(f_total)

        if step % 10 == 0:
            print(f"    Step {step}: E = {e:.6f} eV, lambda = {eigval:.4f} eV/A^2, "
                  f"fmax = {fmax_current:.4f} eV/A")

        trajectory_energies.append(e)
        trajectory_eigenvalues.append((step + MAX_PUSH_STEPS, eigval))
        trajectory_positions.append(atoms_work.positions.copy())

        if fmax_current < SADDLE_FORCE_TOL and eigval < -0.1:
            print(f"    Saddle point converged at step {step}!")
            print(f"    E_saddle = {e:.6f} eV, lambda_min = {eigval:.4f} eV/A^2")
            break

        # Move: follow eigenvector uphill, minimize perpendicular forces
        # Along eigenvector: invert the force (go uphill)
        step_parallel = -0.05 * f_parallel  # invert to go uphill
        step_perp = 0.05 * f_perp           # follow forces downhill perp

        atoms_work.positions += step_parallel + step_perp

    else:
        print(f"    Saddle convergence did not converge in {MAX_ART_STEPS} steps.")
        # Still report the approximate saddle point
        e = atoms_work.get_potential_energy()

    # Record saddle point
    e_saddle = atoms_work.get_potential_energy()
    pos_saddle = atoms_work.positions.copy()
    barrier = e_saddle - e_start

    if barrier < 0.01:
        print(f"    Barrier too small ({barrier:.4f} eV). Likely did not reach a true saddle.")
        return None

    print(f"\n  Saddle point found:")
    print(f"    E_saddle = {e_saddle:.6f} eV")
    print(f"    E_barrier = {barrier:.4f} eV")

    # ----------------------------------------------------------
    # Phase 3: Relax to new minimum
    # ----------------------------------------------------------
    print("  Phase 3: Relaxation to new minimum...")

    # Push slightly past the saddle along the unstable eigenvector
    eigval_saddle, eigvec_saddle = lanczos_lowest_eigenvalue(atoms_work)
    atoms_relax = atoms_work.copy()
    atoms_relax.calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")

    # Push in the direction away from the initial state
    push_dir = eigvec_saddle[art_atom_id]
    if np.dot(push_dir, pos_saddle[art_atom_id] - pos_start[art_atom_id]) < 0:
        eigvec_saddle = -eigvec_saddle
    atoms_relax.positions += 0.3 * eigvec_saddle

    # Minimize to find the new basin
    opt_final = BFGS(atoms_relax, logfile=f"art_relax_search{search_id}.log")
    opt_final.run(fmax=FMAX_RELAX, steps=500)
    e_final = atoms_relax.get_potential_energy()
    pos_final = atoms_relax.positions.copy()

    # Check if final state is different from initial state
    displacement = np.linalg.norm(pos_final - pos_start, axis=1)
    max_displacement = np.max(displacement)
    displaced_atom = np.argmax(displacement)

    print(f"    E_final = {e_final:.6f} eV")
    print(f"    E_reverse_barrier = {e_saddle - e_final:.4f} eV")
    print(f"    Max displacement: atom {displaced_atom}, {max_displacement:.3f} A")

    is_new_state = max_displacement > 0.5  # at least 0.5 A displacement

    result = {
        "search_id": search_id,
        "art_atom_id": art_atom_id,
        "initial_direction": initial_direction.tolist(),
        "e_initial_eV": float(e_start),
        "e_saddle_eV": float(e_saddle),
        "e_final_eV": float(e_final),
        "barrier_forward_eV": float(barrier),
        "barrier_reverse_eV": float(e_saddle - e_final),
        "max_displacement_A": float(max_displacement),
        "displaced_atom": int(displaced_atom),
        "is_new_state": is_new_state,
        "saddle_eigenvalue": float(eigval_saddle),
        "trajectory_energies": [float(e) for e in trajectory_energies],
        "trajectory_eigenvalues": [(int(s), float(v)) for s, v in trajectory_eigenvalues],
    }

    # Save structures
    write(f"art_saddle_search{search_id}.xyz", atoms_work)
    write(f"art_final_search{search_id}.xyz", atoms_relax)

    return result


# ============================================================
# 4. Run multiple ART searches with random directions
# ============================================================
print(f"\nRunning {N_RANDOM_SEARCHES} ART searches with random directions...")
all_results = []

for i in range(N_RANDOM_SEARCHES):
    # Generate random push direction
    direction = np.random.randn(3)
    direction /= np.linalg.norm(direction)

    # Point generally toward the vacancy for better success rate
    to_vacancy = vacancy_pos - atoms.positions[ART_ATOM_ID]
    to_vacancy /= np.linalg.norm(to_vacancy)

    # Mix random direction with bias toward vacancy (70% bias)
    direction = 0.3 * direction + 0.7 * to_vacancy
    direction /= np.linalg.norm(direction)

    result = run_art_search(atoms, ART_ATOM_ID, direction, search_id=i)
    if result is not None:
        all_results.append(result)

# ============================================================
# 5. Analyze and summarize results
# ============================================================
print("\n" + "=" * 60)
print("ART Search Summary")
print("=" * 60)

if not all_results:
    print("No successful ART searches. Try adjusting parameters:")
    print("  - Increase MAX_PUSH_STEPS or MAX_ART_STEPS")
    print("  - Reduce EIGENVALUE_THRESHOLD (less negative)")
    print("  - Try different GAMMA values (0.05-0.3)")
    print("  - Try different push directions or ART atoms")
else:
    print(f"\n{'Search':<8} {'Barrier (eV)':<14} {'Rev. Bar.':<12} {'New state?':<12} {'Max disp. (A)'}")
    print("-" * 60)
    for r in all_results:
        print(f"  {r['search_id']:<6} {r['barrier_forward_eV']:<14.4f} "
              f"{r['barrier_reverse_eV']:<12.4f} {'Yes' if r['is_new_state'] else 'No':<12} "
              f"{r['max_displacement_A']:<.3f}")

    # Find unique saddle points (different barriers indicate different saddle points)
    barriers = sorted(set(round(r["barrier_forward_eV"], 2) for r in all_results))
    print(f"\nDistinct barriers found: {barriers} eV")

    lowest = min(all_results, key=lambda r: r["barrier_forward_eV"])
    print(f"Lowest barrier: {lowest['barrier_forward_eV']:.4f} eV (search #{lowest['search_id']})")

# ============================================================
# 6. Visualization
# ============================================================
if all_results:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Energy trajectories
    ax = axes[0]
    for r in all_results:
        steps = range(len(r["trajectory_energies"]))
        e_rel = [e - r["e_initial_eV"] for e in r["trajectory_energies"]]
        ax.plot(steps, e_rel, "-", linewidth=1.5, alpha=0.8,
                label=f"#{r['search_id']}: {r['barrier_forward_eV']:.3f} eV")
    ax.set_xlabel("ART step", fontsize=12)
    ax.set_ylabel("Energy - E_initial (eV)", fontsize=12)
    ax.set_title("ART Energy Trajectories", fontsize=13)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Eigenvalue evolution
    ax = axes[1]
    for r in all_results:
        if r["trajectory_eigenvalues"]:
            steps, eigvals = zip(*r["trajectory_eigenvalues"])
            ax.plot(steps, eigvals, "o-", markersize=4, linewidth=1.2, alpha=0.8,
                    label=f"#{r['search_id']}")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8, label="Zero")
    ax.axhline(y=EIGENVALUE_THRESHOLD, color="gray", linestyle=":", linewidth=0.8,
               label=f"Threshold ({EIGENVALUE_THRESHOLD})")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Lowest Hessian eigenvalue (eV/$\\AA^2$)", fontsize=12)
    ax.set_title("Eigenvalue During ART Search", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Barrier histogram
    ax = axes[2]
    bar_values = [r["barrier_forward_eV"] for r in all_results]
    ax.hist(bar_values, bins=max(len(bar_values) // 2, 3), color="steelblue",
            edgecolor="black", alpha=0.8)
    ax.set_xlabel("Activation energy (eV)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Barrier Distribution", fontsize=13)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"ART nouveau: {SYSTEM_NAME}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("art_summary.png", dpi=150, bbox_inches="tight")
    print("\nSummary plot saved to art_summary.png")

# Save all results to JSON
with open("art_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=float)
print("Results saved to art_results.json")

# Save initial structure for reference
write("art_initial_structure.xyz", atoms)
print("Initial structure saved to art_initial_structure.xyz")
```

### Method B: ART nouveau for Interstitial Migration

```python
#!/usr/bin/env python3
"""
ART nouveau search for interstitial migration pathways.
Example: Carbon interstitial in BCC Fe (octahedral site).

This demonstrates ART for systems where the migrating species
is an interstitial rather than a vacancy.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from ase.build import bulk
from ase.optimize import BFGS
from ase.io import write
from mace.calculators import mace_mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

np.random.seed(123)

# ============================================================
# 1. Build BCC Fe with C interstitial at octahedral site
# ============================================================
atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
atoms *= [3, 3, 3]

# Add C at octahedral site: (0.5, 0.5, 0) in unit cell coords
# In the supercell, this is at (0.5*a, 0.5*a, 0)
a = 2.87
oct_position = np.array([0.5 * a, 0.5 * a, 0.0])
from ase import Atoms
c_atom = Atoms("C", positions=[oct_position])
atoms += c_atom

n_atoms = len(atoms)
c_index = n_atoms - 1  # C is the last atom
print(f"System: {n_atoms} atoms (BCC Fe + 1 C interstitial)")
print(f"C interstitial at index {c_index}")

# Set up calculator and relax
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
atoms.calc = calc

print("Relaxing initial configuration...")
opt = BFGS(atoms, logfile="art_interstitial_relax.log")
opt.run(fmax=0.01, steps=500)
e_initial = atoms.get_potential_energy()
pos_initial = atoms.positions.copy()
print(f"E_initial = {e_initial:.6f} eV")
print(f"C position after relax: {atoms.positions[c_index]}")


# ============================================================
# 2. ART search functions (reusable)
# ============================================================
def hessian_vector_product(atoms_w, v_direction, dr=0.01):
    """Finite-difference Hessian-vector product."""
    v = v_direction / np.linalg.norm(v_direction)
    pos0 = atoms_w.positions.copy()
    atoms_w.positions = pos0 + dr * v
    f_plus = atoms_w.get_forces()
    atoms_w.positions = pos0 - dr * v
    f_minus = atoms_w.get_forces()
    atoms_w.positions = pos0
    return -(f_plus - f_minus) / (2.0 * dr)


def lanczos_lowest(atoms_w, dr=0.01):
    """Find lowest eigenvalue/eigenvector of the Hessian."""
    n = len(atoms_w) * 3
    def matvec(x):
        return hessian_vector_product(atoms_w, x.reshape(-1, 3), dr).flatten()
    H_op = LinearOperator((n, n), matvec=matvec, dtype=float)
    try:
        vals, vecs = eigsh(H_op, k=1, which="SA", tol=1e-4)
        return vals[0], vecs[:, 0].reshape(-1, 3)
    except Exception:
        return 0.0, np.random.randn(len(atoms_w), 3)


def art_push_and_converge(atoms_start, push_atom, push_dir, gamma=0.15,
                           push_step=0.12, eigval_threshold=-0.3,
                           max_push=30, max_saddle=120, saddle_ftol=0.05):
    """
    Run one ART search: push, find saddle, relax to new minimum.

    Parameters
    ----------
    atoms_start : ase.Atoms
        Relaxed initial state.
    push_atom : int
        Atom index to push.
    push_dir : np.ndarray, shape (3,)
        Initial push direction.
    gamma : float
        Force reversal parameter.
    push_step : float
        Step size during push phase (Angstrom).
    eigval_threshold : float
        Switch to saddle convergence when lowest eigenvalue < this.
    max_push : int
        Maximum push steps.
    max_saddle : int
        Maximum saddle convergence steps.
    saddle_ftol : float
        Force tolerance for saddle convergence.

    Returns
    -------
    dict or None
    """
    atoms_w = atoms_start.copy()
    atoms_w.calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")

    e_start = atoms_w.get_potential_energy()
    pos_start = atoms_w.positions.copy()
    d = push_dir / np.linalg.norm(push_dir)

    # Phase 1: Push
    eigval = 0.0
    for step in range(max_push):
        atoms_w.positions[push_atom] += push_step * d

        forces = atoms_w.get_forces()
        # ART force modification on push atom
        n_hat = d / np.linalg.norm(d)
        f_along = np.dot(forces[push_atom], n_hat) * n_hat
        forces[push_atom] -= (1.0 + gamma) * f_along

        # Small perpendicular relaxation step
        for i in range(len(atoms_w)):
            if i != push_atom:
                f_perp = forces[i] - np.dot(forces[i], n_hat) * n_hat
                atoms_w.positions[i] += 0.03 * f_perp

        if step % 4 == 0 and step > 0:
            eigval, eigvec = lanczos_lowest(atoms_w)
            if eigval < eigval_threshold:
                d = eigvec[push_atom].copy()
                d /= np.linalg.norm(d)
                break
    else:
        return None  # did not activate

    # Phase 2: Saddle convergence
    for step in range(max_saddle):
        e = atoms_w.get_potential_energy()
        forces = atoms_w.get_forces()
        eigval, eigvec = lanczos_lowest(atoms_w)

        # Orient eigenvector consistently
        if np.dot(eigvec[push_atom], d) < 0:
            eigvec = -eigvec

        # Force decomposition
        eigvec_flat = eigvec.flatten()
        eigvec_flat /= np.linalg.norm(eigvec_flat)
        forces_flat = forces.flatten()

        f_par = np.dot(forces_flat, eigvec_flat) * eigvec_flat
        f_perp = forces_flat - f_par

        fmax = np.max(np.linalg.norm(forces, axis=1))

        if fmax < saddle_ftol and eigval < -0.1:
            break

        # Step: go uphill along eigvec, downhill perp
        step_vec = -0.04 * f_par.reshape(-1, 3) + 0.04 * f_perp.reshape(-1, 3)
        atoms_w.positions += step_vec

    e_saddle = atoms_w.get_potential_energy()
    barrier = e_saddle - e_start

    if barrier < 0.01:
        return None

    # Phase 3: Relax to new minimum
    eigval_s, eigvec_s = lanczos_lowest(atoms_w)
    if np.dot(eigvec_s[push_atom], atoms_w.positions[push_atom] - pos_start[push_atom]) < 0:
        eigvec_s = -eigvec_s

    atoms_final = atoms_w.copy()
    atoms_final.calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
    atoms_final.positions += 0.3 * eigvec_s

    opt_f = BFGS(atoms_final, logfile=None)
    opt_f.run(fmax=0.01, steps=500)
    e_final = atoms_final.get_potential_energy()

    disp = np.linalg.norm(atoms_final.positions - pos_start, axis=1)

    return {
        "e_initial": float(e_start),
        "e_saddle": float(e_saddle),
        "e_final": float(e_final),
        "barrier_fwd": float(barrier),
        "barrier_rev": float(e_saddle - e_final),
        "max_displacement": float(np.max(disp)),
        "push_atom_displacement": float(disp[push_atom]),
        "saddle_eigenvalue": float(eigval_s),
    }


# ============================================================
# 3. Run ART searches for C interstitial migration
# ============================================================
print("\nRunning ART searches for C interstitial migration...")
n_searches = 8
results = []

for i in range(n_searches):
    # Random direction for the C interstitial
    d = np.random.randn(3)
    d /= np.linalg.norm(d)

    print(f"\n  Search {i}: direction = [{d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f}]")
    r = art_push_and_converge(atoms, push_atom=c_index, push_dir=d)

    if r is not None:
        print(f"    Barrier = {r['barrier_fwd']:.4f} eV, "
              f"C displacement = {r['push_atom_displacement']:.3f} A")
        r["search_id"] = i
        r["direction"] = d.tolist()
        results.append(r)
    else:
        print(f"    Search failed (no saddle found)")

# ============================================================
# 4. Summary and visualization
# ============================================================
print("\n" + "=" * 60)
print(f"Found {len(results)} saddle points from {n_searches} searches")
print("=" * 60)

if results:
    for r in sorted(results, key=lambda x: x["barrier_fwd"]):
        print(f"  Search {r['search_id']}: barrier = {r['barrier_fwd']:.4f} eV, "
              f"rev = {r['barrier_rev']:.4f} eV, "
              f"C disp = {r['push_atom_displacement']:.3f} A")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Barrier values
    barriers = [r["barrier_fwd"] for r in results]
    search_ids = [r["search_id"] for r in results]
    ax1.bar(range(len(barriers)), sorted(barriers), color="steelblue", edgecolor="black")
    ax1.set_xlabel("Pathway (sorted)", fontsize=12)
    ax1.set_ylabel("Activation energy (eV)", fontsize=12)
    ax1.set_title("C Interstitial Migration Barriers in BCC Fe", fontsize=13)
    ax1.grid(axis="y", alpha=0.3)

    # Forward vs reverse barrier scatter
    fwd = [r["barrier_fwd"] for r in results]
    rev = [r["barrier_rev"] for r in results]
    ax2.scatter(fwd, rev, s=80, c="steelblue", edgecolors="black", zorder=3)
    max_b = max(max(fwd), max(rev)) * 1.1
    ax2.plot([0, max_b], [0, max_b], "k--", alpha=0.3, label="Symmetric")
    ax2.set_xlabel("Forward barrier (eV)", fontsize=12)
    ax2.set_ylabel("Reverse barrier (eV)", fontsize=12)
    ax2.set_title("Path Symmetry Analysis", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    fig.tight_layout()
    fig.savefig("art_interstitial_summary.png", dpi=150)
    print("\nPlot saved to art_interstitial_summary.png")

with open("art_interstitial_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("Results saved to art_interstitial_results.json")
```

### Method C: Saddle Point Verification and Barrier Validation

```python
#!/usr/bin/env python3
"""
Verify ART saddle points and extract validated barriers.

After running ART, it is important to verify that:
1. The saddle point has exactly one negative Hessian eigenvalue (first-order saddle).
2. The barrier is not an artifact of insufficient convergence.
3. The forward and reverse relaxations reach true minima.

This script also compares ART barriers with NEB for validation.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from ase.io import read, write
from ase.optimize import BFGS, FIRE
from ase.mep.neb import NEB, NEBTools, idpp_interpolate
from mace.calculators import mace_mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

# ============================================================
# 1. Load ART results and saddle point structure
# ============================================================
# Load from a previous ART run (or construct test case)
try:
    with open("art_results.json") as f:
        art_results = json.load(f)
    # Use the lowest-barrier result
    best = min(art_results, key=lambda r: r["barrier_forward_eV"])
    search_id = best["search_id"]
    saddle_file = f"art_saddle_search{search_id}.xyz"
    final_file = f"art_final_search{search_id}.xyz"
    initial_file = "art_initial_structure.xyz"

    atoms_initial = read(initial_file)
    atoms_saddle = read(saddle_file)
    atoms_final = read(final_file)
    print(f"Loaded ART search #{search_id}")
    print(f"  ART barrier: {best['barrier_forward_eV']:.4f} eV")
except (FileNotFoundError, json.JSONDecodeError):
    print("No ART results found. Creating a test case...")
    from ase.build import bulk
    atoms_initial = bulk("Fe", "bcc", a=2.87, cubic=True) * [3, 3, 3]
    del atoms_initial[0]
    calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
    atoms_initial.calc = calc
    BFGS(atoms_initial, logfile=None).run(fmax=0.01)
    atoms_saddle = atoms_initial.copy()
    atoms_final = atoms_initial.copy()
    best = {"barrier_forward_eV": 0.0}

# ============================================================
# 2. Verify saddle point: eigenvalue analysis
# ============================================================
print("\n=== Saddle Point Verification ===")
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
atoms_saddle.calc = calc

def compute_lowest_eigenvalues(atoms_w, n_eigs=6, dr=0.01):
    """Compute the n_eigs lowest eigenvalues of the Hessian."""
    n = len(atoms_w) * 3
    def matvec(x):
        v = x.reshape(-1, 3)
        v_norm = v / np.linalg.norm(v)
        pos0 = atoms_w.positions.copy()
        atoms_w.positions = pos0 + dr * v_norm
        fp = atoms_w.get_forces()
        atoms_w.positions = pos0 - dr * v_norm
        fm = atoms_w.get_forces()
        atoms_w.positions = pos0
        return (-(fp - fm) / (2.0 * dr)).flatten()
    H_op = LinearOperator((n, n), matvec=matvec, dtype=float)
    vals, vecs = eigsh(H_op, k=min(n_eigs, n - 2), which="SA", tol=1e-4)
    return vals, vecs

eigenvalues, eigenvectors = compute_lowest_eigenvalues(atoms_saddle, n_eigs=6)

print(f"\nLowest 6 Hessian eigenvalues at saddle point (eV/A^2):")
n_negative = 0
for i, ev in enumerate(eigenvalues):
    marker = " <-- NEGATIVE (unstable mode)" if ev < -0.01 else ""
    print(f"  lambda_{i} = {ev:.4f}{marker}")
    if ev < -0.01:
        n_negative += 1

if n_negative == 1:
    print(f"\nVERIFIED: Exactly 1 negative eigenvalue -- first-order saddle point.")
elif n_negative == 0:
    print(f"\nWARNING: No negative eigenvalues -- this is a local minimum, not a saddle point!")
    print("  The ART search may not have converged properly.")
else:
    print(f"\nWARNING: {n_negative} negative eigenvalues -- higher-order saddle point.")
    print("  This may be a ridge or a poorly converged saddle.")

# Check forces at saddle point
forces_saddle = atoms_saddle.get_forces()
fmax_saddle = np.max(np.linalg.norm(forces_saddle, axis=1))
print(f"\nMax force at saddle: {fmax_saddle:.4f} eV/A")
if fmax_saddle > 0.1:
    print("  WARNING: Forces are large. Consider re-converging the saddle point.")

# ============================================================
# 3. Re-relax endpoints to ensure true minima
# ============================================================
print("\n=== Endpoint Verification ===")

atoms_initial.calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
opt_i = BFGS(atoms_initial, logfile=None)
opt_i.run(fmax=0.005, steps=300)
e_initial = atoms_initial.get_potential_energy()

atoms_final.calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
opt_f = BFGS(atoms_final, logfile=None)
opt_f.run(fmax=0.005, steps=300)
e_final = atoms_final.get_potential_energy()

e_saddle = atoms_saddle.get_potential_energy()

barrier_fwd = e_saddle - e_initial
barrier_rev = e_saddle - e_final

print(f"E_initial = {e_initial:.6f} eV")
print(f"E_saddle  = {e_saddle:.6f} eV")
print(f"E_final   = {e_final:.6f} eV")
print(f"Barrier (forward):  {barrier_fwd:.4f} eV")
print(f"Barrier (reverse):  {barrier_rev:.4f} eV")

# ============================================================
# 4. Compare with NEB (if initial and final states differ)
# ============================================================
displacement = np.linalg.norm(atoms_final.positions - atoms_initial.positions, axis=1)
if np.max(displacement) > 0.5:
    print("\n=== NEB Validation ===")
    print("Running CI-NEB between ART-discovered endpoints...")

    n_images = 5
    images = [atoms_initial.copy()]
    for _ in range(n_images):
        images.append(atoms_initial.copy())
    images.append(atoms_final.copy())

    idpp_interpolate(images, traj=None, log=None)
    for img in images[1:-1]:
        img.calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")

    # Standard NEB first
    neb = NEB(images, climb=False, k=0.1)
    FIRE(neb, logfile="art_neb_validation.log").run(fmax=0.05, steps=300)

    # CI-NEB
    neb_ci = NEB(images, climb=True, k=0.1)
    FIRE(neb_ci, logfile="art_cineb_validation.log").run(fmax=0.03, steps=300)

    tools = NEBTools(images)
    neb_barrier_fwd, neb_barrier_rev = tools.get_barrier()

    print(f"\nNEB barrier (forward):  {neb_barrier_fwd:.4f} eV")
    print(f"NEB barrier (reverse):  {neb_barrier_rev:.4f} eV")
    print(f"\nComparison:")
    print(f"  ART barrier: {barrier_fwd:.4f} eV")
    print(f"  NEB barrier: {neb_barrier_fwd:.4f} eV")
    print(f"  Difference:  {abs(barrier_fwd - neb_barrier_fwd):.4f} eV")

    if abs(barrier_fwd - neb_barrier_fwd) < 0.1:
        print("  GOOD: ART and NEB barriers agree within 0.1 eV.")
    else:
        print("  NOTE: Barriers differ by > 0.1 eV. The ART saddle may be on")
        print("  a different pathway than the NEB minimum energy path.")

    # Plot comparison
    fig, ax = plt.subplots(figsize=(8, 5))

    # NEB profile
    energies_neb = [img.get_potential_energy() for img in images]
    e_ref = energies_neb[0]
    rc_neb = np.linspace(0, 1, len(energies_neb))
    ax.plot(rc_neb, [e - e_ref for e in energies_neb], "o-", color="steelblue",
            linewidth=2, markersize=8, label=f"NEB ({neb_barrier_fwd:.3f} eV)")

    # ART saddle point
    ax.plot(0.5, barrier_fwd, "r*", markersize=18, zorder=5,
            label=f"ART saddle ({barrier_fwd:.3f} eV)")

    # ART energy profile (initial -> saddle -> final)
    ax.plot([0, 0.5, 1], [0, barrier_fwd, e_final - e_initial], "r--",
            linewidth=1.5, alpha=0.5, label="ART (linear)")

    ax.set_xlabel("Reaction coordinate", fontsize=13)
    ax.set_ylabel("Energy (eV)", fontsize=13)
    ax.set_title("ART vs NEB Barrier Comparison", fontsize=14)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("art_vs_neb_comparison.png", dpi=150)
    print("\nComparison plot saved to art_vs_neb_comparison.png")
else:
    print("\nSkipping NEB comparison: initial and final states appear identical.")
    print("The ART search returned to the same basin.")

# ============================================================
# 5. Save verified results
# ============================================================
verified = {
    "e_initial_eV": float(e_initial),
    "e_saddle_eV": float(e_saddle),
    "e_final_eV": float(e_final),
    "barrier_forward_eV": float(barrier_fwd),
    "barrier_reverse_eV": float(barrier_rev),
    "n_negative_eigenvalues": n_negative,
    "lowest_eigenvalues": [float(v) for v in eigenvalues],
    "fmax_at_saddle_eV_A": float(fmax_saddle),
    "is_first_order_saddle": n_negative == 1,
}

with open("art_verified_results.json", "w") as f:
    json.dump(verified, f, indent=2, default=float)
print("\nVerified results saved to art_verified_results.json")
```

### Method D: Systematic ART Catalog for KMC

```python
#!/usr/bin/env python3
"""
Build a catalog of transitions using systematic ART searches.
This is useful as input for kinetic Monte Carlo (KMC) simulations.

For each inequivalent atom near a defect, run multiple ART searches
with random directions to find all accessible saddle points and
transition rates.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from ase.build import bulk
from ase.optimize import BFGS
from ase.io import write
from mace.calculators import mace_mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

np.random.seed(42)

# ============================================================
# 1. Build system: vacancy in FCC Al
# ============================================================
atoms = bulk("Al", "fcc", a=4.05, cubic=True)
atoms *= [3, 3, 3]
vacancy_pos = atoms.positions[0].copy()
del atoms[0]
print(f"FCC Al vacancy: {len(atoms)} atoms")

calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
atoms.calc = calc
BFGS(atoms, logfile=None).run(fmax=0.005, steps=300)
e_initial = atoms.get_potential_energy()
pos_initial = atoms.positions.copy()
print(f"E_initial = {e_initial:.6f} eV")


# ============================================================
# 2. Identify inequivalent neighbor shells
# ============================================================
distances = np.linalg.norm(atoms.positions - vacancy_pos, axis=1)
sorted_idx = np.argsort(distances)

# Group by distance shells (0.2 A tolerance for FCC)
shells = {}
for idx in sorted_idx[:20]:
    d = distances[idx]
    assigned = False
    for shell_d in shells:
        if abs(d - shell_d) < 0.2:
            shells[shell_d].append(idx)
            assigned = True
            break
    if not assigned:
        shells[d] = [idx]

print("\nNeighbor shells around vacancy:")
shell_list = sorted(shells.items())
for i, (d, indices) in enumerate(shell_list[:4]):
    print(f"  Shell {i}: distance = {d:.3f} A, {len(indices)} atoms (e.g., atom {indices[0]})")


# ============================================================
# 3. ART helper functions (compact version)
# ============================================================
def hessian_vec(atoms_w, v, dr=0.01):
    v_n = v / np.linalg.norm(v)
    p0 = atoms_w.positions.copy()
    atoms_w.positions = p0 + dr * v_n
    fp = atoms_w.get_forces()
    atoms_w.positions = p0 - dr * v_n
    fm = atoms_w.get_forces()
    atoms_w.positions = p0
    return -(fp - fm) / (2.0 * dr)


def lowest_eigval(atoms_w, dr=0.01):
    n = len(atoms_w) * 3
    def mv(x):
        return hessian_vec(atoms_w, x.reshape(-1, 3), dr).flatten()
    op = LinearOperator((n, n), matvec=mv, dtype=float)
    try:
        vals, vecs = eigsh(op, k=1, which="SA", tol=1e-4)
        return vals[0], vecs[:, 0].reshape(-1, 3)
    except Exception:
        return 0.0, np.random.randn(len(atoms_w), 3)


def single_art_search(atoms_start, push_atom, push_dir, gamma=0.12,
                       push_step=0.12, max_push=25, max_saddle=100):
    """Compact ART search. Returns result dict or None."""
    atoms_w = atoms_start.copy()
    atoms_w.calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
    e0 = atoms_w.get_potential_energy()
    p0 = atoms_w.positions.copy()
    d = push_dir / np.linalg.norm(push_dir)

    # Push phase
    for step in range(max_push):
        atoms_w.positions[push_atom] += push_step * d
        f = atoms_w.get_forces()
        n_hat = d / np.linalg.norm(d)
        f_along = np.dot(f[push_atom], n_hat) * n_hat
        f[push_atom] -= (1.0 + gamma) * f_along
        for i in range(len(atoms_w)):
            if i != push_atom:
                fp = f[i] - np.dot(f[i], n_hat) * n_hat
                atoms_w.positions[i] += 0.03 * fp
        if step % 4 == 0 and step > 0:
            ev, evec = lowest_eigval(atoms_w)
            if ev < -0.3:
                d = evec[push_atom].copy()
                d /= np.linalg.norm(d)
                break
    else:
        return None

    # Saddle convergence
    for step in range(max_saddle):
        f = atoms_w.get_forces()
        ev, evec = lowest_eigval(atoms_w)
        if np.dot(evec[push_atom], d) < 0:
            evec = -evec
        ef = evec.flatten()
        ef /= np.linalg.norm(ef)
        ff = f.flatten()
        fp = np.dot(ff, ef) * ef
        fpp = ff - fp
        fmax = np.max(np.linalg.norm(f, axis=1))
        if fmax < 0.05 and ev < -0.1:
            break
        atoms_w.positions += (-0.04 * fp + 0.04 * fpp).reshape(-1, 3)

    e_saddle = atoms_w.get_potential_energy()
    barrier = e_saddle - e0
    if barrier < 0.01:
        return None

    # Relax to new minimum
    _, evec_s = lowest_eigval(atoms_w)
    if np.dot(evec_s[push_atom], atoms_w.positions[push_atom] - p0[push_atom]) < 0:
        evec_s = -evec_s
    atoms_f = atoms_w.copy()
    atoms_f.calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
    atoms_f.positions += 0.3 * evec_s
    BFGS(atoms_f, logfile=None).run(fmax=0.01, steps=300)
    e_final = atoms_f.get_potential_energy()

    disp = np.linalg.norm(atoms_f.positions - p0, axis=1)

    return {
        "barrier_fwd": float(barrier),
        "barrier_rev": float(e_saddle - e_final),
        "e_initial": float(e0),
        "e_saddle": float(e_saddle),
        "e_final": float(e_final),
        "max_disp": float(np.max(disp)),
        "is_new_state": float(np.max(disp)) > 0.5,
    }


# ============================================================
# 4. Systematic catalog: search from each shell
# ============================================================
N_SEARCHES_PER_ATOM = 6
catalog = []

for shell_idx, (shell_dist, atom_indices) in enumerate(shell_list[:3]):
    # Use one representative atom per shell
    atom_id = atom_indices[0]
    print(f"\nShell {shell_idx} (d = {shell_dist:.3f} A): atom {atom_id}")

    shell_results = []
    for s in range(N_SEARCHES_PER_ATOM):
        # Random direction biased toward vacancy
        d_rand = np.random.randn(3)
        d_vac = vacancy_pos - atoms.positions[atom_id]
        d_vac /= np.linalg.norm(d_vac)
        d = 0.4 * d_rand + 0.6 * d_vac
        d /= np.linalg.norm(d)

        r = single_art_search(atoms, atom_id, d)
        if r is not None:
            r["shell"] = shell_idx
            r["atom_id"] = atom_id
            r["shell_distance"] = float(shell_dist)
            r["search_idx"] = s
            shell_results.append(r)
            print(f"  Search {s}: barrier = {r['barrier_fwd']:.4f} eV, new_state = {r['is_new_state']}")
        else:
            print(f"  Search {s}: failed")

    catalog.extend(shell_results)

# ============================================================
# 5. Deduplicate and summarize
# ============================================================
print("\n" + "=" * 60)
print("Transition Catalog Summary")
print("=" * 60)

# Group by barrier value (within 0.05 eV tolerance = same transition)
unique_transitions = []
for r in catalog:
    found_match = False
    for ut in unique_transitions:
        if abs(r["barrier_fwd"] - ut["barrier_fwd"]) < 0.05:
            ut["count"] += 1
            found_match = True
            break
    if not found_match:
        unique_transitions.append({**r, "count": 1})

unique_transitions.sort(key=lambda x: x["barrier_fwd"])

print(f"\n{'#':<4} {'Shell':<8} {'Barrier (eV)':<14} {'Rev (eV)':<12} {'New state':<12} {'Count'}")
print("-" * 60)
for i, t in enumerate(unique_transitions):
    print(f"  {i:<4} {t['shell']:<8} {t['barrier_fwd']:<14.4f} "
          f"{t['barrier_rev']:<12.4f} {'Yes' if t['is_new_state'] else 'No':<12} {t['count']}")

# ============================================================
# 6. Compute transition rates (Arrhenius)
# ============================================================
kB = 8.617333262e-5  # eV/K
nu_0 = 1e13          # Hz, attempt frequency
T = 300               # K

print(f"\nTransition rates at {T} K (nu_0 = {nu_0:.0e} Hz):")
for i, t in enumerate(unique_transitions):
    rate = nu_0 * np.exp(-t["barrier_fwd"] / (kB * T))
    print(f"  Transition {i}: rate = {rate:.4e} Hz "
          f"(barrier = {t['barrier_fwd']:.4f} eV)")

# ============================================================
# 7. Visualization
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Barrier catalog
ax = axes[0]
colors = ["steelblue", "coral", "seagreen"]
for t in catalog:
    c = colors[t["shell"] % len(colors)]
    ax.scatter(t["shell_distance"], t["barrier_fwd"], c=c, s=60,
               edgecolors="black", alpha=0.7, zorder=3)
for i, (sd, _) in enumerate(shell_list[:3]):
    ax.axvline(x=sd, color=colors[i], linestyle=":", alpha=0.4)
ax.set_xlabel("Distance from vacancy ($\\AA$)", fontsize=12)
ax.set_ylabel("Activation energy (eV)", fontsize=12)
ax.set_title("ART Transition Catalog", fontsize=13)
ax.grid(True, alpha=0.3)

# Rate spectrum
ax = axes[1]
if unique_transitions:
    barriers_sorted = [t["barrier_fwd"] for t in unique_transitions]
    rates = [nu_0 * np.exp(-b / (kB * T)) for b in barriers_sorted]
    ax.barh(range(len(rates)), np.log10(np.array(rates) + 1e-100),
            color="steelblue", edgecolor="black")
    ax.set_xlabel("log$_{10}$(rate / Hz)", fontsize=12)
    ax.set_ylabel("Transition index", fontsize=12)
    ax.set_title(f"Transition Rates at {T} K", fontsize=13)
    ax.grid(axis="x", alpha=0.3)

fig.tight_layout()
fig.savefig("art_kmc_catalog.png", dpi=150)
print("\nCatalog plot saved to art_kmc_catalog.png")

with open("art_kmc_catalog.json", "w") as f:
    json.dump({
        "catalog": catalog,
        "unique_transitions": unique_transitions,
    }, f, indent=2, default=float)
print("Catalog saved to art_kmc_catalog.json")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `gamma` | 0.05--0.3 | Force reversal strength. Higher values push harder uphill. Start with 0.1--0.15. |
| `push_step` | 0.10--0.20 A | Initial displacement per step. Too large may overshoot; too small is slow. |
| `eigenvalue_threshold` | -0.3 to -1.0 eV/A^2 | Switch from push to saddle convergence. Less negative = earlier switch. |
| `saddle_force_tol` | 0.03--0.08 eV/A | Force convergence at saddle. Tighter is more accurate but slower. |
| `n_lanczos` (or `k` in eigsh) | 1--6 | Number of Hessian eigenvalues to compute. Usually only 1 is needed. |
| `lanczos_dr` | 0.005--0.02 A | Finite-difference step for Hessian-vector product. Smaller = more accurate but noisier. |
| `max_push_steps` | 20--50 | Maximum steps in push phase before giving up. |
| `max_art_steps` | 80--200 | Maximum steps for saddle convergence. |
| `n_random_searches` | 5--20 | Number of random directions to sample. More = better coverage. |
| Supercell size | >= 3x3x3 for cubic | Must be large enough to avoid periodic image interactions. |
| Calculator | MACE `medium` or `large` | Use `large` for better accuracy at higher cost. |

## Interpreting Results

1. **Activation energy (barrier)**: `E_a = E_saddle - E_initial`. This determines the rate of the transition via Arrhenius: `rate = nu_0 * exp(-E_a / kBT)`. Barriers below 0.5 eV are fast at room temperature; above 1.5 eV means negligible rate below ~1000 K.

2. **Number of negative eigenvalues**: A valid first-order saddle point has exactly **one** negative Hessian eigenvalue. Zero negatives means a minimum (not a saddle). Two or more indicate a higher-order saddle or poorly converged point.

3. **Forward vs reverse barrier**: If `E_a(fwd) != E_a(rev)`, the initial and final states have different energies (inequivalent sites). Equal barriers indicate equivalent sites (e.g., vacancy hopping between equivalent lattice positions).

4. **New state discovery**: If the relaxed final state has large atomic displacements (> 0.5 A) from the initial state, ART found a genuine transition to a new basin. If displacements are small, the system returned to the original basin.

5. **Barrier distribution**: Running many ART searches with random directions samples different transition pathways. The distribution of barriers reveals the energy landscape around the defect. Cluster barriers within ~0.05 eV to identify unique transition types.

6. **ART vs NEB agreement**: ART finds the saddle point directly, while NEB finds the minimum energy path. If both give similar barriers (within 0.1 eV), the ART saddle lies on the MEP. Larger differences suggest the ART saddle is on a different (possibly higher-energy) pathway.

7. **KMC input**: The catalog of transitions (barriers, pre-exponential factors, final states) can be used as input for kinetic Monte Carlo simulations to study long-time diffusion dynamics.

## Common Issues

| Problem | Solution |
|---|---|
| Push phase never finds a negative eigenvalue | Increase `max_push_steps`; increase `push_step`; reduce `eigenvalue_threshold` (less negative, e.g., -0.2); try a different atom or direction |
| Saddle convergence oscillates without converging | Reduce the step size in the saddle convergence phase (from 0.05 to 0.02); increase `max_art_steps`; tighten Lanczos tolerance |
| All searches return to the initial basin | The atom is not near a defect or the push is not strong enough. Increase `gamma` or `push_step`. Try atoms closer to the defect. |
| Very high barriers (> 5 eV) | Likely an artifact: the search went over a high ridge instead of finding a nearby saddle. Try shorter push distances or different directions. |
| Very low barriers (< 0.01 eV) | The system did not truly leave the initial basin. Increase the displacement before checking for negative eigenvalues. |
| Lanczos eigenvalue solver fails to converge | Increase `tol` (e.g., 1e-3), increase `maxiter`, or increase `lanczos_dr`. With MACE, very flat PES regions can cause numerical issues. |
| Multiple negative eigenvalues at "saddle" | The point is not a true first-order saddle. Continue optimization with more steps, or restart the search with a different direction. |
| Different results with different random seeds | Expected -- ART is stochastic. Run 10--20 searches to adequately sample the landscape. Report statistics (mean, min, distribution). |
| MACE gives unphysical energies | System outside MACE training data. Switch to a LAMMPS potential (EAM for metals) or DFT for validation. |
| Search is very slow | Reduce Lanczos eigenvalue checks (every 5 steps instead of 3). Use a smaller supercell for initial exploration. Use MACE `medium` instead of `large`. |
| Saddle point has forces > 0.1 eV/A | The saddle is not well converged. Run additional convergence steps with tighter tolerance, or use a finer step size for the gradient descent. |
