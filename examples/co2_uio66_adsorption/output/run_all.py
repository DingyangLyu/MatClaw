import subprocess, os, json

pressures_bar = [0.1, 1.0, 5.0, 10.0]
pressures_Pa  = [p * 1e5 for p in pressures_bar]

base_dir = "/workspace/group/uio66_co2"

for P_bar, P_Pa in zip(pressures_bar, pressures_Pa):
    label = f"P{P_bar:.1f}bar".replace(".", "p")
    run_dir = os.path.join(base_dir, label)
    os.makedirs(run_dir, exist_ok=True)

    # Symlink shared files
    for f in ["UIO-66.cif", "force_field.json", "CO2.json"]:
        dst = os.path.join(run_dir, f)
        if not os.path.exists(dst):
            os.symlink(os.path.join(base_dir, f), dst)

    sim = {
        "SimulationType": "MonteCarlo",
        "NumberOfCycles": 50000,
        "NumberOfInitializationCycles": 10000,
        "PrintEvery": 5000,
        "Systems": [{
            "Type": "Framework",
            "Name": "UIO-66",
            "NumberOfUnitCells": [1, 1, 1],
            "HeliumVoidFraction": 0.47,
            "ChargeMethod": "Ewald",
            "ExternalTemperature": 298.0,
            "ExternalPressure": P_Pa
        }],
        "Components": [{
            "Name": "CO2",
            "IdealGasRosenbluthWeight": 1.0,
            "TranslationProbability": 0.5,
            "RotationProbability": 0.5,
            "ReinsertionProbability": 0.5,
            "SwapProbability": 1.0,
            "WidomProbability": 0.0,
            "CreateNumberOfMolecules": 0
        }]
    }

    with open(os.path.join(run_dir, "simulation.json"), "w") as f:
        json.dump(sim, f, indent=2)

    print(f"Running P = {P_bar} bar ({P_Pa:.0f} Pa) in {run_dir} ...")
    result = subprocess.run(["raspa3"], cwd=run_dir,
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-500:]}")
    else:
        print(f"  Done.")

print("All pressures complete.")
