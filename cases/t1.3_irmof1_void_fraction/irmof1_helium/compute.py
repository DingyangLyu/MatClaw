#!/usr/bin/env python3
import json
import math

# Values from simulation output
void_fraction = 0.8030146
void_fraction_error = 0.005516825
cell_length = 25.83200  # Å
cell_volume = cell_length ** 3  # Å^3
cell_volume_cm3 = cell_volume * 1e-24

# Mass of unit cell (from JSON output)
mass_per_unitcell = 6158.5942400000185  # g/mol? Actually in internal units, but likely g/mol
avogadro = 6.02214076e23
mass_grams = mass_per_unitcell / avogadro  # grams per unit cell

# Pore volume per unit cell
pore_volume_ang3 = void_fraction * cell_volume
pore_volume_cm3 = pore_volume_ang3 * 1e-24

# Pore volume per gram (cm^3/g)
pore_volume_per_gram = pore_volume_cm3 / mass_grams

# Also compute pore volume per cm^3 of crystal (void fraction)
pore_volume_per_cm3 = void_fraction  # cm^3 pore per cm^3 crystal

print("Helium-accessible void fraction:", void_fraction, "+/-", void_fraction_error)
print("Unit cell volume (Å^3):", cell_volume)
print("Unit cell volume (cm^3):", cell_volume_cm3)
print("Mass per unit cell (g/mol):", mass_per_unitcell)
print("Mass per unit cell (g):", mass_grams)
print("Pore volume per unit cell (Å^3):", pore_volume_ang3)
print("Pore volume per unit cell (cm^3):", pore_volume_cm3)
print("Pore volume per gram (cm^3/g):", pore_volume_per_gram)
print("Pore volume per cm^3 crystal (cm^3/cm^3):", pore_volume_per_cm3)

# Error propagation
pore_volume_per_gram_error = pore_volume_per_gram * (void_fraction_error / void_fraction)
print("Pore volume per gram error (approx):", pore_volume_per_gram_error)