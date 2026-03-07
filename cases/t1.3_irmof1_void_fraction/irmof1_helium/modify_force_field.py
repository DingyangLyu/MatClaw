#!/usr/bin/env python3
import json
import sys

with open('force_field.json', 'r') as f:
    data = json.load(f)

# Add helium pseudo atom
he_pseudo = {
    "name": "He",
    "framework": False,
    "print_to_output": True,
    "element": "He",
    "print_as": "He",
    "mass": 4.0026,
    "charge": 0.0,
    "source": "TraPPE"
}
data['PseudoAtoms'].append(he_pseudo)

# Add helium self-interaction
he_self = {
    "name": "He",
    "type": "lennard-jones",
    "parameters": [10.22, 2.58],  # epsilon/K, sigma/A
    "source": "TraPPE"
}
data['SelfInteractions'].append(he_self)

with open('force_field.json', 'w') as f:
    json.dump(data, f, indent=2)
print("Modified force_field.json")