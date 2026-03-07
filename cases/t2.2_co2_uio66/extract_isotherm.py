#!/usr/bin/env python3
import os
import re
import json

pressures = [0.1, 1, 5, 10]  # bar
base_dir = '.'

data = []
for p in pressures:
    dirname = f'{p}bar'
    txt_file = None
    # find the output txt file
    output_dir = os.path.join(base_dir, dirname, 'output')
    if not os.path.exists(output_dir):
        print(f'Warning: {output_dir} not found')
        continue
    for f in os.listdir(output_dir):
        if f.endswith('.txt'):
            txt_file = os.path.join(output_dir, f)
            break
    if not txt_file:
        print(f'Warning: no txt file in {output_dir}')
        continue
    # parse the txt file
    with open(txt_file, 'r') as fp:
        content = fp.read()
    # find absolute loading average line (molecules per unit cell)
    # pattern: "Abs. loading average   1.956000e+00 +/-  2.357006e-01 [molecules/uc]"
    pattern = r'Abs\. loading average\s+([\d\.eE+-]+)\s+\+/-.*\[molecules/uc\]'
    match = re.search(pattern, content)
    if match:
        loading = float(match.group(1))
        # find uncertainty (same line)
        uncertainty_pattern = r'Abs\. loading average\s+[\d\.eE+-]+\s+\+/-.*?([\d\.eE+-]+)\s+\[molecules/uc\]'
        unc_match = re.search(uncertainty_pattern, content)
        uncertainty = float(unc_match.group(1)) if unc_match else 0.0
        data.append((p, loading, uncertainty))
        print(f'Pressure {p} bar: {loading} +/- {uncertainty} molecules/uc')
    else:
        # maybe not yet finished, try to find any loading line
        print(f'Pressure {p} bar: no absolute loading found')

# print table
print('\nAdsorption Isotherm for CO2 in UiO-66 at 298 K')
print('Pressure (bar) | Loading (molecules/uc) | Uncertainty')
print('------------------------------------------------------')
for p, load, unc in data:
    print(f'{p:13.1f} | {load:20.4e} | {unc:.4e}')

# optional: save to CSV
import csv
with open('adsorption_isotherm.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Pressure (bar)', 'Loading (molecules/uc)', 'Uncertainty'])
    for row in data:
        writer.writerow(row)
print('Results saved to adsorption_isotherm.csv')

# also compute in mg/g if needed
# from test output: Abs. loading average   1.296087e+01 +/-  1.561802e+00 [mg/g-framework]
# we can extract that as well
for p in pressures:
    dirname = f'{p}bar'
    txt_file = None
    output_dir = os.path.join(base_dir, dirname, 'output')
    if not os.path.exists(output_dir):
        continue
    for f in os.listdir(output_dir):
        if f.endswith('.txt'):
            txt_file = os.path.join(output_dir, f)
            break
    if txt_file:
        with open(txt_file, 'r') as fp:
            content = fp.read()
        pattern = r'Abs\. loading average\s+([\d\.eE+-]+)\s+\+/-.*\[mg/g-framework\]'
        match = re.search(pattern, content)
        if match:
            mg_g = float(match.group(1))
            print(f'Pressure {p} bar: {mg_g} mg/g')