#!/usr/bin/env python3
import os, re, json, csv

pressures = [0.1, 1, 5, 10]
base = '.'

rows = []
for p in pressures:
    dirname = f'{p}bar'
    txt = None
    outdir = os.path.join(base, dirname, 'output')
    if not os.path.exists(outdir):
        rows.append([p, None, None, None, None, None, None])
        continue
    for f in os.listdir(outdir):
        if f.endswith('.txt'):
            txt = os.path.join(outdir, f)
            break
    if not txt:
        rows.append([p, None, None, None, None, None, None])
        continue
    with open(txt, 'r') as fp:
        content = fp.read()
    # Absolute loading molecules/uc
    abs_uc = re.search(r'Abs\. loading average\s+([\d\.eE+-]+)\s+\+/-.*\[molecules/uc\]', content)
    abs_uc_err = re.search(r'Abs\. loading average\s+[\d\.eE+-]+\s+\+/-.*?([\d\.eE+-]+)\s+\[molecules/uc\]', content)
    # Absolute loading mol/kg
    abs_molkg = re.search(r'Abs\. loading average\s+([\d\.eE+-]+)\s+\+/-.*\[mol/kg-framework\]', content)
    abs_molkg_err = re.search(r'Abs\. loading average\s+[\d\.eE+-]+\s+\+/-.*?([\d\.eE+-]+)\s+\[mol/kg-framework\]', content)
    # Absolute loading mg/g
    abs_mgg = re.search(r'Abs\. loading average\s+([\d\.eE+-]+)\s+\+/-.*\[mg/g-framework\]', content)
    abs_mgg_err = re.search(r'Abs\. loading average\s+[\d\.eE+-]+\s+\+/-.*?([\d\.eE+-]+)\s+\[mg/g-framework\]', content)
    # Excess loading molecules/uc
    ex_uc = re.search(r'Excess loading average\s+([\d\.eE+-]+)\s+\+/-.*\[molecules/uc\]', content)
    ex_uc_err = re.search(r'Excess loading average\s+[\d\.eE+-]+\s+\+/-.*?([\d\.eE+-]+)\s+\[molecules/uc\]', content)

    row = [p]
    if abs_uc:
        row.append(float(abs_uc.group(1)))
        row.append(float(abs_uc_err.group(1)) if abs_uc_err else 0.0)
    else:
        row.extend([None, None])
    if abs_molkg:
        row.append(float(abs_molkg.group(1)))
        row.append(float(abs_molkg_err.group(1)) if abs_molkg_err else 0.0)
    else:
        row.extend([None, None])
    if abs_mgg:
        row.append(float(abs_mgg.group(1)))
        row.append(float(abs_mgg_err.group(1)) if abs_mgg_err else 0.0)
    else:
        row.extend([None, None])
    if ex_uc:
        row.append(float(ex_uc.group(1)))
        row.append(float(ex_uc_err.group(1)) if ex_uc_err else 0.0)
    else:
        row.extend([None, None])
    rows.append(row)

# Print table
print('CO2 Adsorption in UiO-66 at 298 K')
print('='*80)
print('Pressure | Abs. loading (molecules/uc) ± err | Abs. loading (mol/kg) ± err | Abs. loading (mg/g) ± err | Excess loading (molecules/uc) ± err')
print('-'*80)
for row in rows:
    p = row[0]
    abs_uc = row[1]; abs_uc_err = row[2]
    abs_molkg = row[3]; abs_molkg_err = row[4]
    abs_mgg = row[5]; abs_mgg_err = row[6]
    ex_uc = row[7]; ex_uc_err = row[8]
    if abs_uc is None:
        print(f'{p:8.1f} | Not available yet')
    else:
        print(f'{p:8.1f} | {abs_uc:9.4e} ± {abs_uc_err:9.4e} | {abs_molkg:9.4e} ± {abs_molkg_err:9.4e} | {abs_mgg:9.4e} ± {abs_mgg_err:9.4e} | {ex_uc:9.4e} ± {ex_uc_err:9.4e}')

# Save to CSV
with open('adsorption_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Pressure (bar)', 'Abs_loading_molecules_per_uc', 'Abs_err', 'Abs_loading_mol_per_kg', 'Abs_err', 'Abs_loading_mg_per_g', 'Abs_err', 'Excess_loading_molecules_per_uc', 'Excess_err'])
    writer.writerows(rows)
print('\nResults saved to adsorption_results.csv')

# Check if all pressures are available
if all(r[1] is not None for r in rows):
    print('\nAll simulations completed. Isotherm data ready.')
else:
    print('\nSome simulations still running. Missing pressures:', [p for p, r in zip(pressures, rows) if r[1] is None])