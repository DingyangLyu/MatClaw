#!/usr/bin/env python3

with open("melting_loop.in", "r") as f:
    lines = f.readlines()

# Find variable T index line
for i, line in enumerate(lines):
    if line.startswith("variable T index"):
        lines[i] = "variable T index 300 400 500\n"
        break

# Reduce run steps from 2000 to 500 for test
for i, line in enumerate(lines):
    if "run 2000" in line:
        lines[i] = line.replace("2000", "500")
        break

with open("test.in", "w") as f:
    f.writelines(lines)

print("Created test.in")