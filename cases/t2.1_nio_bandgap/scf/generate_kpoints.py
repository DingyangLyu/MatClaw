#!/usr/bin/env python3
import numpy as np

# High-symmetry points in fractional coordinates (crystal)
points = {
    'G': [0.0, 0.0, 0.0],
    'X': [0.5, 0.5, 0.0],
    'K': [0.375, 0.75, 0.375],
    'L': [0.5, 0.5, 0.5]
}
# Path segments
segments = [('G', 'X'), ('X', 'K'), ('K', 'G'), ('G', 'L')]
npoints_per_segment = 20

klist = []
for seg in segments:
    start = np.array(points[seg[0]])
    end = np.array(points[seg[1]])
    for i in range(npoints_per_segment):
        t = i / (npoints_per_segment - 1) if npoints_per_segment > 1 else 0
        k = start + t * (end - start)
        klist.append(k)
# Remove duplicates (junction points appear twice)
# We'll keep all for simplicity
print(len(klist))
for k in klist:
    print(f"{k[0]:.10f} {k[1]:.10f} {k[2]:.10f} 0.0")