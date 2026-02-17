#!/usr/bin/env python3
"""Test the fallback logic for variant file selection."""
import re

# Simulate files in folder
test_files = [
    # ex-32 files - has g and e variants
    "ex-32_3210_00_0001-g-00-org.png",
    "ex-32_3210_00_0001-e-00-org.png",
    "ex-32_3210_00_0001-k-00-org.png",  # Should be excluded (k variant)
    # c32 files - only has default (no g or e)
    "c32_3210_00_0001-00-org.png",
    "c32_3210_00_0002-00-org.png",
    # Hypothetical case: if ex-32 HAD a default, it should be excluded
    "ex-32_3210_00_0001-00-org.png",  # Should be excluded (g/e exist)
]

# Apply the logic
g_files = [f for f in test_files if f.lower().endswith('-g-00-org.png')]
e_files = [f for f in test_files if f.lower().endswith('-e-00-org.png')]

# Extract base filenames from g and e variants
variant_bases = set()
for f in g_files:
    variant_bases.add(f.replace('-g-00-org.png', '').replace('-G-00-ORG.PNG', ''))
for f in e_files:
    variant_bases.add(f.replace('-e-00-org.png', '').replace('-E-00-ORG.PNG', ''))

# For default, check fallback logic
default_files = []
for f in test_files:
    if re.search(r'\d-00-org\.png$', f.lower()):
        # Extract base filename for this default file
        base = re.sub(r'-00-org\.png$', '', f, flags=re.IGNORECASE)
        if base not in variant_bases:
            default_files.append(f)

image_files = g_files + e_files + default_files

print("G variants:")
for f in g_files:
    print(f"  ✓ {f}")

print("\nE variants:")
for f in e_files:
    print(f"  ✓ {f}")

print("\nDefault variants (fallback - only if no g/e):")
for f in default_files:
    print(f"  ✓ {f}")

print("\nBase filenames with g/e variants (default excluded):")
for base in sorted(variant_bases):
    print(f"  - {base}")

print(f"\nTotal files to process: {len(image_files)}")
print("\nExpected behavior:")
print("  ✓ ex-32_3210_00_0001: Process g and e variants ONLY (not default)")
print("  ✓ c32_3210_00_0001: Process default (no g/e exist)")
print("  ✓ c32_3210_00_0002: Process default (no g/e exist)")
print("  ✗ -k-00-org.png: Excluded (not g, e, or digit-default)")
