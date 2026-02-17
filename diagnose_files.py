#!/usr/bin/env python3
"""
Diagnostic script to check what files are present and what would be discovered.
"""
import os
import glob

# Check the main workspace directory
main_folder = r"c:\Users\ofirn\OneDrive\Documents\Work\ConsultingServices\Outsense\Python\EstimateUnifiedFOV"

print(f"Scanning: {main_folder}")
print()

# Look for subdirectories that might contain images
subdirs = [d for d in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, d)) and "output" in d.lower()]
print(f"Found {len(subdirs)} output directories")
for d in subdirs[:5]:
    print(f"  - {d}")
print()

# Check for image files in main folder
all_images = [f for f in os.listdir(main_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
g_files = [f for f in all_images if f.lower().endswith('-g-00-org.png')]
e_files = [f for f in all_images if f.lower().endswith('-e-00-org.png')]
default_files = [f for f in all_images if f.lower().endswith('-00-org.png') and not f.lower().endswith('-g-00-org.png') and not f.lower().endswith('-e-00-org.png')]

print(f"Files in main folder: {len(all_images)} total")
print(f"  G variant (-g-00-org.png): {len(g_files)}")
print(f"  E variant (-e-00-org.png): {len(e_files)}")
print(f"  Default variant (-00-org.png): {len(default_files)}")
print()

if g_files:
    print("Sample G files:")
    for f in g_files[:3]:
        print(f"  {f}")
else:
    print("No G files found in main folder")
    
if e_files:
    print("\nSample E files:")
    for f in e_files[:3]:
        print(f"  {f}")
else:
    print("No E files found in main folder")

if default_files:
    print("\nSample Default files:")
    for f in default_files[:3]:
        print(f"  {f}")
else:
    print("No Default files found in main folder")

# Check subdirectories for images
print("\n" + "="*60)
print("Checking image subdirectories...")
for subdir in subdirs[:3]:
    path = os.path.join(main_folder, subdir)
    try:
        images = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        g = [f for f in images if f.lower().endswith('-g-00-org.png')]
        e = [f for f in images if f.lower().endswith('-e-00-org.png')]
        d = [f for f in images if f.lower().endswith('-00-org.png') and not f.lower().endswith(('-g-00-org.png', '-e-00-org.png'))]
        print(f"\n{subdir}: {len(images)} images total")
        print(f"  G: {len(g)}, E: {len(e)}, D: {len(d)}")
    except Exception as ex:
        print(f"\n{subdir}: Error - {ex}")
