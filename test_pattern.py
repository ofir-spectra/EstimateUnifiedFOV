#!/usr/bin/env python3
"""Test the regex pattern for filtering default files."""
import re

test_files = [
    "c32_3210_00_0001-00-org.png",  # Should match (digit before -00)
    "ex-32_3210_00_0001-g-00-org.png",  # Should NOT match (has -g-)
    "ex-32_3210_00_0001-e-00-org.png",  # Should NOT match (has -e-)
    "ex-32_3210_00_0001-k-00-org.png",  # Should NOT match (has -k-)
    "ex-32_3210_00_0001-t-00-org.png",  # Should NOT match (has -t-)
    "ex-32_3210_00_0001-x-00-org.png",  # Should NOT match (has -x-)
    "ex-32_3210_00_0001-zsl-00-org.png",  # Should NOT match (has -zsl-)
    "c32_3210_00_0001-01-org.png",  # Should NOT match (ends with -01, not -00)
]

pattern = r'\d-00-org\.png$'

print("Testing pattern: " + pattern)
print()

for f in test_files:
    matches = bool(re.search(pattern, f.lower()))
    status = "✓ MATCH" if matches else "✗ NO MATCH"
    print(f"{status}: {f}")
