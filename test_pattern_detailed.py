#!/usr/bin/env python3
"""More detailed test to understand the regex issue."""

import re

# The actual pattern from the code with all parts
full_pattern = r"^([^-–—]+?)\s*[-–—]\s*([^(\[]+)(?:\s*[\(\[][^)\]]*[Kk]araoke[^)\]]*[\)\]])?"

# Let's break it down step by step
test_title = "Karaoke - What Child Is This - Christmas Traditional"

print(f"Full title: '{test_title}'")
print(f"Full pattern: {full_pattern}")
print()

# Test each part
parts = [
    ("Part 1: ^([^-–—]+?)", r"^([^-–—]+?)"),
    ("Part 2: \\s*[-–—]\\s*", r"\s*[-–—]\s*"),
    ("Part 3: ([^(\\[]+)", r"([^(\[]+)"),
    ("Part 4: optional karaoke", r"(?:\s*[\(\[][^)\]]*[Kk]araoke[^)\]]*[\)\]])?"),
]

print("Testing each part separately:")
remaining_text = test_title
pos = 0

for part_name, part_pattern in parts:
    print(f"\n{part_name}: {part_pattern}")
    print(f"Testing against: '{remaining_text[pos:]}'")

    match = re.match(part_pattern, remaining_text[pos:])
    if match:
        matched_text = match.group(0)
        print(f"Matches: '{matched_text}'")
        if match.groups():
            for i, group in enumerate(match.groups(), 1):
                print(f"  Group {i}: '{group}'")
        pos += len(matched_text)
    else:
        print("No match")
        break

print("\nFull pattern test:")
match = re.search(full_pattern, test_title)
if match:
    print(f"Groups: {match.groups()}")
    for i, group in enumerate(match.groups(), 1):
        print(f"Group {i}: '{group}' (length: {len(group)})")

# Now let's test what happens when we process this through the validation
print("\nWhat happens after _clean_extracted_text:")


def _clean_extracted_text(text):
    """Simulated version of the cleaning function."""
    if not text:
        return ""

    # Remove extra quotes and brackets
    cleaned = re.sub(r'^["\'`]+|["\'`]+$', "", text.strip())
    cleaned = re.sub(r"^\([^)]*\)|^\[[^\]]*\]", "", cleaned).strip()

    # Remove trailing noise
    noise_patterns = [
        r"\s*\([^)]*(?:[Kk]araoke|[Ii]nstrumental|[Mm]inus|[Mm][Rr])[^)]*\)$",
        r"\s*\[[^\]]*(?:[Kk]araoke|[Ii]nstrumental|[Mm]inus|[Mm][Rr])[^\]]*\]$",
        r"\s*-\s*[Kk]araoke.*$",
        r"\s*[Mm][Rr]$",
        r"\s*[Ii]nst\.?$",
        r"\s*\([^)]*[Kk]ey\)$",
    ]

    for pattern in noise_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    # Clean up whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


if match:
    print(f"Raw group 1: '{match.group(1)}'")
    print(f"Cleaned group 1: '{_clean_extracted_text(match.group(1))}'")
    print(f"Raw group 2: '{match.group(2)}'")
    print(f"Cleaned group 2: '{_clean_extracted_text(match.group(2))}'")
