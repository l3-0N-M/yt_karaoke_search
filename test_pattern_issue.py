#!/usr/bin/env python3
"""Test script to identify the exact pattern issue."""

import re

# The problematic pattern from processor.py line 494-497
pattern = r"^([^-–—]+?)\s*[-–—]\s*([^(\[]+)(?:\s*[\(\[][^)\]]*[Kk]araoke[^)\]]*[\)\]])?"

# Test titles from the database
test_titles = [
    "Karaoke - What Child Is This - Christmas Traditional",
    "Karaoke - Since I Fell For You - Charlie Rich",
    "Karaoke - Amigos Para Siempre(Friends For Life) - Sarah Brightman & Jose Carreras"
]

print("Testing the problematic pattern:")
print(f"Pattern: {pattern}")
print("=" * 80)

for title in test_titles:
    print(f"\nTitle: '{title}'")
    match = re.search(pattern, title, re.IGNORECASE | re.UNICODE)

    if match:
        print(f"MATCH! Groups: {match.groups()}")
        print(f"Group 1: '{match.group(1)}'")
        print(f"Group 2: '{match.group(2)}'")

        # This pattern has artist_group=2, title_group=1
        # So song_title would be match.group(1) and original_artist would be match.group(2)
        print("Would extract as:")
        print(f"  song_title: '{match.group(1)}'")
        print(f"  original_artist: '{match.group(2)}'")
    else:
        print("No match")

print("\n" + "=" * 80)
print("The issue: Group 2 has [^([]+ which should match multiple chars, but it's only getting 1.")
print("Let's test the regex parts separately:")

test_title = "Karaoke - What Child Is This - Christmas Traditional"
print(f"\nTesting parts of: '{test_title}'")

# Test the second capture group pattern
second_group_pattern = r"([^(\[]+)"
test_text = "What Child Is This - Christmas Traditional"
match = re.search(second_group_pattern, test_text)
if match:
    print(f"Second group pattern matches: '{match.group(1)}'")
else:
    print("Second group pattern doesn't match")

# Test the non-greedy quantifier issue
print("\nTesting greedy vs non-greedy:")
greedy_pattern = r"^([^-–—]+)\s*[-–—]\s*([^(\[]+)"
non_greedy_pattern = r"^([^-–—]+?)\s*[-–—]\s*([^(\[]+?)"

for pattern_name, pattern in [("Greedy", greedy_pattern), ("Non-greedy", non_greedy_pattern)]:
    match = re.search(pattern, test_title)
    if match:
        print(f"{pattern_name:10}: Group1='{match.group(1)}', Group2='{match.group(2)}'")
