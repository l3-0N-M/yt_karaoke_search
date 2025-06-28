#!/usr/bin/env python3
"""Debug channel template patterns."""

import re

# Test the specific simple_dash pattern that should match KaraFun titles
test_title = "Karaoke Gib mir sonne - Rosenstolz *"
channel_name = "KaraFun Deutschland"

# The pattern from the channel template pass
pattern = r"^([^-–—]+)\s*[-–—]\s*([^(\[]+)"

print(f"Testing pattern: {pattern}")
print(f"Against title: '{test_title}'")
print()

match = re.search(pattern, test_title, re.IGNORECASE | re.UNICODE)

if match:
    print("MATCH!")
    print(f"Group 1: '{match.group(1)}'")
    print(f"Group 2: '{match.group(2)}'")
    print()
    print("With fixed group assignments (artist_group=2, title_group=1):")
    print(f"Artist (Group 2): '{match.group(2).strip()}'")
    print(f"Song (Group 1): '{match.group(1).strip()}'")
else:
    print("NO MATCH")
    
print()
print("Checking if channel is detected as karaoke channel:")
channel_lower = channel_name.lower()
karaoke_indicators = [
    "karaoke", "karaoké", "karaokê", "караоке", "sing along", 
    "backing track", "instrumental", "covers", "tribute", "piano version"
]

is_karaoke_channel = any(indicator in channel_lower for indicator in karaoke_indicators)
print(f"Channel: '{channel_name}'")
print(f"Is karaoke channel: {is_karaoke_channel}")

for indicator in karaoke_indicators:
    if indicator in channel_lower:
        print(f"  Matched indicator: '{indicator}'")
        break