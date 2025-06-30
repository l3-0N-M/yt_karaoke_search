#!/usr/bin/env python3
"""Script to update field names from old schema to new optimized schema."""

import os
from pathlib import Path


def update_file_field_names(file_path: Path):
    """Update field names in a single file."""
    print(f"Updating {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content

    # Update original_artist to artist (but not in ParseResult class definition)
    # Skip lines that are property definitions
    lines = content.split("\n")
    updated_lines = []

    for line in lines:
        # Skip property definitions and comments about the field
        if (
            "def original_artist" in line
            or "@original_artist.setter" in line
            or "renamed from original_artist" in line
            or "Backward compatibility" in line
        ):
            updated_lines.append(line)
        elif "original_artist=" in line:
            updated_lines.append(line.replace("original_artist=", "artist="))
        elif ".original_artist" in line:
            updated_lines.append(line.replace(".original_artist", ".artist"))
        elif "original_artist," in line:
            updated_lines.append(line.replace("original_artist,", "artist,"))
        elif '"original_artist"' in line:
            updated_lines.append(line.replace('"original_artist"', '"artist"'))
        else:
            updated_lines.append(line)

    content = "\n".join(updated_lines)

    # Additional replacements for SQL fields
    content = content.replace("like_dislike_to_views_ratio", "engagement_ratio")
    content = content.replace("estimated_release_year", "release_year")
    content = content.replace("musicbrainz_confidence", "parse_confidence")

    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  ✅ Updated {file_path}")
        return True
    else:
        print(f"  ➖ No changes needed for {file_path}")
        return False


def main():
    """Update field names in all Python files."""

    base_dir = Path("collector")
    files_to_update = []

    # Find all Python files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file
                files_to_update.append(file_path)

    updated_count = 0

    for file_path in sorted(files_to_update):
        # Skip the advanced_parser.py as we already updated it
        if file_path.name == "advanced_parser.py":
            continue

        if update_file_field_names(file_path):
            updated_count += 1

    print(f"\n🎉 Updated {updated_count} files")


if __name__ == "__main__":
    main()
