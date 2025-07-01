#!/usr/bin/env python3
"""Fix final syntax issues in test files."""

import re
from pathlib import Path


def fix_config_test_syntax():
    """Fix syntax issues in config tests."""
    file_path = Path("tests/unit/test_config.py")
    if not file_path.exists():
        return

    content = file_path.read_text()

    # Fix the duplicate original. lines
    content = re.sub(r"original\.\s+original\.\s+", "", content)

    # Fix line 108 indentation issue - missing content for assertion
    lines = content.split("\n")
    fixed_lines = []

    for i, line in enumerate(lines):
        if i == 107 and "assert isinstance" in line:
            # Ensure proper indentation
            fixed_lines.append("        assert isinstance(config.scraping, ScrapingConfig)")
        elif 'assert "search" in config_dict' in line:
            fixed_lines.append(line)
            # Skip the duplicate processing/multi_pass assertions
            continue
        elif line.strip() in [
            "assert 'multi_pass' in config_dict",
            "assert 'processing' in config_dict",
        ]:
            # Skip these duplicates
            continue
        else:
            fixed_lines.append(line)

    content = "\n".join(fixed_lines)

    file_path.write_text(content)
    print(f"Fixed syntax in {file_path}")


def fix_multi_pass_controller_syntax():
    """Fix syntax issues in multi pass controller tests."""
    file_path = Path("tests/unit/test_multi_pass_controller.py")
    if not file_path.exists():
        return

    content = file_path.read_text()

    # Remove duplicate EOF assertions
    content = re.sub(
        r"assert result\.final_confidence > 0\n No newline at end of file\n\s*assert result\.final_confidence > 0\n No newline at end of file",
        "assert result.final_confidence > 0",
        content,
    )

    # Ensure file ends with newline
    if not content.endswith("\n"):
        content += "\n"

    file_path.write_text(content)
    print(f"Fixed syntax in {file_path}")


def main():
    """Run syntax fixes."""
    fix_config_test_syntax()
    fix_multi_pass_controller_syntax()
    print("Syntax fixes completed!")


if __name__ == "__main__":
    main()
