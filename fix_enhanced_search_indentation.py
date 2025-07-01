#!/usr/bin/env python3
"""Fix indentation issues in enhanced search test."""

from pathlib import Path


def fix_indentation():
    """Fix all indentation issues in enhanced search test."""
    file_path = Path("tests/unit/test_enhanced_search.py")
    content = file_path.read_text()

    lines = content.split("\n")
    fixed_lines = []

    for i, line in enumerate(lines):
        # Fix specific problematic lines
        if line.strip() and line.startswith("            assert ") and i > 0:
            # These assert lines should have normal indent (8 spaces)
            fixed_lines.append("        " + line.strip())
        elif line.strip() == "# search_engine uses internal ranking":
            # This is a comment that should be indented
            fixed_lines.append("            # search_engine uses internal ranking")
        elif "engine.ranker.rank_results" in line and line.strip().startswith("engine."):
            # This should be indented inside the with block
            fixed_lines.append("            " + line.strip())
        else:
            fixed_lines.append(line)

    # Write back
    file_path.write_text("\n".join(fixed_lines))
    print(f"Fixed indentation in {file_path}")


def main():
    """Run fix."""
    fix_indentation()
    print("Indentation fixes completed!")


if __name__ == "__main__":
    main()
