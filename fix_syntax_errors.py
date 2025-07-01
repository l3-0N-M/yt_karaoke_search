#!/usr/bin/env python3
"""Fix remaining syntax errors in test files."""

import re
from pathlib import Path


def fix_duckduckgo_provider():
    """Fix syntax in duckduckgo provider tests."""
    file_path = Path("tests/unit/test_duckduckgo_provider.py")
    content = file_path.read_text()

    lines = content.split("\n")
    fixed_lines = []

    for i, line in enumerate(lines):
        # Fix line 27 - should be part of class or properly indented
        if i == 26 and line.strip():  # Line 27 (0-indexed)
            # Check previous lines for context
            if i > 0 and "class" in lines[i - 1]:
                fixed_lines.append("    " + line.strip())
            elif i > 0 and "def" in lines[i - 1]:
                fixed_lines.append("        " + line.strip())
            else:
                # Default to method-level indentation
                fixed_lines.append("        " + line.strip())
        else:
            fixed_lines.append(line)

    file_path.write_text("\n".join(fixed_lines))
    print(f"Fixed {file_path}")


def fix_enhanced_search():
    """Fix syntax in enhanced search tests."""
    file_path = Path("tests/unit/test_enhanced_search.py")
    content = file_path.read_text()

    lines = content.split("\n")
    fixed_lines = []

    for i, line in enumerate(lines):
        # Fix line 22
        if i == 21 and line.strip():  # Line 22 (0-indexed)
            if "assert" in line or "return" in line:
                fixed_lines.append("        " + line.strip())
            else:
                fixed_lines.append("    " + line.strip())
        else:
            fixed_lines.append(line)

    file_path.write_text("\n".join(fixed_lines))
    print(f"Fixed {file_path}")


def fix_multi_pass_controller():
    """Fix syntax in multi pass controller tests."""
    file_path = Path("tests/unit/test_multi_pass_controller.py")
    content = file_path.read_text()

    lines = content.split("\n")
    fixed_lines = []

    for i, line in enumerate(lines):
        # Fix line 121 - seems to be a continuation or orphaned line
        if i == 120:  # Line 121 (0-indexed)
            if line.strip() == "return controller":
                # This should be indented as part of the fixture
                fixed_lines.append("            return controller")
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    file_path.write_text("\n".join(fixed_lines))
    print(f"Fixed {file_path}")


def fix_result_ranker():
    """Fix syntax in result ranker tests."""
    file_path = Path("tests/unit/test_result_ranker.py")
    content = file_path.read_text()

    # Fix the SearchResult with syntax error - missing closing parenthesis
    content = re.sub(
        r'SearchResult\(video_id="test1", url="https://youtube.com/watch\?v=test1", title="Artist Name - Song Title \(Karaoke Version, channel_id="test_channel", channel="Test Channel"\)"\)',
        'SearchResult(video_id="test1", url="https://youtube.com/watch?v=test1", title="Artist Name - Song Title (Karaoke Version)", channel_id="test_channel", channel="Test Channel")',
        content,
    )

    # Fix any other malformed SearchResult calls
    def fix_search_result_syntax(match):
        text = match.group(0)
        # Count parentheses
        open_count = text.count("(")
        close_count = text.count(")")

        if open_count > close_count:
            # Missing closing parentheses
            return text + ")" * (open_count - close_count)
        return text

    # Apply fix to all SearchResult instances
    content = re.sub(r"SearchResult\([^;]+\)", fix_search_result_syntax, content, flags=re.DOTALL)

    file_path.write_text(content)
    print(f"Fixed {file_path}")


def main():
    """Run all syntax fixes."""
    fix_duckduckgo_provider()
    fix_enhanced_search()
    fix_multi_pass_controller()
    fix_result_ranker()
    print("\nSyntax fixes completed!")


if __name__ == "__main__":
    main()
