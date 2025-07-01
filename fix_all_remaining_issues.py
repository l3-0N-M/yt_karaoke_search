#!/usr/bin/env python3
"""Fix all remaining test issues comprehensively."""

import re
from pathlib import Path


def fix_enhanced_search_indentation():
    """Fix indentation issues in enhanced search tests."""
    file_path = Path("tests/unit/test_enhanced_search.py")
    content = file_path.read_text()

    # Fix the specific indentation issue at lines 22-23
    lines = content.split("\n")
    fixed_lines = []

    for i, line in enumerate(lines):
        if i == 21 and "config.use_multi_strategy = True" in line:
            # This should be indented inside the function
            fixed_lines.append("        config.use_multi_strategy = True")
        elif i == 22 and "return config" in line:
            # This should also be indented inside the function
            fixed_lines.append("        return config")
        else:
            fixed_lines.append(line)

    file_path.write_text("\n".join(fixed_lines))
    print(f"Fixed indentation in {file_path}")


def fix_result_ranker_comprehensive():
    """Comprehensively fix result ranker tests."""
    file_path = Path("tests/unit/test_result_ranker.py")
    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return

    content = file_path.read_text()

    # Fix SearchResult calls with proper syntax
    # Pattern to match SearchResult with potential syntax issues
    def fix_search_result_call(match):
        call = match.group(0)

        # Extract parameters
        params_start = call.find("(") + 1
        params_end = call.rfind(")")
        params = call[params_start:params_end]

        # Parse and fix parameters
        fixed_params = []
        current_param = ""
        in_quotes = False
        paren_depth = 0

        for char in params:
            if char == '"' and (not current_param or current_param[-1] != "\\"):
                in_quotes = not in_quotes
            elif char == "(" and not in_quotes:
                paren_depth += 1
            elif char == ")" and not in_quotes:
                paren_depth -= 1
            elif char == "," and not in_quotes and paren_depth == 0:
                if current_param.strip():
                    fixed_params.append(current_param.strip())
                current_param = ""
                continue

            current_param += char

        if current_param.strip():
            fixed_params.append(current_param.strip())

        # Ensure all required fields are present
        param_dict = {}
        for param in fixed_params:
            if "=" in param:
                key, value = param.split("=", 1)
                param_dict[key.strip()] = value.strip()

        # Add missing required fields
        required_fields = {
            "video_id": '"test_video"',
            "url": '"https://youtube.com/watch?v=test"',
            "title": '"Test Title"',
            "channel": '"Test Channel"',
            "channel_id": '"channel123"',
        }

        for field, default in required_fields.items():
            if field not in param_dict:
                param_dict[field] = default

        # Reconstruct the call
        new_params = ", ".join(f"{k}={v}" for k, v in param_dict.items())
        return f"SearchResult({new_params})"

    # Apply fixes
    content = re.sub(r"SearchResult\([^)]*\)", fix_search_result_call, content)

    file_path.write_text(content)
    print(f"Fixed SearchResult calls in {file_path}")


def fix_config_test_final():
    """Final fixes for config tests."""
    file_path = Path("tests/unit/test_config.py")
    content = file_path.read_text()

    # Fix any remaining EOF issues
    if not content.endswith("\n"):
        content += "\n"

    # Remove duplicate lines at end
    lines = content.split("\n")
    while len(lines) > 1 and lines[-1] == lines[-2]:
        lines.pop()

    content = "\n".join(lines)

    file_path.write_text(content)
    print(f"Fixed final issues in {file_path}")


def fix_multi_pass_final():
    """Final fixes for multi pass controller tests."""
    file_path = Path("tests/unit/test_multi_pass_controller.py")
    content = file_path.read_text()

    # Remove duplicate assertions at end
    lines = content.split("\n")
    if len(lines) > 2:
        # Check for duplicate final assertions
        if (
            "assert result.final_confidence > 0" in lines[-1]
            and "assert result.final_confidence > 0" in lines[-2]
        ):
            lines.pop()

    # Ensure proper EOF
    while lines and not lines[-1].strip():
        lines.pop()
    lines.append("")  # Add single empty line at end

    content = "\n".join(lines)
    file_path.write_text(content)
    print(f"Fixed final issues in {file_path}")


def fix_all_import_errors():
    """Fix any remaining import errors."""
    test_files = list(Path("tests/unit").glob("test_*.py"))

    for file_path in test_files:
        content = file_path.read_text()
        modified = False

        # Ensure asyncio is imported where needed
        if "asyncio." in content and "import asyncio" not in content:
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("import") or line.startswith("from"):
                    lines.insert(i + 1, "import asyncio")
                    modified = True
                    break
            content = "\n".join(lines)

        # Fix missing PathLike imports
        if "PathLike" in content and "from os import PathLike" not in content:
            content = "from os import PathLike\n" + content
            modified = True

        if modified:
            file_path.write_text(content)
            print(f"Fixed imports in {file_path}")


def main():
    """Run all remaining fixes."""
    print("Fixing all remaining test issues...")

    fix_enhanced_search_indentation()
    fix_result_ranker_comprehensive()
    fix_config_test_final()
    fix_multi_pass_final()
    fix_all_import_errors()

    print("\nAll remaining fixes completed!")


if __name__ == "__main__":
    main()
