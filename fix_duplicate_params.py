#!/usr/bin/env python3
"""Fix duplicate parameter issues in tests."""

import re
from pathlib import Path


def fix_duplicate_params():
    """Fix duplicate parameters in SearchResult calls."""
    file_path = Path("tests/unit/test_result_ranker.py")
    content = file_path.read_text()

    # Find all SearchResult calls with duplicate url parameters
    pattern = r'SearchResult\([^)]*url="[^"]*"[^)]*url="[^"]*"[^)]*\)'

    def fix_duplicate_url(match):
        text = match.group(0)
        # Keep only the last url parameter
        parts = text.split(",")
        seen_url = False
        fixed_parts = []

        for part in parts:
            if "url=" in part and seen_url:
                # Skip duplicate url
                continue
            elif "url=" in part:
                seen_url = True
            fixed_parts.append(part)

        return ",".join(fixed_parts)

    # Apply fix
    content = re.sub(pattern, fix_duplicate_url, content)

    # Also fix any SearchResult calls that are missing required fields
    def ensure_required_fields(match):
        text = match.group(0)
        # Parse parameters
        if "SearchResult(" not in text:
            return text

        # Extract the parameters section
        params_start = text.find("(") + 1
        params_end = text.rfind(")")
        params = text[params_start:params_end]

        # Check for required fields
        required = {
            "video_id": '"test_video"',
            "url": '"https://youtube.com/watch?v=test"',
            "title": '"Test Title"',
            "channel": '"Test Channel"',
            "channel_id": '"channel123"',
        }

        # Parse existing parameters
        param_dict = {}
        current_param = ""
        in_quotes = False
        paren_depth = 0

        for char in params + ",":
            if char == '"' and (not current_param or current_param[-1] != "\\"):
                in_quotes = not in_quotes
            elif char == "(" and not in_quotes:
                paren_depth += 1
            elif char == ")" and not in_quotes:
                paren_depth -= 1
            elif char == "," and not in_quotes and paren_depth == 0:
                if "=" in current_param:
                    key, value = current_param.split("=", 1)
                    param_dict[key.strip()] = value.strip()
                current_param = ""
                continue

            current_param += char

        # Add missing required fields
        for field, default in required.items():
            if field not in param_dict:
                param_dict[field] = default

        # Reconstruct the call
        new_params = ", ".join(f"{k}={v}" for k, v in param_dict.items())
        return f"SearchResult({new_params})"

    # Apply comprehensive fix
    content = re.sub(r"SearchResult\([^;]+?\)", ensure_required_fields, content, flags=re.DOTALL)

    file_path.write_text(content)
    print(f"Fixed duplicate parameters in {file_path}")


def main():
    """Run fixes."""
    fix_duplicate_params()
    print("Duplicate parameter fixes completed!")


if __name__ == "__main__":
    main()
