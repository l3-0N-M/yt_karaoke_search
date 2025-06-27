#!/usr/bin/env python3
"""Test the advanced parser to see why it's truncating song_title."""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from collector.advanced_parser import AdvancedTitleParser
from collector.config import CollectorConfig


def test_advanced_parser():
    """Test the advanced parser with the problematic title."""
    config = CollectorConfig()

    print(f"Advanced parser enabled in config: {config.search.use_advanced_parser}")

    if config.search.use_advanced_parser:
        parser = AdvancedTitleParser(config)

        test_title = "Karaoke - What Child Is This - Christmas Traditional"
        print(f"\nTesting advanced parser with: '{test_title}'")
        print("=" * 80)

        result = parser.parse_title(test_title, "", "", "Test Channel")

        print("Advanced parser result:")
        print(
            f"  original_artist: '{result.original_artist}' (length: {len(result.original_artist) if result.original_artist else 'None'})"
        )
        print(
            f"  song_title: '{result.song_title}' (length: {len(result.song_title) if result.song_title else 'None'})"
        )
        print(f"  confidence: {result.confidence}")
        print(f"  method: {result.method}")
        print(f"  pattern_used: {result.pattern_used}")

        if result.alternative_results:
            print(f"  alternative_results: {result.alternative_results}")

        # Let's test the core pattern matching specifically
        print("\nTesting core pattern matching:")
        core_result = parser._parse_with_core_patterns(test_title)
        print("  Core patterns result:")
        print(
            f"    original_artist: '{core_result.original_artist}' (length: {len(core_result.original_artist) if core_result.original_artist else 'None'})"
        )
        print(
            f"    song_title: '{core_result.song_title}' (length: {len(core_result.song_title) if core_result.song_title else 'None'})"
        )
        print(f"    confidence: {core_result.confidence}")
        print(f"    method: {core_result.method}")
        print(f"    pattern_used: {core_result.pattern_used}")

        # Test individual patterns from the advanced parser
        print("\nTesting individual core patterns:")
        for i, (pattern, artist_group, title_group, confidence, pattern_name) in enumerate(
            parser.core_patterns
        ):
            import re

            match = re.search(pattern, test_title, re.IGNORECASE | re.UNICODE)
            if match:
                print(f"  Pattern {i+1} ({pattern_name}) MATCHED:")
                print(f"    Pattern: {pattern}")
                print(f"    Groups: {match.groups()}")
                print(f"    artist_group={artist_group}, title_group={title_group}")

                if artist_group and artist_group <= len(match.groups()):
                    raw_artist = match.group(artist_group)
                    cleaned_artist = parser._clean_extracted_text(raw_artist)
                    valid_artist = parser._is_valid_artist_name(cleaned_artist)
                    print(
                        f"    Artist: raw='{raw_artist}' -> cleaned='{cleaned_artist}' -> valid={valid_artist}"
                    )

                if title_group and title_group <= len(match.groups()):
                    raw_title = match.group(title_group)
                    cleaned_title = parser._clean_extracted_text(raw_title)
                    valid_title = parser._is_valid_song_title(cleaned_title)
                    print(
                        f"    Title: raw='{raw_title}' -> cleaned='{cleaned_title}' -> valid={valid_title}"
                    )

                break
    else:
        print("Advanced parser is disabled in config")


if __name__ == "__main__":
    test_advanced_parser()
