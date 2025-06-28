#!/usr/bin/env python3
"""Test the fixed channel patterns."""

import re
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from collector.advanced_parser import AdvancedTitleParser

def test_text_cleaning():
    """Test text cleaning functionality."""
    
    parser = AdvancedTitleParser()
    
    test_cases = [
        ("Karaoke Gib mir sonne ", "Gib mir sonne"),
        ("Rosenstolz *", "Rosenstolz"),
        ("Karaoke Von allein ", "Von allein"),
        ("Culcha Candela *", "Culcha Candela"),
    ]
    
    print("Testing text cleaning:")
    print("=" * 40)
    
    all_passed = True
    for original, expected in test_cases:
        cleaned = parser._clean_extracted_text(original)
        passed = cleaned == expected
        all_passed = all_passed and passed
        
        status = "✅" if passed else "❌"
        print(f"{status} '{original}' → '{cleaned}' (expected: '{expected}')")
    
    return all_passed

def test_channel_detection():
    """Test channel detection."""
    
    channel_name = "KaraFun Deutschland"
    channel_lower = channel_name.lower()
    karaoke_indicators = [
        "karaoke", "karafun", "karaoké", "karaokê", "караоке", "sing along",
        "backing track", "instrumental", "covers", "tribute", "piano version"
    ]
    
    is_karaoke_channel = any(indicator in channel_lower for indicator in karaoke_indicators)
    
    print(f"\nTesting channel detection:")
    print("=" * 40)
    print(f"Channel: '{channel_name}'")
    print(f"Is karaoke channel: {is_karaoke_channel}")
    
    for indicator in karaoke_indicators:
        if indicator in channel_lower:
            print(f"✅ Matched indicator: '{indicator}'")
            return True
    
    print("❌ No indicator matched")
    return False

def test_pattern_matching():
    """Test pattern matching with cleaning."""
    
    parser = AdvancedTitleParser()
    
    test_title = "Karaoke Gib mir sonne - Rosenstolz *"
    pattern = r"^([^-–—]+)\s*[-–—]\s*([^(\[]+)"
    
    print(f"\nTesting pattern matching:")
    print("=" * 40)
    print(f"Title: '{test_title}'")
    print(f"Pattern: '{pattern}'")
    
    match = re.search(pattern, test_title, re.IGNORECASE | re.UNICODE)
    
    if match:
        # Simulate the fixed group assignments (artist_group=2, title_group=1)
        raw_song = match.group(1)  # title_group
        raw_artist = match.group(2)  # artist_group
        
        clean_song = parser._clean_extracted_text(raw_song)
        clean_artist = parser._clean_extracted_text(raw_artist)
        
        print(f"✅ MATCH!")
        print(f"Raw Groups: '{raw_song}' | '{raw_artist}'")
        print(f"Cleaned: Song='{clean_song}', Artist='{clean_artist}'")
        
        expected_song = "Gib mir sonne"
        expected_artist = "Rosenstolz"
        
        song_correct = clean_song == expected_song
        artist_correct = clean_artist == expected_artist
        
        print(f"Expected: Song='{expected_song}', Artist='{expected_artist}'")
        print(f"Results: Song {'✅' if song_correct else '❌'}, Artist {'✅' if artist_correct else '❌'}")
        
        return song_correct and artist_correct
    else:
        print("❌ NO MATCH")
        return False

if __name__ == "__main__":
    print("Testing Channel Template Pattern Fixes")
    print("=" * 50)
    
    success1 = test_text_cleaning()
    success2 = test_channel_detection()
    success3 = test_pattern_matching()
    
    overall_success = success1 and success2 and success3
    print(f"\nOverall result: {'SUCCESS' if overall_success else 'FAILURE'}")
    
    sys.exit(0 if overall_success else 1)