#!/usr/bin/env python3
"""Test script to validate pattern fixes against known problematic records."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collector.advanced_parser import AdvancedTitleParser

def test_channel_patterns():
    """Test our new channel-specific patterns against known problematic cases."""
    
    parser = AdvancedTitleParser()
    
    # Test cases from the problematic records we identified
    test_cases = [
        # Let's Sing Karaoke format: "LastName, FirstName - Song Title (Karaoke & Lyrics)"
        {
            "title": "Benson, George - This Masquerade (Karaoke & Lyrics)",
            "channel": "Let's Sing Karaoke",
            "expected_artist": "George Benson",
            "expected_song": "This Masquerade",
            "description": "Let's Sing Karaoke format"
        },
        {
            "title": "Lopez, Jennifer - Ain't If Funny (Karaoke & Lyrics)", 
            "channel": "Let's Sing Karaoke",
            "expected_artist": "Jennifer Lopez",
            "expected_song": "Ain't If Funny",
            "description": "Let's Sing Karaoke format"
        },
        {
            "title": "Houston, Whitney - So Emotional (Karaoke & Lyrics)",
            "channel": "Let's Sing Karaoke", 
            "expected_artist": "Whitney Houston",
            "expected_song": "So Emotional",
            "description": "Let's Sing Karaoke format"
        },
        
        # Lugn format: "ARTIST • Song Title • Karaoke"
        {
            "title": "AYLIVA • Nein! • Karaoke (Stripped Version)",
            "channel": "Lugn",
            "expected_artist": "AYLIVA",
            "expected_song": "Nein!",
            "description": "Lugn bullet format"
        },
        {
            "title": "AYLIVA • Nein! • Karaoke",
            "channel": "Lugn",
            "expected_artist": "AYLIVA", 
            "expected_song": "Nein!",
            "description": "Lugn bullet format"
        },
        {
            "title": "Kendrick Lamar • Not Like Us • Karaoke",
            "channel": "Lugn",
            "expected_artist": "Kendrick Lamar",
            "expected_song": "Not Like Us", 
            "description": "Lugn bullet format"
        },
        
        # KaraFun Deutschland format: "Karaoke Song Title - Artist Name *"
        {
            "title": "Karaoke Feliz Navidad - Helene Fischer *",
            "channel": "KaraFun Deutschland - Karaoke",
            "expected_artist": "Helene Fischer",
            "expected_song": "Feliz Navidad",
            "description": "KaraFun Deutschland format"
        },
        {
            "title": "Karaoke Die Rose (Live) - Helene Fischer *",
            "channel": "KaraFun Deutschland - Karaoke", 
            "expected_artist": "Helene Fischer",
            "expected_song": "Die Rose (Live)",
            "description": "KaraFun Deutschland format"
        },
        {
            "title": "Karaoke Alles Gute - Badesalz *",
            "channel": "KaraFun Deutschland - Karaoke",
            "expected_artist": "Badesalz", 
            "expected_song": "Alles Gute",
            "description": "KaraFun Deutschland format"
        },
    ]
    
    print("=== TESTING CHANNEL PATTERN FIXES ===")
    print()
    
    total_tests = len(test_cases)
    passed_tests = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}/{total_tests}: {test_case['description']}")
        print(f"  Title: '{test_case['title']}'")
        print(f"  Channel: '{test_case['channel']}'")
        print(f"  Expected: Artist='{test_case['expected_artist']}', Song='{test_case['expected_song']}'")
        
        # Parse the title
        result = parser.parse_title(
            title=test_case['title'],
            channel_name=test_case['channel']
        )
        
        print(f"  Parsed: Artist='{result.original_artist}', Song='{result.song_title}'")
        print(f"  Method: {result.method}, Confidence: {result.confidence:.3f}")
        
        # Check if the result matches expectations
        artist_correct = (result.original_artist or "").strip() == test_case['expected_artist'].strip()
        song_correct = (result.song_title or "").strip() == test_case['expected_song'].strip()
        
        if artist_correct and song_correct:
            print("  ✅ PASS: Both artist and song extracted correctly")
            passed_tests += 1
        else:
            print("  ❌ FAIL:", end="")
            if not artist_correct:
                print(f" Wrong artist (got '{result.original_artist}', expected '{test_case['expected_artist']}')", end="")
            if not song_correct:
                print(f" Wrong song (got '{result.song_title}', expected '{test_case['expected_song']}')", end="")
            print()
        
        print()
    
    print("=== SUMMARY ===")
    print(f"Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Pattern fixes are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Pattern fixes need adjustment.")
        return False

if __name__ == "__main__":
    test_channel_patterns()