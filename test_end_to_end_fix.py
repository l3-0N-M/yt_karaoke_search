#!/usr/bin/env python3
"""Test the end-to-end fix for artist/title swapping."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from collector.advanced_parser import AdvancedTitleParser

def test_pattern_fixes():
    """Test that the pattern fixes work correctly."""
    
    parser = AdvancedTitleParser()
    
    # Test cases from the database that were previously swapped
    test_cases = [
        {
            'title': 'Karaoke Gib mir sonne - Rosenstolz',
            'expected_artist': 'Rosenstolz',
            'expected_song': 'Gib mir sonne'
        },
        {
            'title': 'Karaoke Alles Rot - Silly',
            'expected_artist': 'Silly', 
            'expected_song': 'Alles Rot'
        },
        {
            'title': 'Karaoke Von allein - Culcha Candela',
            'expected_artist': 'Culcha Candela',
            'expected_song': 'Von allein'
        },
        {
            'title': 'Karaoke Lili Marlene - Marlene Dietrich',
            'expected_artist': 'Marlene Dietrich',
            'expected_song': 'Lili Marlene'
        },
        {
            'title': 'Karaoke Himmel auf - Silbermond',
            'expected_artist': 'Silbermond',
            'expected_song': 'Himmel auf'
        }
    ]
    
    print("Testing fixed pattern parsing:")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        title = test_case['title']
        expected_artist = test_case['expected_artist']
        expected_song = test_case['expected_song']
        
        print(f"\n{i}. Testing: '{title}'")
        
        # Parse the title
        result = parser.parse_title(title)
        
        # Check results
        artist_correct = result.original_artist == expected_artist
        song_correct = result.song_title == expected_song
        
        print(f"   Expected: Artist='{expected_artist}', Song='{expected_song}'")
        print(f"   Got:      Artist='{result.original_artist}', Song='{result.song_title}'")
        print(f"   Method:   {result.method}")
        print(f"   Pattern:  {result.pattern_used}")
        print(f"   Confidence: {result.confidence}")
        
        if artist_correct and song_correct:
            print("   ✅ PASS")
            passed += 1
        else:
            print("   ❌ FAIL")
            if not artist_correct:
                print(f"      - Artist mismatch: got '{result.original_artist}', expected '{expected_artist}'")
            if not song_correct:
                print(f"      - Song mismatch: got '{result.song_title}', expected '{expected_song}'")
            failed += 1
    
    print(f"\n" + "=" * 80)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The pattern fix is working correctly.")
    else:
        print("⚠️  Some tests failed. The pattern may need further adjustment.")
    
    return failed == 0

if __name__ == "__main__":
    success = test_pattern_fixes()
    sys.exit(0 if success else 1)