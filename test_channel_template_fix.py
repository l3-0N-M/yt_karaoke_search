#!/usr/bin/env python3
"""Test the channel template pass fixes for artist/title swapping."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from collector.passes.channel_template_pass import EnhancedChannelTemplatePass
from collector.advanced_parser import AdvancedTitleParser

def test_channel_template_fixes():
    """Test that the channel template pass fixes work correctly."""
    
    # Set up parser and pass
    parser = AdvancedTitleParser()
    channel_pass = EnhancedChannelTemplatePass(parser)
    
    # Test cases from the KaraFun Deutschland channel
    test_cases = [
        {
            'title': 'Karaoke Gib mir sonne - Rosenstolz *',
            'channel_name': 'KaraFun Deutschland',
            'channel_id': 'UCtest123',
            'expected_artist': 'Rosenstolz',
            'expected_song': 'Gib mir sonne'
        },
        {
            'title': 'Karaoke Alles Rot - Silly *',
            'channel_name': 'KaraFun Deutschland',
            'channel_id': 'UCtest123',
            'expected_artist': 'Silly', 
            'expected_song': 'Alles Rot'
        },
        {
            'title': 'Karaoke Von allein - Culcha Candela *',
            'channel_name': 'KaraFun Deutschland',
            'channel_id': 'UCtest123',
            'expected_artist': 'Culcha Candela',
            'expected_song': 'Von allein'
        },
        {
            'title': 'Karaoke Himmel auf - Silbermond *',
            'channel_name': 'KaraFun Deutschland',
            'channel_id': 'UCtest123',
            'expected_artist': 'Silbermond',
            'expected_song': 'Himmel auf'
        }
    ]
    
    print("Testing channel template pass fixes:")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        title = test_case['title']
        channel_name = test_case['channel_name']
        channel_id = test_case['channel_id']
        expected_artist = test_case['expected_artist']
        expected_song = test_case['expected_song']
        
        print(f"\n{i}. Testing: '{title}'")
        print(f"   Channel: {channel_name}")
        
        # Parse using channel template pass
        result = channel_pass.parse(
            title=title,
            channel_name=channel_name,
            channel_id=channel_id
        )
        
        if result:
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
        else:
            print("   ❌ FAIL - No result returned")
            failed += 1
    
    print(f"\n" + "=" * 80)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All channel template tests passed!")
    else:
        print("⚠️  Some channel template tests failed.")
    
    return failed == 0

def test_confidence_threshold():
    """Test that the new confidence threshold works correctly."""
    
    parser = AdvancedTitleParser()
    channel_pass = EnhancedChannelTemplatePass(parser)
    
    # Test with a title that should get lower confidence (not cause early exit)
    result = channel_pass.parse(
        title="Some unclear title - maybe artist",
        channel_name="KaraFun Deutschland",
        channel_id="UCtest123"
    )
    
    print("\nTesting confidence threshold:")
    print("=" * 40)
    
    if result:
        print(f"Result confidence: {result.confidence}")
        print(f"Should be < 0.75 to avoid early exit: {result.confidence < 0.75}")
        
        if result.confidence < 0.75:
            print("✅ Confidence threshold working correctly")
            return True
        else:
            print("⚠️  Confidence might be too high")
            return False
    else:
        print("✅ No result for unclear title (good)")
        return True

if __name__ == "__main__":
    print("Testing Channel Template Pass Fixes")
    print("=" * 50)
    
    success1 = test_channel_template_fixes()
    success2 = test_confidence_threshold()
    
    overall_success = success1 and success2
    print(f"\nOverall result: {'SUCCESS' if overall_success else 'FAILURE'}")
    
    sys.exit(0 if overall_success else 1)