#!/usr/bin/env python3
"""Test script to verify Discogs search improvements."""

import asyncio
import os
import sys
from pathlib import Path

# Add the collector module to Python path
sys.path.insert(0, str(Path(__file__).parent))

from collector.config import CollectorConfig
from collector.passes.discogs_search_pass import DiscogsClient
from collector.utils import DiscogsRateLimiter


async def test_discogs_search():
    """Test Discogs search with problematic cases from the log analysis."""

    # Test cases that previously failed
    test_cases = [
        # Recent releases that failed
        ("ROSÉ", "Messy"),
        ("JENNIE", "JENNIE on Seoul City"),
        ("Lola Young", "One Thing"),
        ("Katseye", "Gnarly"),
        # Special characters
        ("A$AP Rocky", "Sundress"),
        ("PLUTO (72) & YKNIECE", "Whim Whamiee"),
        ("The Click (2)", "The Best of The Click"),
        # Interview/special tracks
        ("Stryper", "Stryper Interview"),
        ("BIGBANG", "IF YOU (BIGBANG SPECIAL EVENT 2017)"),
        ("Shania Twain", "Shania Twain Hitmix"),
        # Other failed searches
        ("Cody Johnson", "I'm Gonna Love You"),
        ("Mahmood", "Soldi"),
        ("Sleep Token", "Like That (instrumental)"),
        ("Donny Hathaway", "Be There"),
        ("Donny Hathaway", "He Ain't Heavy, He's My Brother"),
        ("Bobby Womack", "Woman's Gotta Have It"),
    ]

    # Initialize components
    config = CollectorConfig()

    # Check if Discogs token is available
    token = os.getenv("DISCOGS_TOKEN") or os.getenv("DISCOGS-TOKEN")
    if not token:
        print("❌ DISCOGS_TOKEN not found in environment. Please set it to run tests.")
        return

    rate_limiter = DiscogsRateLimiter(requests_per_minute=60)
    client = DiscogsClient(
        token=token, rate_limiter=rate_limiter, user_agent=config.data_sources.discogs_user_agent
    )

    print("🔍 Testing Discogs search improvements...\n")

    successful = 0
    total = len(test_cases)

    for artist, track in test_cases:
        print(f"Testing: {artist} - {track}")

        try:
            # Test the search
            matches = await client.search_release(
                artist=artist, track=track, max_results=20, timeout=10
            )

            if matches:
                best_match = max(matches, key=lambda x: x.confidence)
                if best_match.confidence >= 0.4:  # New threshold
                    successful += 1
                    print(
                        f"✅ Found: {best_match.artist_name} - {best_match.song_title} "
                        f"(confidence: {best_match.confidence:.2f}, year: {best_match.year})"
                    )
                else:
                    print(
                        f"⚠️  Low confidence: {best_match.confidence:.2f} for "
                        f"{best_match.artist_name} - {best_match.song_title}"
                    )
            else:
                print("❌ No matches found")

        except Exception as e:
            print(f"❌ Error: {e}")

        print()

        # Small delay between requests
        await asyncio.sleep(0.5)

    # Summary
    success_rate = (successful / total) * 100
    print("\n📊 Results Summary:")
    print(f"   Successful searches: {successful}/{total} ({success_rate:.1f}%)")
    print("   Target success rate: 95%")

    if success_rate >= 95:
        print("   ✅ Target achieved!")
    else:
        print(f"   ⚠️  Need to improve by {95 - success_rate:.1f}%")

    # Test artist variation generation
    print("\n🔧 Testing artist variation generation:")
    test_artists = ["A$AP Rocky", "ROSÉ", "The Click (2)", "PLUTO (72) & YKNIECE"]
    for artist in test_artists:
        variations = client._generate_artist_variations(artist)
        print(f"   {artist} → {variations}")

    # Test query normalization
    print("\n🔧 Testing query normalization:")
    test_queries = [
        "Messy (karaoke version)",
        "Sundress [Official Audio]",
        "Like That (instrumental)",
        "IF YOU (BIGBANG SPECIAL EVENT 2017)",
    ]
    for query in test_queries:
        normalized = client._normalize_search_query(query)
        print(f"   '{query}' → '{normalized}'")


if __name__ == "__main__":
    asyncio.run(test_discogs_search())
