#!/usr/bin/env python3
"""
Quick Discogs API Prototype - Phase 0
Test basic Discogs API functionality before full implementation.
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import aiohttp


@dataclass
class DiscogsTestResult:
    """Test result from Discogs API."""

    artist: str
    title: str
    year: Optional[int]
    genres: List[str]
    styles: List[str]
    release_id: str
    master_id: Optional[str]
    confidence: float


class DiscogsPrototype:
    """Simple Discogs API client for prototyping."""

    BASE_URL = "https://api.discogs.com"

    def __init__(self, token: str):
        self.token = token
        self.headers = {
            "User-Agent": "KaraokeCollector/2.1 +https://github.com/karaoke/search",
            "Authorization": f"Discogs token={token}",
        }

    async def test_api_connection(self) -> bool:
        """Test if API connection works."""
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(f"{self.BASE_URL}/users/test") as response:
                    return response.status in [200, 401]  # 401 is fine, means auth works
        except Exception as e:
            print(f"API connection test failed: {e}")
            return False

    async def search_track(
        self, artist: str, track: str, max_results: int = 5
    ) -> List[DiscogsTestResult]:
        """Search for tracks on Discogs."""
        params = {"artist": artist, "track": track, "type": "release", "per_page": max_results}

        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(
                    f"{self.BASE_URL}/database/search", params=params
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

                    results = []
                    for item in data.get("results", []):
                        result = DiscogsTestResult(
                            artist=item.get("artist", "Unknown"),
                            title=item.get("title", "Unknown"),
                            year=item.get("year"),
                            genres=item.get("genre", []),
                            styles=item.get("style", []),
                            release_id=str(item.get("id", "")),
                            master_id=(
                                str(item.get("master_id", "")) if item.get("master_id") else None
                            ),
                            confidence=self._calculate_confidence(item),
                        )
                        results.append(result)

                    return results

        except Exception as e:
            print(f"Search failed: {e}")
            return []

    def _calculate_confidence(self, item: Dict) -> float:
        """Calculate simple confidence score."""
        confidence = 0.5

        if item.get("master_id"):
            confidence += 0.1
        if item.get("year"):
            confidence += 0.1
        if item.get("genre"):
            confidence += 0.1
        if item.get("style"):
            confidence += 0.1

        # Higher confidence for more popular releases
        community = item.get("community", {})
        if community.get("have", 0) > 100:
            confidence += 0.1

        return min(confidence, 1.0)


async def run_prototype_tests():
    """Run comprehensive prototype tests."""

    # Get token from environment or user input
    token = os.getenv("DISCOGS_TOKEN")
    if not token:
        print("Please set DISCOGS_TOKEN environment variable or get one from:")
        print("https://www.discogs.com/settings/developers")
        return

    print("🎵 Discogs API Prototype Test")
    print("=" * 40)

    client = DiscogsPrototype(token)

    # Test 1: API Connection
    print("\n1. Testing API Connection...")
    connection_ok = await client.test_api_connection()
    print(f"   ✅ Connection: {'OK' if connection_ok else 'FAILED'}")

    if not connection_ok:
        print("   ❌ Cannot proceed without API connection")
        return

    # Test 2: Popular Track Search
    print("\n2. Testing Popular Track Search...")
    test_cases = [
        ("Adele", "Hello"),
        ("The Beatles", "Yesterday"),
        ("Queen", "Bohemian Rhapsody"),
        ("Ed Sheeran", "Shape of You"),
        ("Taylor Swift", "Shake It Off"),
    ]

    total_searches = 0
    successful_searches = 0
    total_results = 0

    for artist, track in test_cases:
        print(f"\n   Searching: {artist} - {track}")
        results = await client.search_track(artist, track, max_results=3)

        total_searches += 1
        if results:
            successful_searches += 1
            total_results += len(results)

            print(f"   ✅ Found {len(results)} results")
            for i, result in enumerate(results[:2], 1):  # Show top 2
                print(f"      {i}. {result.artist} - {result.title}")
                print(f"         Year: {result.year}, Confidence: {result.confidence:.2f}")
                if result.genres:
                    print(f"         Genres: {', '.join(result.genres[:3])}")
        else:
            print("   ❌ No results found")

        # Rate limiting - wait 1 second between requests
        await asyncio.sleep(1)

    # Test 3: Edge Cases
    print("\n3. Testing Edge Cases...")
    edge_cases = [
        ("Unknown Artist", "Unknown Song"),
        ("", ""),
        ("Artist with Special Chars áéíóú", "Song (Karaoke Version)"),
    ]

    for artist, track in edge_cases:
        print(f"\n   Edge case: '{artist}' - '{track}'")
        results = await client.search_track(artist, track, max_results=1)
        print(f"   Results: {len(results)}")
        await asyncio.sleep(1)

    # Test Summary
    print("\n" + "=" * 40)
    print("📊 PROTOTYPE TEST SUMMARY")
    print("=" * 40)

    success_rate = (successful_searches / total_searches * 100) if total_searches > 0 else 0
    avg_results = (total_results / successful_searches) if successful_searches > 0 else 0

    print(f"Total searches: {total_searches}")
    print(f"Successful searches: {successful_searches}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Average results per successful search: {avg_results:.1f}")
    print(f"Total results found: {total_results}")

    # Recommendations
    print("\n🎯 RECOMMENDATIONS:")
    if success_rate >= 80:
        print("✅ API performance is excellent - proceed with full implementation")
    elif success_rate >= 60:
        print("⚠️  API performance is good - consider optimization strategies")
    else:
        print("❌ API performance is poor - investigate issues before proceeding")

    if avg_results >= 2:
        print("✅ Good result diversity - multiple options available")
    else:
        print("⚠️  Limited result diversity - may need broader search strategies")


if __name__ == "__main__":
    print("To run this prototype:")
    print("1. Get Discogs token from: https://www.discogs.com/settings/developers")
    print("2. Set environment variable: export DISCOGS_TOKEN=your_token_here")
    print("3. Run: python discogs_prototype.py")
    print()

    # Check if running directly
    if os.getenv("DISCOGS_TOKEN"):
        asyncio.run(run_prototype_tests())
    else:
        print("DISCOGS_TOKEN not found in environment variables.")
        print("Please set it and run again.")
