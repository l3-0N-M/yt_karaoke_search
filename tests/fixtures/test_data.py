"""Test fixtures and mock data for unit tests."""

from datetime import datetime

# Sample video data for testing
SAMPLE_VIDEO_DATA = {
    "basic": {
        "video_id": "dQw4w9WgXcQ",
        "url": "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "title": "Rick Astley - Never Gonna Give You Up (Karaoke Version)",
        "description": "Classic 80s hit in karaoke format",
        "duration_seconds": 213,
        "view_count": 1000000,
        "like_count": 50000,
        "comment_count": 5000,
        "upload_date": "20200515",
        "thumbnail_url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
        "channel_name": "Karaoke Hits",
        "channel_id": "UC123456789",
        "artist": "Rick Astley",
        "song_title": "Never Gonna Give You Up",
        "release_year": 1987,
        "genre": "Pop",
    },
    "with_discogs": {
        "video_id": "test_discogs_123",
        "url": "https://youtube.com/watch?v=test_discogs_123",
        "title": "Queen - Bohemian Rhapsody (Karaoke)",
        "artist": "Queen",
        "song_title": "Bohemian Rhapsody",
        "release_year": 2025,  # Wrong year from title parsing
        "discogs_artist_id": "12345",
        "discogs_artist_name": "Queen",
        "discogs_release_id": "67890",
        "discogs_release_title": "A Night at the Opera",
        "discogs_release_year": 1975,  # Correct year from Discogs
        "discogs_label": "EMI",
        "discogs_genre": "Rock",
        "discogs_style": "Prog Rock, Art Rock",
        "discogs_checked": 1,
        "channel_id": "UC_karaoke",
        "channel_name": "Classic Karaoke",
    },
    "with_nulls": {
        "video_id": "null_test_456",
        "url": "https://youtube.com/watch?v=null_test_456",
        "title": "Unknown Artist - Unknown Song",
        "artist": None,
        "song_title": None,
        "featured_artists": None,
        "release_year": None,
        "genre": None,
        "description": None,
        "channel_id": "UC_unknown",
        "channel_name": "Unknown Channel",
    },
    "with_special_chars": {
        "video_id": "special_789",
        "url": "https://youtube.com/watch?v=special_789",
        "title": "Beyoncé - Déjà Vu (feat. Jay-Z) [Karaoke] 🎤",
        "artist": "Beyoncé",
        "song_title": "Déjà Vu",
        "featured_artists": 'Jay-Z, Beyoncé\'s Backup Singers & The "Special" Crew',
        "description": "Karaoke version with lyrics!\n\nOriginal: ♪ ♫ ♬",
        "channel_id": "UC_special",
        "channel_name": "Karaoke★Stars",
    },
    "year_edge_cases": {
        "video_id": "year_test_111",
        "url": "https://youtube.com/watch?v=year_test_111",
        "title": f"New Hit {datetime.now().year} - Latest Release",
        "description": "Brand new song just released this year!",
        "upload_date": f"{datetime.now().year}0101",
        "release_year": datetime.now().year,  # Should be rejected
        "channel_id": "UC_new",
        "channel_name": "New Music Karaoke",
    },
}

# Sample search results for web search testing
SAMPLE_SEARCH_RESULTS = [
    {
        "title": "Madonna - Like a Prayer (Official Video)",
        "artist": "Madonna",
        "duration": 344,
        "year": 1989,
        "views": 50000000,
    },
    {"title": None, "artist": "Unknown", "duration": None, "year": None},  # Test null handling
    {
        "title": "Elton John - Your Song",
        "artist": None,  # Test null artist
        "duration": 241,
        "year": 1970,
    },
]

# Problematic strings that previously caused issues
PROBLEMATIC_STRINGS = {
    "very_long": "A" * 1000,  # Very long string
    "with_nulls": "String\x00with\x00null\x00bytes",
    "unicode_mix": "Émojis 🎵🎤 and spëcial çhars ñ",
    "control_chars": "Text\x01with\x02control\x03chars",
    "sql_injection": "'; DROP TABLE videos; --",
    "json_like": '{"key": "value", "array": [1, 2, 3]}',
    "html_entities": '&lt;script&gt;alert("test")&lt;/script&gt;',
    "mixed_quotes": """String with 'single' and "double" quotes and backticks `test`""",
    "surrogate_pairs": "Test \udcff invalid surrogate",
    "rtl_text": "مرحبا Hello שלום",
}

# Rate limit test scenarios
RATE_LIMIT_SCENARIOS = {
    "burst_requests": {
        "requests": 10,
        "interval": 0.1,  # 100ms between requests
        "expected_fast": 3,  # Only burst tokens should be fast
    },
    "sustained_load": {
        "requests": 100,
        "duration": 120,  # 2 minutes
        "max_rate": 48,  # 80% of 60 requests/minute
    },
    "429_errors": {
        "consecutive_errors": 5,
        "retry_after_values": [None, 30, None, 60, None],
        "expected_backoffs": [1.0, 30, 4.0, 60, 16.0],  # Exponential or server-specified
    },
}

# Database migration test cases
MIGRATION_TEST_CASES = {
    "old_schema": """
        CREATE TABLE videos (
            video_id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            artist TEXT,
            song_title TEXT,
            release_year INTEGER
        )
    """,
    "intermediate_schema": """
        CREATE TABLE videos (
            video_id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            artist TEXT,
            song_title TEXT,
            release_year INTEGER,
            genre TEXT,
            parse_confidence REAL
        )
    """,
    "expected_columns": [
        "video_id",
        "url",
        "title",
        "description",
        "duration_seconds",
        "view_count",
        "like_count",
        "comment_count",
        "upload_date",
        "thumbnail_url",
        "channel_name",
        "channel_id",
        "artist",
        "song_title",
        "featured_artists",
        "release_year",
        "genre",
        "parse_confidence",
        "quality_score",
        "engagement_ratio",
        "discogs_artist_id",
        "discogs_artist_name",
        "discogs_release_id",
        "discogs_release_title",
        "discogs_release_year",
        "discogs_label",
        "discogs_genre",
        "discogs_style",
        "discogs_checked",
        "musicbrainz_checked",
        "web_search_performed",
        "scraped_at",
    ],
}


def get_mock_parse_result(confidence=0.85):
    """Create a mock ParseResult for testing."""
    from unittest.mock import Mock

    result = Mock()
    result.artist = "Test Artist"
    result.song_title = "Test Song"
    result.confidence = confidence
    result.source = "test"
    result.featured_artists = ["Featured Artist 1", "Featured Artist 2"]
    result.release_year = 2020
    result.genre = "Pop"
    return result


def get_mock_discogs_match():
    """Create a mock Discogs match result."""
    from unittest.mock import Mock

    match = Mock()
    match.artist_name = "Verified Artist"
    match.song_title = "Verified Song"
    match.year = 1985
    match.confidence = 0.95
    match.artist_id = "discogs_12345"
    match.release_id = "release_67890"
    match.label = "Test Records"
    match.genres = ["Rock", "Pop"]
    match.styles = ["Classic Rock", "Soft Rock"]
    return match
