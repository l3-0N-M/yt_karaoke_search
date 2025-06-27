# Karaoke Video Collector

[![CI](https://github.com/your-username/karaoke-video-collector/workflows/CI/badge.svg)](https://github.com/your-username/karaoke-video-collector/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive tool for collecting karaoke video data from YouTube with confidence scoring and Return YouTube Dislike integration.

## Features

- **Fast yt-dlp-based search** - No browser automation needed for most searches
- **Return YouTube Dislike integration** - Get estimated dislike counts with confidence scoring
- **Comprehensive metadata extraction** - Artist names, song titles, featured artists, release years
- **Engagement analytics** - Like/dislike ratios and engagement scoring with penalties
- **Karaoke feature detection** - Guide vocals, scrolling lyrics, instrumental-only detection
- **External music data** - Release year lookup via MusicBrainz API
- **Quality scoring** - Technical, engagement, and metadata completeness scores
- **Modular architecture** - Clean, maintainable codebase with proper separation of concerns
- **Robust error handling** - Retry logic, graceful shutdown, and comprehensive logging
- **Database management** - SQLite with migrations, backups, and performance optimization

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd karaoke-video-collector

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .

# For development
pip install -e ".[dev]"
```

## Usage

### Command Line Interface

```bash
# Basic collection with default queries
karaoke-collector collect

# Custom queries
karaoke-collector collect -q "piano karaoke" -q "acoustic karaoke" -m 100

# With custom configuration
karaoke-collector collect --config config.yaml --verbose

# Create configuration template
karaoke-collector create-config --output my_config.yaml

# Show database statistics
karaoke-collector stats
```

### Python API

```python
from collector import KaraokeCollector, CollectorConfig

# Create configuration
config = CollectorConfig()
config.scraping.max_concurrent_workers = 8
config.database.path = "my_karaoke_db.db"

# Initialize collector
collector = KaraokeCollector(config)

# Collect videos
queries = ["piano karaoke", "guitar karaoke"]
total_processed = await collector.collect_videos(queries, max_videos_per_query=50)

# Get statistics
stats = await collector.get_statistics()
print(f"Total videos: {stats['total_videos']}")
```

## Configuration

Create a `config.yaml` file to customize behavior:

```yaml
database:
  path: "karaoke_videos.db"
  backup_enabled: true

scraping:
  max_concurrent_workers: 5
  max_retries: 3
  timeout_seconds: 30

data_sources:
  ryd_api_enabled: true

search:
  max_results_per_query: 100
```

## Database Schema

The collector creates a normalized SQLite database with the following tables:

- `videos` - Main video metadata including:
  - Basic info (title, URL, duration, views, likes)
  - Artist data (original_artist, featured_artists, song_title, estimated_release_year)
  - Engagement metrics (like_dislike_to_views_ratio)
- `video_features` - Karaoke-specific features with confidence scores
- `quality_scores` - Technical, engagement, and metadata quality metrics
- `ryd_data` - Return YouTube Dislike data with confidence scoring
- `search_history` - Track search performance over time
- `error_log` - Debugging and monitoring

### Key Fields Collected:
✅ **YouTube URL** - Direct video links  
✅ **Title** - Full video title  
✅ **Original artist** - Extracted with confidence scoring  
✅ **Featured artists** - "feat./ft./featuring" collaborations  
✅ **Release year** - Via MusicBrainz API lookup  
✅ **Video length** - Duration in seconds  
✅ **View count** - Current view statistics  
✅ **Like/dislike ratio** - Computed engagement metric  
✅ **Karaoke features** - Guide vocals, lyrics style, instrumentals

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Keep yt-dlp updated for YouTube API changes
pip install --upgrade yt-dlp

# Run tests
pytest

# Run linting
ruff check .

# Format code
black .

# Install pre-commit hooks
pre-commit install
```

### 🔄 **Staying Current with YouTube Changes**

YouTube frequently updates their internal APIs. Keep your collector robust:

```bash
# Update to latest yt-dlp (includes recent innertube 'sts'-token patches)
pip install --upgrade yt-dlp

# Check for breaking changes
karaoke-collector collect -q "test" -m 5 --verbose
```

**Pro Tip:** yt-dlp releases nightly builds that patch new YouTube restrictions. The collector automatically benefits from these updates! 🚀

## Testing

The package includes comprehensive tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=collector

# Run specific test file
pytest tests/test_db.py -v
```

## Legal Considerations

⚠️ **Important**: This tool scrapes YouTube data outside of the official API, which may violate YouTube's Terms of Service. Use responsibly and consider the legal implications:

- YouTube's ToS prohibits scraping content without permission
- Running this tool may result in account termination
- Consider using official YouTube Data API for production use
- Respect rate limits and be mindful of server load

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run tests and linting
6. Submit a pull request

## License

This project is for educational purposes only. Users are responsible for ensuring compliance with applicable laws and terms of service.