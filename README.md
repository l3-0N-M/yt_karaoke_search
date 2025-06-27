# Karaoke Video Collector

[![CI](https://github.com/l3-0N-M/yt_karaoke_search/workflows/CI/badge.svg)](https://github.com/l3-0N-M/yt_karaoke_search/actions)
[![Python 3.8–3.11](https://img.shields.io/badge/python-3.8--3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive tool for collecting karaoke video data from YouTube with confidence scoring and Return YouTube Dislike integration.

## Features

- **Fast yt-dlp-based search** - No browser automation needed for most searches
- **Channel-based collection** - Systematic processing of entire YouTube channels
- **Enterprise scalability** - Handles 2,000-10,000 videos per channel efficiently
- **Return YouTube Dislike integration** - Get estimated dislike counts with confidence scoring
- **Advanced title parsing** - 5-pass parsing ladder with confidence-based progression, ML techniques, and adaptive learning
- **Comprehensive metadata extraction** - Artist names, song titles, featured artists, release years  
- **Multi-language support** - Handles Korean, Japanese, Chinese, and other international formats
- **Engagement analytics** - Like/dislike ratios and engagement scoring with penalties
- **Karaoke feature detection** - Guide vocals, scrolling lyrics, instrumental-only detection
- **External music data** - Release year lookup via MusicBrainz API
- **Quality scoring** - Technical, engagement, and metadata completeness scores
- **Incremental updates** - Smart date filtering for efficient re-processing
- **Rate limiting protection** - Prevents YouTube API blocking with intelligent backoff
- **Memory management** - Bounded caches and automatic cleanup for long-running operations
- **Database optimization** - Connection pooling, performance indexes, and migrations
- **Modular architecture** - Clean, maintainable codebase with proper separation of concerns
- **Robust error handling** - Retry logic, graceful shutdown, and comprehensive logging

## Installation


Requires **Python 3.8–3.11**. Create and activate a virtual environment, then install:

```bash
# Clone the repository
git clone https://github.com/l3-0N-M/yt_karaoke_search.git
cd yt_karaoke_search
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package (dependencies are installed automatically)
python -m pip install -e .

# For development with additional tools
pip install -e ".[dev]"
```
**Troubleshooting:** If `pip install -e .` fails with a permission error even in a virtual environment, check for a `~/.pydistutils.cfg` or `/etc/distutils.cfg` file. These files can override the install prefix. Rename or remove them and retry.

## Usage

### Command Line Interface

```bash
# Basic collection with default queries
karaoke-collector collect

# Custom queries
karaoke-collector collect -q "piano karaoke" -q "acoustic karaoke" -m 100

# Channel-based collection (single channel)
karaoke-collector collect-channel "https://www.youtube.com/@KaraokeChannel" --max-videos 5000

# Multiple channels from file
karaoke-collector collect-channels channels.txt --max-videos 10000

# With custom configuration
karaoke-collector collect --config config.yaml --verbose

# Create configuration template
karaoke-collector create-config --output my_config.yaml

# Show database statistics
karaoke-collector stats
```

### Multi-Strategy Search

The collector can optionally run an advanced search engine that queries multiple
providers and ranks results for better coverage. Enable it in the configuration
file with `search.use_multi_strategy: true`:

```yaml
search:
  use_multi_strategy: true
```

Run channel collection with this option enabled:

```bash
karaoke-collector collect-channel "https://www.youtube.com/@KaraokeChannel" --config config.yaml
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

# Collect videos from search queries
queries = ["piano karaoke", "guitar karaoke"]
total_processed = await collector.collect_videos(queries, max_videos_per_query=50)

# Collect from specific channel
channel_url = "https://www.youtube.com/@KaraokeChannel"
processed = await collector.collect_from_channel(channel_url, max_videos=5000)

# Collect from multiple channels
channel_urls = ["https://www.youtube.com/@Channel1", "https://www.youtube.com/@Channel2"]
total = await collector.collect_from_channels(channel_urls, max_videos_per_channel=10000)

# Get statistics
stats = await collector.get_statistics()
print(f"Total videos: {stats.get('total_videos', 0):,}")

# Cleanup resources when done
await collector.cleanup()
```

## Advanced Title Parsing

The collector features a sophisticated multi-strategy parsing system that handles diverse karaoke video naming schemes with an advanced 5-pass parsing ladder:

### Multi-Pass Parsing Ladder

The system implements a progressive 5-pass parsing ladder with confidence-based stopping and intelligent resource management:

```
Pass 0: Channel Template Match     (Threshold: 0.85) - Fast, channel-specific patterns
Pass 1: Auto-retemplate           (Threshold: 0.8)  - Recent upload pattern learning  
Pass 2: ML/Embedding Similarity   (Threshold: 0.75) - Semantic and fuzzy matching
Pass 3: Web Search Integration    (Threshold: 0.7)  - SERP-based parsing with caching
Pass 4: Acoustic Fingerprint      (Threshold: 0.9)  - Audio analysis (placeholder)
```

**Key Features:**
- **Confidence-based progression** - Stops at first pass meeting threshold
- **Budget management** - CPU time and API call limits per video
- **Drift detection** - Automatically detects when channel patterns change
- **SERP caching** - 1-week TTL cache for web search results
- **Language-aware** - Multi-language filler word removal
- **Exponential backoff** - Intelligent retry with increasing delays

### Supported Formats
- **Quoted formats**: `"Artist" - "Title" "(Karaoke Version)"`
- **Style-based**: `"Title" "(in the Style of "Artist")"`  
- **Channel-specific**: `Sing King Karaoke - "Title" "(in the Style of "Artist")"`
- **Complex formats**: `"Artist"-"Title" "("Movie"OST)" "(MR/Inst.)" "(-1Key)" "(Karaoke Version)"`
- **International**: `[짱가라오케/원키/노래방] Artist-Title [ZZang KARAOKE]`
- **Movie soundtracks**: `"Frozen" - "Let It Go" ("Idina Menzel") (Disney Karaoke)`

### Advanced Features
- **Multi-pass parsing** - Uses 5 different parsing strategies with confidence weighting
- **Language detection** - Automatically detects Korean, Japanese, Chinese, and other languages
- **Channel-specific patterns** - Learns patterns from specific karaoke channels
- **Fuzzy matching** - Matches against known artists/songs with spelling variations
- **Adaptive learning** - Improves accuracy over time by learning from successful parses
- **Featured artist extraction** - Identifies collaborations and featured artists
- **Confidence scoring** - Each extraction includes accuracy confidence metrics
- **Robust regex patterns** - Improved pattern matching for "Artist - Title" and "Title - Artist" formats
- **Performance statistics** - Tracks parsing success rates and pattern effectiveness

### Parser Configuration
```yaml
search:
  # Enable advanced multi-strategy parsing
  use_advanced_parser: true
  
  # Enable fuzzy matching against known data  
  enable_fuzzy_matching: true
  
  # Enable pattern learning for continuous improvement
  enable_pattern_learning: true
  
  # Minimum confidence threshold for extractions
  min_confidence_threshold: 0.5
  
  # Enable multi-language pattern detection
  enable_multi_language: true
  
  # Multi-pass parsing ladder configuration
  multi_pass:
    enabled: true
    stop_on_first_success: true
    total_cpu_budget: 60.0
    total_api_budget: 100
    
    # Individual pass settings
    channel_template:
      confidence_threshold: 0.85
      timeout_seconds: 10.0
    web_search:
      enabled: true
      confidence_threshold: 0.7
      api_budget_limit: 20
```

## Multi-Pass Parsing System

The collector includes an advanced multi-pass parsing ladder system for maximum accuracy across diverse karaoke video formats. This system progressively applies different parsing strategies until a confident result is achieved.

### Quick Setup

```python
from collector import KaraokeCollector, CollectorConfig
from collector.config import load_config

# Load configuration and enable multi-pass parsing
config = load_config("config.yaml")
config.search.multi_pass.enabled = True

# Initialize with multi-pass enabled
collector = KaraokeCollector(config)

# Parse a challenging title
result = await collector.multi_pass_controller.parse_video(
    video_id="abc123",
    title='Sing King Karaoke - "Bohemian Rhapsody" (Style of "Queen")',
    channel_name="Sing King Karaoke",
    channel_id="UC1234567890"
)

# Check results
if result.final_result:
    print(f"Artist: {result.final_result.original_artist}")
    print(f"Song: {result.final_result.song_title}")
    print(f"Confidence: {result.final_confidence:.2f}")
    print(f"Method: {result.stopped_at_pass.value}")
```

### System Architecture

The multi-pass system implements intelligent progression through five specialized passes:

1. **Pass 0 - Channel Template**: Channel-specific learned patterns with drift detection
2. **Pass 1 - Auto-retemplate**: Temporal pattern analysis for format changes
3. **Pass 2 - ML/Embedding**: Semantic similarity with optional transformer models
4. **Pass 3 - Web Search**: SERP integration with intelligent query cleaning
5. **Pass 4 - Acoustic**: Audio fingerprinting (placeholder for future implementation)

### Configuration Examples

#### High-Accuracy Mode (Thorough)
```yaml
search:
  multi_pass:
    enabled: true
    stop_on_first_success: false  # Try all passes
    total_cpu_budget: 120.0
    channel_template:
      confidence_threshold: 0.9
```

#### Fast Mode (Speed-Optimized)
```yaml
search:
  multi_pass:
    enabled: true
    total_cpu_budget: 30.0
    auto_retemplate:
      enabled: false
    acoustic_fingerprint:
      enabled: false
```

#### Web Search Focused
```yaml
search:
  multi_pass:
    enabled: true
    web_search:
      confidence_threshold: 0.6
      api_budget_limit: 50
      timeout_seconds: 180.0
```

### Performance Monitoring

```python
# Get comprehensive statistics
stats = collector.multi_pass_controller.get_statistics()
print(f"Success rate: {stats['success_rates']}")
print(f"Average processing time: {stats['average_processing_time']}")

# Monitor individual pass performance
channel_stats = collector.multi_pass_controller.channel_template_pass.get_statistics()
print(f"Learned patterns: {channel_stats['total_learned_patterns']}")
print(f"Drift detected: {channel_stats['channels_with_drift']} channels")
```

### Integration Files

- **Complete Guide**: See `MULTI_PASS_GUIDE.md` for detailed documentation
- **Integration Example**: `collector/multi_pass_integration_example.py` for setup examples
- **Test Suite**: `test_multi_pass_system.py` and `test_multi_pass_edge_cases.py` for comprehensive testing

The multi-pass system is designed to be backward-compatible and disabled by default. Enable it when you need maximum parsing accuracy for challenging video titles.

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
- `channels` - YouTube channel information and processing state
- `video_features` - Karaoke-specific features with confidence scores
- `quality_scores` - Technical, engagement, and metadata quality metrics
- `ryd_data` - Return YouTube Dislike data with confidence scoring
- `search_history` - Track search performance over time
- `error_log` - Debugging and monitoring
- `search_cache` - Cached search results for faster subsequent lookups
- `search_analytics` - Aggregated statistics on query performance
- `fuzzy_reference_data` - Known artists and songs used for fuzzy matching
- `validation_results` - Data validation outcomes and issues

### Performance Optimizations:
✅ **15+ database indexes** - Optimized for large-scale queries  
✅ **Connection pooling** - Handles high-concurrency operations  
✅ **Memory management** - Bounded caches prevent memory exhaustion  
✅ **Rate limiting** - Prevents API blocking with smart backoff  
✅ **Incremental processing** - Only processes new videos on subsequent runs

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
# Install development dependencies (includes pytest-cov for coverage)
pip install -e ".[dev]"

# Keep yt-dlp updated for YouTube API changes
pip install --upgrade yt-dlp

# Run tests with coverage
pytest --cov=collector

# Run linting with auto-fix
ruff check . --fix

# Format code with black
black .

# Type checking (if using mypy/pylance)
# VS Code with Pylance provides excellent type checking support

# Install pre-commit hooks (if available)
pre-commit install
```

### Code Quality Tools

The project uses several tools to maintain code quality:

- **pytest** - Testing framework
- **pytest-cov** - Coverage reporting plugin for pytest
- **ruff** - Fast Python linter with auto-fix capabilities
- **black** - Code formatting for consistent style
- **Pylance/mypy** - Type checking for better code reliability

### Debugging and Development Tools

- **debug_extraction.py** - Debug script for testing title parsing patterns
- **Comprehensive test suite** - Unit tests for all major components
- **Database migration system** - Automatic schema updates
- **Performance monitoring** - Built-in statistics tracking

### 🔄 **Staying Current with YouTube Changes**

YouTube often changes its internal APIs. Keep your collector working by upgrading yt-dlp:

```bash
# Update to the latest yt-dlp
pip install --upgrade yt-dlp

# Check for breaking changes
karaoke-collector collect -q "test" -m 5 --verbose
```

**Pro Tip:** yt-dlp releases nightly builds that patch new YouTube restrictions. The collector automatically benefits from these updates! 🚀

## Recent Improvements

- ✅ **Multi-pass parsing ladder** - Advanced 5-pass system with confidence-based progression and budget management
- ✅ **Channel drift detection** - Automatically detects and adapts to changing channel patterns
- ✅ **SERP integration** - Web search with intelligent query cleaning and caching
- ✅ **Enhanced ML/embedding support** - Optional sentence-transformers with semantic similarity
- ✅ **Comprehensive testing** - Complete test suite for multi-pass system with edge case coverage
- ✅ **Enhanced regex patterns** - Fixed artist-song extraction for better accuracy
- ✅ **Type safety improvements** - Resolved Pylance type checking issues  
- ✅ **Code formatting** - Applied black formatting for consistent style
- ✅ **Advanced parser stability** - Improved PatternStats dataclass handling

## Testing

The package includes comprehensive tests with coverage reporting:

```bash
# Run all tests with coverage (configured in pytest.ini)
pytest

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_db.py -v

# Run tests with coverage report
pytest --cov-report=html

# Skip slow tests
pytest -m "not slow"
```

### Test Configuration
Tests are configured with:
- Coverage reporting (HTML and terminal)  
- Strict markers for test categorization
- Support for unit/integration test separation
- Warning suppression for cleaner output

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

This project is licensed under the MIT License. Users remain responsible for ensuring compliance with applicable laws and terms of service.
