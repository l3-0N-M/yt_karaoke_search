# YouTube Karaoke Search

A sophisticated Python application for collecting, parsing, and analyzing karaoke videos from YouTube. The project uses advanced multi-pass parsing techniques, external music databases, and machine learning to accurately extract artist names, song titles, and metadata from karaoke video titles.

## Features

- **Multi-Pass Parsing System**: Intelligent parsing pipeline that learns from channel patterns and validates against music databases
- **Multiple Data Sources**: Integration with MusicBrainz, Discogs, and web search APIs for metadata enrichment
- **Advanced Search**: Multi-provider search with intelligent caching and result ranking
- **Async Architecture**: High-performance async/await patterns for concurrent operations
- **Comprehensive CLI**: Rich command-line interface for various collection and analysis tasks
- **Machine Learning Support**: Optional ML-based parsing using sentence transformers
- **Smart Caching**: Three-tier caching system for optimized performance

## Installation

### Requirements

- Python 3.8 or higher
- SQLite 3.35+ (for advanced features)
- yt-dlp

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/yt_karaoke_search.git
cd yt_karaoke_search

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Optional Dependencies

For machine learning features:
```bash
pip install -r requirements-ml.txt
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required for Discogs integration
DISCOGS_TOKEN=your_discogs_token

# Optional API keys for enhanced search
BING_API_KEY=your_bing_api_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CX=your_google_custom_search_id

# Optional for web scraping
BROWSERLESS_API_KEY=your_browserless_key
```

### Configuration File

Generate a configuration template:

```bash
python -m collector.cli create-config
```

This creates a `config.yaml` file with customizable options for:
- Search parameters
- Parsing rules
- Cache settings
- API configurations
- Processing options

## Usage

### Basic Commands

```bash
# Check environment setup
python -m collector.cli check-env

# Collect karaoke videos using search queries
python -m collector.cli collect "artist name" --limit 50

# Collect from specific YouTube channel
python -m collector.cli collect-channel "CHANNEL_ID"

# Batch collection from multiple channels
python -m collector.cli collect-channels channels.txt

# View database statistics
python -m collector.cli stats
```

### Advanced Usage

```bash
# Use specific search provider
python -m collector.cli collect "song title" --provider bing

# Enable all parsing passes
python -m collector.cli collect "artist" --enable-all-passes

# Custom configuration
python -m collector.cli collect "query" --config custom-config.yaml
```

## Architecture

### Core Components

- **`collector/cli.py`**: Command-line interface
- **`collector/main.py`**: Main application logic and video processor
- **`collector/advanced_parser.py`**: Multi-pass parsing system
- **`collector/multi_pass_controller.py`**: Orchestrates parsing passes
- **`collector/database.py`**: Database operations and schema management
- **`collector/search/`**: Search providers and result management
- **`collector/passes/`**: Individual parsing pass implementations

### Parsing Passes

1. **ChannelTemplatePass**: Learns parsing patterns from specific channels
2. **AutoRetemplatePass**: Automatically generates parsing templates
3. **MusicBrainzSearchPass**: Validates against MusicBrainz database
4. **DiscogsSearchPass**: Enriches with Discogs metadata (95% success rate)
5. **MLEmbeddingPass**: Uses ML for semantic similarity matching
6. **WebSearchPass**: Fallback web search for difficult cases

### Database Schema

The SQLite database includes:
- `videos`: Core video metadata and parsed information
- `channels`: YouTube channel information
- `musicbrainz_cache`: Cached MusicBrainz lookups
- `parse_failures`: Failed parsing attempts for analysis

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=collector

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Run integration tests only
```

### Code Style

The project follows PEP 8 guidelines. Run linting:

```bash
# Type checking
mypy collector/

# Linting
flake8 collector/
```

## Performance Considerations

- **Async Operations**: All I/O operations are async for maximum concurrency
- **Intelligent Caching**: Three-tier cache reduces API calls
- **Batch Processing**: Efficient batch operations for database writes
- **Connection Pooling**: Reuses HTTP connections for better performance

## Troubleshooting

### Common Issues

1. **Database locked errors**: Ensure only one instance is running
2. **API rate limits**: Configure appropriate delays in `config.yaml`
3. **Parsing failures**: Enable additional passes or adjust templates
4. **Memory usage**: Adjust batch sizes for large collections

### Debug Mode

Enable verbose logging:
```bash
python -m collector.cli collect "query" --log-level DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Uses [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube interaction
- Integrates with [MusicBrainz](https://musicbrainz.org/) and [Discogs](https://www.discogs.com/) APIs
- Built with [Click](https://click.palletsprojects.com/) for CLI functionality