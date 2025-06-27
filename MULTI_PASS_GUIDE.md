# Multi-Pass Parsing Ladder Guide

## Overview

The Multi-Pass Parsing Ladder is an advanced parsing system that progressively applies different parsing strategies to extract artist and song information from karaoke video titles. It implements a 5-pass ladder with confidence-based stopping, budget management, and intelligent retry logic.

## Architecture

### Pass Hierarchy

```
Pass 0: Channel Template Match     (Threshold: 0.85) - Fast, channel-specific patterns
Pass 1: Auto-retemplate           (Threshold: 0.8)  - Recent upload pattern learning  
Pass 2: ML/Embedding Similarity   (Threshold: 0.75) - Semantic and fuzzy matching
Pass 3: Web Search Integration    (Threshold: 0.7)  - SERP-based parsing with caching
Pass 4: Acoustic Fingerprint      (Threshold: 0.9)  - Audio analysis (placeholder)
```

### Key Features

- **Confidence-based progression**: Stops at first pass meeting threshold
- **Budget management**: CPU time and API call limits per video
- **Exponential backoff**: Intelligent retry with increasing delays
- **Drift detection**: Automatically detects when channel patterns change
- **SERP caching**: 1-week TTL cache for web search results
- **Language-aware**: Multi-language filler word removal
- **Statistics tracking**: Comprehensive performance monitoring

## Quick Start

### 1. Basic Integration

```python
from collector.multi_pass_controller import MultiPassParsingController
from collector.advanced_parser import AdvancedTitleParser
from collector.config import load_config

# Load configuration
config = load_config("config.yaml")
config.search.multi_pass.enabled = True

# Initialize components
advanced_parser = AdvancedTitleParser(config.search)
controller = MultiPassParsingController(
    config=config.search.multi_pass,
    advanced_parser=advanced_parser,
    search_engine=search_engine,  # Optional for Pass 3
    db_manager=db_manager          # Optional for persistence
)

# Parse a video title
result = await controller.parse_video(
    video_id="abc123",
    title="Sing King Karaoke - \"Bohemian Rhapsody\" (Style of \"Queen\")",
    description="High quality karaoke version",
    channel_name="Sing King Karaoke",
    channel_id="UC1234567890"
)

# Check results
if result.final_result:
    print(f"Artist: {result.final_result.original_artist}")
    print(f"Song: {result.final_result.song_title}")
    print(f"Confidence: {result.final_confidence:.2f}")
    print(f"Stopped at: {result.stopped_at_pass.value}")
```

### 2. Configuration

Create `config.yaml` with multi-pass settings:

```yaml
search:
  multi_pass:
    enabled: true
    stop_on_first_success: true
    total_cpu_budget: 60.0
    total_api_budget: 100
    
    channel_template:
      confidence_threshold: 0.85
      timeout_seconds: 10.0
      cpu_budget_limit: 2.0
      
    web_search:
      enabled: true
      confidence_threshold: 0.7
      timeout_seconds: 120.0
      api_budget_limit: 20
```

### 3. Enhanced Search Integration

For Pass 3 (web search), initialize with enhanced search engine:

```python
from collector.enhanced_search import MultiStrategySearchEngine

search_engine = MultiStrategySearchEngine(
    config.search, config.scraping, db_manager
)

controller = MultiPassParsingController(
    config=config.search.multi_pass,
    advanced_parser=advanced_parser,
    search_engine=search_engine,  # Enables Pass 3
    db_manager=db_manager
)
```

## Pass Details

### Pass 0: Enhanced Channel Template Match

**Purpose**: Channel-specific pattern recognition with learning
**Speed**: Very fast (< 2s)
**API Calls**: 0

**Features**:
- Learns channel-specific title formats
- Tracks pattern effectiveness over time  
- Detects channel format drift
- Enhanced context detection for karaoke channels

**Example Patterns**:
- `Sing King Karaoke - "Artist" (Style of "Song")`
- `"Artist" - "Song" - Zoom Karaoke`
- `[Channel] Artist - Song [Karaoke]`

### Pass 1: Auto-retemplate on Recent Uploads

**Purpose**: Temporal pattern analysis and evolution
**Speed**: Fast (< 5s)
**API Calls**: Low (0-2)

**Features**:
- Tracks pattern changes over time
- Automatically deprecates old patterns
- Revives patterns when they work again
- Learns from recent upload patterns

**Use Cases**:
- Channels that change their title format
- Seasonal or promotional title variations
- Channel rebranding detection

### Pass 2: Enhanced ML/Embedding Similarity

**Purpose**: Semantic understanding and fuzzy matching
**Speed**: Medium (< 10s)
**API Calls**: Low (0-5)

**Features**:
- Optional sentence-transformers embeddings
- Advanced fuzzy matching with phonetic similarity
- Entity extraction from titles
- Hybrid semantic + string similarity

**Dependencies**:
- `sentence-transformers` (optional, for embeddings)
- `scikit-learn` (optional, for cosine similarity)
- Built-in fuzzy matching as fallback

### Pass 3: Enhanced Web Search Integration

**Purpose**: Search-engine-assisted parsing
**Speed**: Slow (< 15s)
**API Calls**: High (5-20)

**Features**:
- Multi-strategy query generation
- Language-aware filler word removal
- SERP result caching (1-week TTL)
- Consensus-based result selection

**Query Strategies**:
1. Cleaned original title
2. Description + title context
3. Quoted text extraction
4. Language-specific cleaning
5. Minimal cleaning (fallback)

### Pass 4: Acoustic Fingerprint Batch

**Purpose**: Audio-based song identification
**Speed**: Very slow (< 60s)
**API Calls**: High (10-50)

**Status**: Placeholder implementation
**Future Libraries**: pyacoustid, aubio, chromaprint

## Configuration Reference

### Global Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `false` | Enable multi-pass parsing |
| `stop_on_first_success` | `true` | Stop at first successful pass |
| `global_timeout_seconds` | `300.0` | Maximum total processing time |
| `total_cpu_budget` | `60.0` | CPU seconds per video |
| `total_api_budget` | `100` | API calls per video |

### Per-Pass Settings

Each pass supports these configuration options:

| Setting | Description | Typical Range |
|---------|-------------|---------------|
| `enabled` | Enable this pass | `true`/`false` |
| `confidence_threshold` | Minimum confidence to accept result | `0.6` - `0.9` |
| `timeout_seconds` | Maximum time for this pass | `5.0` - `300.0` |
| `cpu_budget_limit` | CPU seconds budget | `1.0` - `60.0` |
| `api_budget_limit` | API calls budget | `0` - `50` |
| `max_retries` | Retry attempts on failure | `1` - `5` |

### Configuration Presets

#### High Accuracy Mode
```yaml
multi_pass:
  enabled: true
  stop_on_first_success: false  # Try all passes
  total_cpu_budget: 120.0
  channel_template:
    confidence_threshold: 0.9
  ml_embedding:
    timeout_seconds: 120.0
```

#### Fast Mode
```yaml
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
multi_pass:
  enabled: true
  web_search:
    confidence_threshold: 0.6
    api_budget_limit: 50
    timeout_seconds: 180.0
```

## Performance Monitoring

### Statistics Collection

```python
# Get overall statistics
stats = controller.get_statistics()
print(f"Success rate: {stats['success_rates']}")
print(f"Average processing time: {stats['average_processing_time']}")

# Get individual pass statistics
channel_stats = controller.channel_template_pass.get_statistics()
print(f"Patterns learned: {channel_stats['total_learned_patterns']}")

ml_stats = controller.ml_embedding_pass.get_statistics() 
print(f"Embedding model: {ml_stats['embedding_model_name']}")

search_stats = controller.web_search_pass.get_statistics()
print(f"Cache hit rate: {search_stats['cache_hit_rate']}")
```

### Key Metrics

- **Pass success rates**: Percentage of videos each pass successfully parses
- **Budget efficiency**: Average CPU/API consumption per video
- **Cache performance**: Hit rates for SERP and embedding caches
- **Drift detection**: Number of channels with format changes
- **Processing times**: Average and percentile processing times

## Troubleshooting

### Common Issues

#### Low Success Rates
```yaml
# Solution: Lower confidence thresholds
channel_template:
  confidence_threshold: 0.7  # From 0.85
ml_embedding:
  confidence_threshold: 0.6   # From 0.75
```

#### Timeout Issues
```yaml
# Solution: Increase timeouts and budgets
global_timeout_seconds: 600.0  # From 300.0
total_cpu_budget: 120.0        # From 60.0
web_search:
  timeout_seconds: 240.0       # From 120.0
```

#### High API Usage
```yaml
# Solution: Reduce API budgets and enable caching
total_api_budget: 50          # From 100
web_search:
  api_budget_limit: 10        # From 20
```

#### Poor Channel Pattern Learning
```python
# Check pattern statistics
stats = controller.channel_template_pass.get_statistics()
print(f"Channels with drift: {stats['channels_with_drift']}")

# Reset patterns for problematic channels
controller.channel_template_pass._refresh_channel_patterns(channel_id)
```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger('collector.multi_pass_controller').setLevel(logging.DEBUG)
logging.getLogger('collector.passes').setLevel(logging.DEBUG)
```

### Performance Optimization

#### For High Volume Processing
```yaml
multi_pass:
  stop_on_first_success: true
  total_cpu_budget: 30.0
  acoustic_fingerprint:
    enabled: false
```

#### For Accuracy-Critical Applications
```yaml
multi_pass:
  stop_on_first_success: false
  total_cpu_budget: 120.0
  max_total_retries: 10
```

## Integration Examples

See `multi_pass_integration_example.py` for:
- Complete setup demonstration
- Configuration examples
- Performance monitoring
- Error handling patterns

## Future Enhancements

### Planned Features
1. **Real-time drift detection**: Automated alerts and pattern refresh
2. **Machine learning optimization**: Auto-tuning of confidence thresholds  
3. **Advanced caching**: Multi-level cache with LRU eviction
4. **Batch processing**: Optimized processing for large video collections
5. **Custom pass plugins**: User-defined parsing strategies

### Acoustic Fingerprint Implementation
When implementing Pass 4, consider:
- **pyacoustid**: MusicBrainz integration
- **aubio**: Real-time audio analysis
- **chromaprint**: Fast fingerprint generation
- **Database design**: Efficient fingerprint storage and retrieval
- **Batch processing**: Background fingerprint generation

## Best Practices

1. **Start with defaults**: Enable multi-pass with default settings
2. **Monitor performance**: Track success rates and processing times
3. **Tune gradually**: Adjust one parameter at a time
4. **Use appropriate mode**: Fast for real-time, accurate for batch
5. **Cache optimization**: Monitor cache hit rates
6. **Budget management**: Set appropriate CPU/API limits
7. **Channel-specific tuning**: Different settings per channel type

## Support

For questions and issues:
1. Check logs with DEBUG level enabled
2. Review configuration against examples
3. Monitor statistics for performance insights
4. Test with integration example code