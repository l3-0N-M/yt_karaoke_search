"""Integration example for the multi-pass parsing ladder system."""

import asyncio
import logging

from .advanced_parser import AdvancedTitleParser
from .config import load_config
from .enhanced_search import MultiStrategySearchEngine
from .multi_pass_controller import MultiPassParsingController

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_multi_pass_parsing():
    """Demonstrate the multi-pass parsing ladder system."""

    print("=== Multi-Pass Parsing Ladder Demo ===\n")

    # Step 1: Load configuration
    print("1. Loading configuration...")
    config = load_config("config.yaml")  # Load from config file, or use defaults

    # Enable multi-pass parsing in config
    config.search.multi_pass.enabled = True
    config.search.multi_pass.stop_on_first_success = True
    config.search.multi_pass.total_cpu_budget = 60.0
    config.search.multi_pass.total_api_budget = 100

    print(f"   Multi-pass enabled: {config.search.multi_pass.enabled}")
    print(f"   CPU budget: {config.search.multi_pass.total_cpu_budget}s")
    print(f"   API budget: {config.search.multi_pass.total_api_budget} calls\n")

    # Step 2: Initialize components
    print("2. Initializing components...")

    # Advanced parser (existing component)
    advanced_parser = AdvancedTitleParser(config.search)
    print(f"   Advanced parser initialized with {len(advanced_parser.core_patterns)} patterns")

    # Enhanced search engine (optional, for Pass 3)
    search_engine = None
    if config.search.use_multi_strategy:
        search_engine = MultiStrategySearchEngine(
            config.search, config.scraping, None  # db_manager=None for demo
        )
        print("   Enhanced search engine initialized")
    else:
        print("   Enhanced search engine not enabled")

    # Multi-pass controller
    controller = MultiPassParsingController(
        config=config.search.multi_pass,
        advanced_parser=advanced_parser,
        search_engine=search_engine,
        db_manager=None,  # Would use real database in production
    )
    print("   Multi-pass controller initialized\n")

    # Step 3: Test with various karaoke video titles
    test_titles = [
        {
            "title": 'Sing King Karaoke - "Bohemian Rhapsody" (Style of "Queen")',
            "description": "High quality karaoke version",
            "channel_name": "Sing King Karaoke",
            "channel_id": "UC1234567890",
            "expected_pass": "Pass 0 (Channel Template)",
        },
        {
            "title": '"Hotel California" - "Eagles" - Zoom Karaoke',
            "description": "Classic rock karaoke with scrolling lyrics",
            "channel_name": "Zoom Karaoke",
            "channel_id": "UC0987654321",
            "expected_pass": "Pass 1 (Auto-retemplate)",
        },
        {
            "title": "Amazing Grace Karaoke Version Instrumental Track",
            "description": "Traditional hymn backing track",
            "channel_name": "Karaoke Mugen",
            "channel_id": "UC1122334455",
            "expected_pass": "Pass 2 (ML/Embedding)",
        },
        {
            "title": "Sweet Child O Mine by Guns N Roses Guitar Hero Style",
            "description": "Rock anthem for karaoke night",
            "channel_name": "Random Music Channel",
            "channel_id": "UC9988776655",
            "expected_pass": "Pass 3 (Web Search)",
        },
        {
            "title": "Unknown Song with Strange Title Format [HD]",
            "description": "Mysterious karaoke track",
            "channel_name": "Unknown Channel",
            "channel_id": "UC5544332211",
            "expected_pass": "Multiple passes, may fail",
        },
    ]

    print("3. Testing multi-pass parsing with sample titles:\n")

    total_processing_time = 0.0
    successful_parses = 0

    for i, test_case in enumerate(test_titles, 1):
        print(f"Test {i}: {test_case['expected_pass']}")
        print(f"Title: {test_case['title']}")
        print(f"Channel: {test_case['channel_name']}")

        # Parse with multi-pass system
        result = await controller.parse_video(
            video_id=f"test_video_{i}",
            title=test_case["title"],
            description=test_case["description"],
            channel_name=test_case["channel_name"],
            channel_id=test_case["channel_id"],
        )

        # Display results
        if result.final_result:
            successful_parses += 1
            print(f"✓ SUCCESS - Confidence: {result.final_confidence:.2f}")
            print(f"  Artist: {result.final_result.original_artist}")
            print(f"  Song: {result.final_result.song_title}")
            print(f"  Method: {result.final_result.method}")
            print(
                f"  Stopped at: {result.stopped_at_pass.value if result.stopped_at_pass else 'None'}"
            )
        else:
            print("✗ FAILED - No result")

        print(f"  Passes attempted: {len(result.passes_attempted)}")
        print(f"  Processing time: {result.total_processing_time:.2f}s")
        print(f"  CPU budget used: {result.budget_consumed.get('cpu', 0):.2f}s")
        print(f"  API budget used: {result.budget_consumed.get('api', 0)} calls")

        total_processing_time += result.total_processing_time
        print()

    # Step 4: Display system statistics
    print("4. System Performance Statistics:")
    stats = controller.get_statistics()

    print(f"   Videos processed: {stats['total_videos_processed']}")
    print(
        f"   Success rate: {successful_parses}/{len(test_titles)} ({successful_parses/len(test_titles)*100:.1f}%)"
    )
    print(f"   Average processing time: {total_processing_time/len(test_titles):.2f}s")
    print(
        f"   Average CPU per video: {stats.get('budget_efficiency', {}).get('avg_cpu_per_video', 0):.2f}s"
    )
    print(
        f"   Average API per video: {stats.get('budget_efficiency', {}).get('avg_api_per_video', 0):.1f} calls"
    )

    print("\n   Pass Success Rates:")
    for pass_name, success_rate in stats.get("success_rates", {}).items():
        attempts = stats["passes_attempted"].get(pass_name.upper(), 0)
        successes = stats["passes_successful"].get(pass_name.upper(), 0)
        print(f"     {pass_name}: {success_rate:.1%} ({successes}/{attempts})")

    # Step 5: Individual pass statistics
    print("\n5. Individual Pass Statistics:")

    # Channel template pass stats
    if hasattr(controller, "channel_template_pass"):
        channel_stats = controller.channel_template_pass.get_statistics()
        print("   Pass 0 (Channel Template):")
        print(f"     Channels tracked: {channel_stats.get('total_channels', 0)}")
        print(f"     Patterns learned: {channel_stats.get('total_learned_patterns', 0)}")
        print(f"     Drift detected: {channel_stats.get('channels_with_drift', 0)} channels")

    # Auto-retemplate pass stats
    if hasattr(controller, "auto_retemplate_pass"):
        retemplate_stats = controller.auto_retemplate_pass.get_statistics()
        print("   Pass 1 (Auto-retemplate):")
        print(f"     Active patterns: {retemplate_stats.get('total_active_patterns', 0)}")
        print(f"     Deprecated patterns: {retemplate_stats.get('total_deprecated_patterns', 0)}")
        print(
            f"     Pattern changes: {retemplate_stats.get('channels_with_pattern_changes', 0)} channels"
        )

    # ML embedding pass stats
    if hasattr(controller, "ml_embedding_pass"):
        ml_stats = controller.ml_embedding_pass.get_statistics()
        print("   Pass 2 (ML/Embedding):")
        print(f"     Embedding model: {ml_stats.get('embedding_model_name', 'None')}")
        print(f"     Artist candidates: {ml_stats.get('artist_candidates', 0)}")
        print(f"     Song candidates: {ml_stats.get('song_candidates', 0)}")
        print(f"     Cache size: {ml_stats.get('embedding_cache_size', 0)}")

    # Web search pass stats
    if hasattr(controller, "web_search_pass") and controller.web_search_pass:
        search_stats = controller.web_search_pass.get_statistics()
        print("   Pass 3 (Web Search):")
        print(f"     Cache hit rate: {search_stats.get('cache_hit_rate', 0):.1%}")
        print(
            f"     Cache entries: {search_stats.get('cache_statistics', {}).get('total_entries', 0)}"
        )
        print(f"     Search success rate: {search_stats.get('success_rate', 0):.1%}")
    else:
        print("   Pass 3 (Web Search): Not enabled (requires search engine)")

    # Acoustic fingerprint pass stats
    if hasattr(controller, "acoustic_fingerprint_pass"):
        acoustic_stats = controller.acoustic_fingerprint_pass.get_statistics()
        print("   Pass 4 (Acoustic Fingerprint):")
        print(f"     Enabled: {acoustic_stats.get('enabled', False)}")
        print(f"     Requests: {acoustic_stats.get('total_requests', 0)}")
        if not acoustic_stats.get("enabled", False):
            print("     Status: Placeholder implementation")

    print("\n=== Demo Complete ===")


async def demonstrate_configuration_options():
    """Demonstrate various configuration options for multi-pass parsing."""

    print("\n=== Configuration Options Demo ===\n")

    # Load base configuration
    config = load_config()

    print("1. Default Configuration:")
    print(f"   Multi-pass enabled: {config.search.multi_pass.enabled}")
    print(f"   Stop on first success: {config.search.multi_pass.stop_on_first_success}")
    print(f"   Global timeout: {config.search.multi_pass.global_timeout_seconds}s")
    print(f"   Total CPU budget: {config.search.multi_pass.total_cpu_budget}s")
    print(f"   Total API budget: {config.search.multi_pass.total_api_budget} calls")

    print("\n2. Per-Pass Configuration:")
    passes = [
        ("Channel Template", config.search.multi_pass.channel_template),
        ("Auto-retemplate", config.search.multi_pass.auto_retemplate),
        ("ML/Embedding", config.search.multi_pass.ml_embedding),
        ("Web Search", config.search.multi_pass.web_search),
        ("Acoustic Fingerprint", config.search.multi_pass.acoustic_fingerprint),
    ]

    for pass_name, pass_config in passes:
        print(f"   {pass_name}:")
        print(f"     Enabled: {pass_config.enabled}")
        print(f"     Confidence threshold: {pass_config.confidence_threshold}")
        print(f"     Timeout: {pass_config.timeout_seconds}s")
        print(f"     CPU budget: {pass_config.cpu_budget_limit}s")
        print(f"     API budget: {pass_config.api_budget_limit} calls")
        print(f"     Max retries: {pass_config.max_retries}")

    print("\n3. Customization Examples:")

    # Example 1: High-accuracy mode
    print("   High-Accuracy Mode (slower, more thorough):")
    config.search.multi_pass.enabled = True
    config.search.multi_pass.stop_on_first_success = False  # Try all passes
    config.search.multi_pass.total_cpu_budget = 120.0  # More time
    config.search.multi_pass.channel_template.confidence_threshold = 0.9  # Higher bar
    config.search.multi_pass.ml_embedding.timeout_seconds = 120.0  # More time for ML
    print("     - Disabled early stopping")
    print("     - Increased CPU budget to 120s")
    print("     - Higher confidence thresholds")
    print("     - Longer timeouts for complex passes")

    # Example 2: Fast mode
    print("\n   Fast Mode (quicker, less thorough):")
    config.search.multi_pass.total_cpu_budget = 30.0  # Less time
    config.search.multi_pass.channel_template.timeout_seconds = 5.0  # Shorter timeouts
    config.search.multi_pass.auto_retemplate.enabled = False  # Skip some passes
    config.search.multi_pass.acoustic_fingerprint.enabled = False  # Skip expensive pass
    print("     - Reduced CPU budget to 30s")
    print("     - Shorter timeouts")
    print("     - Disabled expensive passes")

    # Example 3: Web search focused
    print("\n   Web Search Focused (for difficult titles):")
    config.search.multi_pass.web_search.confidence_threshold = 0.6  # Lower bar
    config.search.multi_pass.web_search.api_budget_limit = 50  # More API calls
    config.search.multi_pass.web_search.timeout_seconds = 180.0  # More time
    print("     - Lower confidence threshold for web search")
    print("     - Increased API budget for searches")
    print("     - Extended timeout for web operations")


def demonstrate_yaml_configuration():
    """Show example YAML configuration for multi-pass parsing."""

    print("\n=== YAML Configuration Example ===\n")

    yaml_config = """
# Multi-pass parsing configuration example
search:
  # Basic search settings
  use_multi_strategy: true
  max_results_per_query: 100

  # Fuzzy matching configuration
  fuzzy_matching:
    min_similarity: 0.7
    min_phonetic: 0.8
    max_edit_distance: 3

  # Multi-pass parsing ladder
  multi_pass:
    # Global settings
    enabled: true
    max_total_retries: 5
    global_timeout_seconds: 300.0
    stop_on_first_success: true

    # Budget management
    total_cpu_budget: 60.0    # seconds per video
    total_api_budget: 100     # API calls per video

    # Backoff and retry
    base_retry_delay: 1.0
    max_retry_delay: 300.0
    retry_exponential_base: 2.0

    # Pass 0: Channel template matching
    channel_template:
      enabled: true
      confidence_threshold: 0.85
      timeout_seconds: 10.0
      cpu_budget_limit: 2.0
      api_budget_limit: 0
      max_retries: 3
      exponential_backoff_base: 2.0
      exponential_backoff_max: 60.0

    # Pass 1: Auto-retemplate on recent uploads
    auto_retemplate:
      enabled: true
      confidence_threshold: 0.8
      timeout_seconds: 30.0
      cpu_budget_limit: 5.0
      api_budget_limit: 2
      max_retries: 3

    # Pass 2: ML/embedding similarity
    ml_embedding:
      enabled: true
      confidence_threshold: 0.75
      timeout_seconds: 60.0
      cpu_budget_limit: 10.0
      api_budget_limit: 5
      max_retries: 3

    # Pass 3: Web search with query cleaning
    web_search:
      enabled: true
      confidence_threshold: 0.7
      timeout_seconds: 120.0
      cpu_budget_limit: 15.0
      api_budget_limit: 20
      max_retries: 3

    # Pass 4: Acoustic fingerprint (experimental)
    acoustic_fingerprint:
      enabled: false  # Disabled by default
      confidence_threshold: 0.9
      timeout_seconds: 300.0
      cpu_budget_limit: 60.0
      api_budget_limit: 50
      max_retries: 2
"""

    print("Save this configuration to 'config.yaml':")
    print(yaml_config)

    print("\nConfiguration Guidelines:")
    print("- Set 'enabled: false' to disable multi-pass parsing")
    print("- Adjust confidence thresholds based on your accuracy needs")
    print("- Lower thresholds = more results, potentially lower quality")
    print("- Higher budgets = more thorough analysis, slower processing")
    print("- Enable stop_on_first_success for faster processing")
    print("- Disable expensive passes (web_search, acoustic) for speed")


async def main():
    """Main demonstration function."""

    try:
        # Run the main demo
        await demonstrate_multi_pass_parsing()

        # Show configuration options
        await demonstrate_configuration_options()

        # Show YAML configuration
        demonstrate_yaml_configuration()

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())
