"""Command-line interface for the karaoke collector."""

import asyncio
import logging
import sys

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv not installed, skip loading
    pass

try:
    import click
except ImportError:  # pragma: no cover - fallback minimal stub
    from types import SimpleNamespace

    def _nop_decorator(*dargs, **dkwargs):
        def _decorator(func):
            return func

        return _decorator

    click = SimpleNamespace(
        group=_nop_decorator,
        version_option=_nop_decorator,
        command=_nop_decorator,
        option=_nop_decorator,
        Path=lambda *a, **k: str,
    )

from collector.config import CollectorConfig, load_config, save_config_template
from collector.main import KaraokeCollector
from collector.utils import setup_logging


async def _collect_async(collector: KaraokeCollector, queries, max_per_query: int) -> None:
    """Helper to run video collection and print statistics."""
    try:
        total_processed = await collector.collect_videos(list(queries), max_per_query)
        stats = await collector.get_statistics()

        print("\n" + "=" * 50)
        print("COLLECTION RESULTS")
        print("=" * 50)
        print(f"Total videos processed: {total_processed}")
        print(f"Total videos in database: {stats.get('total_videos', 0):,}")
        print(f"Videos with artist info: {stats.get('videos_with_artist', 0):,}")
        print(f"Average confidence score: {stats.get('avg_confidence', 0):.2f}")

        if top_artists := stats.get("top_artists", []):
            print("\nTop 10 Artists:")
            for artist, count, avg_views in top_artists[:10]:
                print(f"  {artist}: {count} videos (avg {avg_views:,.0f} views)")
    finally:
        await collector.cleanup()


async def _collect_channel_async(
    collector: KaraokeCollector, channel_url: str, max_videos, incremental: bool
) -> None:
    """Helper to collect from a single channel and print statistics."""
    try:
        total_processed = await collector.collect_from_channel(channel_url, max_videos, incremental)
        stats = await collector.get_channel_statistics()

        print("\n" + "=" * 50)
        print("CHANNEL COLLECTION RESULTS")
        print("=" * 50)
        print(f"Videos processed: {total_processed}")
        print(f"Total channels in database: {stats.get('total_channels', 0)}")
        print(f"Total videos from channels: {stats.get('total_videos_from_channels', 0):,}")

        for channel in stats.get("channels", []):
            if channel_url in channel.get("channel_url", ""):
                print(f"\nChannel: {channel['channel_name']}")
                print(f"  Videos collected: {channel['collected_videos']}")
                print(f"  Karaoke focused: {'Yes' if channel['is_karaoke_focused'] else 'No'}")
                if channel["last_processed_at"]:
                    print(f"  Last processed: {channel['last_processed_at']}")
                break
    finally:
        await collector.cleanup()


async def _collect_channels_async(
    collector: KaraokeCollector, channel_urls, max_videos_per_channel
) -> None:
    """Helper to collect from multiple channels and print statistics."""
    try:
        total_processed = await collector.collect_from_channels(
            channel_urls, max_videos_per_channel
        )
        stats = await collector.get_channel_statistics()

        print("\n" + "=" * 50)
        print("MULTI-CHANNEL COLLECTION RESULTS")
        print("=" * 50)
        print(f"Total videos processed: {total_processed}")
        print(f"Channels processed: {len(channel_urls)}")
        print(f"Total channels in database: {stats.get('total_channels', 0)}")
        print(f"Total videos from channels: {stats.get('total_videos_from_channels', 0):,}")

        print("\nChannels breakdown:")
        for channel in stats.get("channels", [])[:10]:
            print(f"  {channel['channel_name']}: {channel['collected_videos']} videos")
    finally:
        await collector.cleanup()


@click.group()
@click.version_option("2.0.0")
def cli():
    """Karaoke Video Collector - Extract comprehensive karaoke video data from YouTube."""
    pass


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to configuration file")
@click.option("--queries", "-q", multiple=True, help="Search queries to use")
@click.option("--max-per-query", "-m", default=50, help="Max videos per query")
@click.option("--output-db", "-o", help="Output database path")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
@click.option("--dry-run", is_flag=True, help="Dry run mode (no database writes)")
@click.option(
    "--multi-strategy",
    is_flag=True,
    help="Use multi-strategy search engine",
)
@click.option(
    "--multi-pass",
    is_flag=True,
    help="Enable multi-pass parsing system",
)
def collect(
    config, queries, max_per_query, output_db, verbose, dry_run, multi_strategy, multi_pass
):
    """Collect karaoke videos from YouTube."""

    if config:
        collector_config = load_config(config)
    else:
        collector_config = CollectorConfig()

    # Override config with CLI options
    if output_db:
        collector_config.database.path = output_db
    if dry_run:
        collector_config.dry_run = True
    if multi_strategy:
        collector_config.search.use_multi_strategy = True
    if multi_pass:
        collector_config.search.multi_pass.enabled = True

    log_cfg = collector_config.logging
    log_level = logging.DEBUG if verbose else getattr(logging, log_cfg.level.upper(), logging.INFO)
    setup_logging(
        level=log_level,
        log_file=log_cfg.file_path,
        max_bytes=log_cfg.max_file_size_mb * 1024 * 1024,
        backup_count=log_cfg.backup_count,
        console_output=log_cfg.console_output,
    )

    # Default queries if none provided
    if not queries:
        queries = [
            "karaoke",
            "karaoke with lyrics",
            "piano karaoke",
            "acoustic karaoke",
            "pop karaoke",
            "rock karaoke",
        ]

    # Run collection
    collector = KaraokeCollector(collector_config)

    try:
        asyncio.run(_collect_async(collector, queries, max_per_query))

    except KeyboardInterrupt:
        logging.info("Collection interrupted by user")
    except Exception as e:
        logging.error(f"Collection failed: {e}")
        sys.exit(1)


@cli.command()
@click.argument("channel_url")
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to configuration file")
@click.option("--max-videos", "-m", type=int, help="Maximum videos to process from channel")
@click.option("--no-incremental", is_flag=True, help="Process all videos, not just new ones")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option(
    "--multi-strategy",
    is_flag=True,
    help="Use multi-strategy search engine",
)
@click.option(
    "--multi-pass",
    is_flag=True,
    help="Enable multi-pass parsing system",
)
def collect_channel(
    channel_url, config, max_videos, no_incremental, log_level, multi_strategy, multi_pass
):
    """Collect karaoke videos from a specific YouTube channel."""

    if config:
        collector_config = load_config(config)
    else:
        collector_config = CollectorConfig()

    if multi_strategy:
        collector_config.search.use_multi_strategy = True
    if multi_pass:
        collector_config.search.multi_pass.enabled = True

    log_cfg = collector_config.logging
    level = (
        getattr(logging, log_level.upper())
        if log_level
        else getattr(logging, log_cfg.level.upper(), logging.INFO)
    )
    setup_logging(
        level=level,
        log_file=log_cfg.file_path,
        max_bytes=log_cfg.max_file_size_mb * 1024 * 1024,
        backup_count=log_cfg.backup_count,
        console_output=log_cfg.console_output,
    )

    collector = KaraokeCollector(collector_config)

    try:
        print(f"Starting channel collection from: {channel_url}")
        if max_videos:
            print(f"Maximum videos to process: {max_videos}")

        incremental = not no_incremental
        if incremental:
            print("Using incremental mode (only new videos)")
        else:
            print("Processing all videos in channel")

        asyncio.run(_collect_channel_async(collector, channel_url, max_videos, incremental))

    except KeyboardInterrupt:
        logging.info("Channel collection interrupted by user")
    except Exception as e:
        logging.error(f"Channel collection failed: {e}")
        sys.exit(1)


@cli.command()
@click.argument("channels_file", type=click.Path(exists=True))
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to configuration file")
@click.option("--max-videos", "-m", type=int, help="Maximum videos per channel")
@click.option("--log-level", default="INFO", help="Logging level")
def collect_channels(channels_file, config, max_videos, log_level):
    """Collect karaoke videos from multiple channels (one URL per line in file)."""

    if config:
        collector_config = load_config(config)
    else:
        collector_config = CollectorConfig()

    log_cfg = collector_config.logging
    level = (
        getattr(logging, log_level.upper())
        if log_level
        else getattr(logging, log_cfg.level.upper(), logging.INFO)
    )
    setup_logging(
        level=level,
        log_file=log_cfg.file_path,
        max_bytes=log_cfg.max_file_size_mb * 1024 * 1024,
        backup_count=log_cfg.backup_count,
        console_output=log_cfg.console_output,
    )

    # Read channel URLs from file
    try:
        with open(channels_file, "r") as f:
            channel_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except Exception as e:
        print(f"Error reading channels file: {e}")
        sys.exit(1)

    if not channel_urls:
        print("No channel URLs found in file")
        sys.exit(1)

    collector = KaraokeCollector(collector_config)

    try:
        print(f"Starting collection from {len(channel_urls)} channels")
        if max_videos:
            print(f"Maximum videos per channel: {max_videos}")

        asyncio.run(_collect_channels_async(collector, channel_urls, max_videos))

    except KeyboardInterrupt:
        logging.info("Multi-channel collection interrupted by user")
    except Exception as e:
        logging.error(f"Multi-channel collection failed: {e}")
        sys.exit(1)


@cli.command()
@click.option("--output", "-o", default="config_template.yaml", help="Output path for template")
def create_config(output):
    """Create a configuration file template."""
    save_config_template(output)


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to configuration file")
def stats(config):
    """Show database statistics."""
    if config:
        collector_config = load_config(config)
    else:
        collector_config = CollectorConfig()

    collector = KaraokeCollector(collector_config)
    stats = asyncio.run(collector.get_statistics())

    print("\n" + "=" * 50)
    print("DATABASE STATISTICS")
    print("=" * 50)
    print(f"Total videos: {stats.get('total_videos', 0):,}")
    print(f"Videos with artist info: {stats.get('videos_with_artist', 0):,}")
    print(f"Average confidence score: {stats.get('avg_confidence', 0):.2f}")
    print(f"Average quality score: {stats.get('avg_quality', 0):.2f}")

    if top_artists := stats.get("top_artists", []):
        print("\nTop 10 Artists:")
        for artist, count, avg_views in top_artists:
            print(f"  {artist}: {count} videos (avg {avg_views:,.0f} views)")


@cli.command()
def check_env():
    """Check environment variables for external services."""
    import os

    print("\n" + "=" * 50)
    print("ENVIRONMENT VARIABLES")
    print("=" * 50)

    # Check Discogs token
    discogs_token = os.getenv("DISCOGS_TOKEN") or os.getenv("DISCOGS-TOKEN")
    if discogs_token:
        print(f"✓ DISCOGS_TOKEN: Set (length: {len(discogs_token)} characters)")
        print(f"  First 8 chars: {discogs_token[:8]}...")
    else:
        print("✗ DISCOGS_TOKEN: Not set")
        print("  Set DISCOGS_TOKEN environment variable to enable Discogs search")
        print("  Get token from: https://www.discogs.com/settings/developers")

    # Check other relevant environment variables
    env_vars = [
        ("DISCOGS_REQUESTS_PER_MINUTE", "Discogs rate limit"),
        ("DISCOGS_MAX_RESULTS", "Max Discogs results per search"),
        ("DISCOGS_ENABLED", "Enable/disable Discogs integration"),
    ]

    print("\nOptional environment variables:")
    for var_name, description in env_vars:
        value = os.getenv(var_name)
        if value:
            print(f"  {var_name}: {value}")
        else:
            print(f"  {var_name}: Not set (using default)")

    # Check if .env file exists
    from pathlib import Path

    env_file = Path(".env")
    if env_file.exists():
        print(f"\n✓ .env file found: {env_file.absolute()}")
    else:
        print(f"\n✗ .env file not found in current directory: {Path.cwd()}")
        print("  Create a .env file with your environment variables")

    print("\nEnvironment check complete!")


if __name__ == "__main__":
    cli()
