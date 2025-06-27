"""Command-line interface for the karaoke collector."""

import asyncio
import logging
import sys
import click

from .config import CollectorConfig, load_config, save_config_template
from .main import KaraokeCollector
from .utils import setup_logging

@click.group()
@click.version_option("2.0.0")
def cli():
    """Karaoke Video Collector - Extract comprehensive karaoke video data from YouTube."""
    pass

@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to configuration file')
@click.option('--queries', '-q', multiple=True, help='Search queries to use')
@click.option('--max-per-query', '-m', default=50, help='Max videos per query')
@click.option('--output-db', '-o', help='Output database path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose logging')
@click.option('--dry-run', is_flag=True, help='Dry run mode (no database writes)')
def collect(config, queries, max_per_query, output_db, verbose, dry_run):
    """Collect karaoke videos from YouTube."""
    
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Load configuration
    if config:
        collector_config = load_config(config)
    else:
        collector_config = CollectorConfig()
    
    # Override config with CLI options
    if output_db:
        collector_config.database.path = output_db
    if dry_run:
        collector_config.dry_run = True
    
    # Default queries if none provided
    if not queries:
        queries = [
            "karaoke",
            "karaoke with lyrics", 
            "piano karaoke",
            "acoustic karaoke",
            "pop karaoke",
            "rock karaoke"
        ]
    
    # Run collection
    collector = KaraokeCollector(collector_config)
    
    try:
        total_processed = asyncio.run(collector.collect_videos(list(queries), max_per_query))
        
        # Show results
        stats = asyncio.run(collector.get_statistics())
        print("\n" + "="*50)
        print("COLLECTION RESULTS")
        print("="*50)
        print(f"Total videos processed: {total_processed}")
        print(f"Total videos in database: {stats.get('total_videos', 0):,}")
        print(f"Videos with artist info: {stats.get('videos_with_artist', 0):,}")
        print(f"Average confidence score: {stats.get('avg_confidence', 0):.2f}")
        
        if top_artists := stats.get('top_artists', []):
            print("\nTop 10 Artists:")
            for artist, count, avg_views in top_artists[:10]:
                print(f"  {artist}: {count} videos (avg {avg_views:,.0f} views)")
        
    except KeyboardInterrupt:
        logging.info("Collection interrupted by user")
    except Exception as e:
        logging.error(f"Collection failed: {e}")
        sys.exit(1)
    finally:
        asyncio.run(collector.cleanup())

@cli.command()
@click.option('--output', '-o', default='config_template.yaml', help='Output path for template')
def create_config(output):
    """Create a configuration file template."""
    save_config_template(output)

@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to configuration file')
def stats(config):
    """Show database statistics."""
    if config:
        collector_config = load_config(config)
    else:
        collector_config = CollectorConfig()
    
    collector = KaraokeCollector(collector_config)
    stats = asyncio.run(collector.get_statistics())
    
    print("\n" + "="*50)
    print("DATABASE STATISTICS")
    print("="*50)
    print(f"Total videos: {stats.get('total_videos', 0):,}")
    print(f"Videos with artist info: {stats.get('videos_with_artist', 0):,}")
    print(f"Average confidence score: {stats.get('avg_confidence', 0):.2f}")
    print(f"Average quality score: {stats.get('avg_quality', 0):.2f}")
    
    if top_artists := stats.get('top_artists', []):
        print("\nTop 10 Artists:")
        for artist, count, avg_views in top_artists:
            print(f"  {artist}: {count} videos (avg {avg_views:,.0f} views)")

if __name__ == '__main__':
    cli()