"""CLI commands for Discogs integration monitoring and management."""

import json
import logging
from pathlib import Path

import click

from .config import load_config
from .discogs_monitor import DiscogsMonitor

logger = logging.getLogger(__name__)


@click.group(name="discogs")
def discogs_cli():
    """Discogs integration management commands."""
    pass


@discogs_cli.command()
@click.option("--config", "-c", help="Configuration file path")
@click.option("--format", "-f", type=click.Choice(["table", "json"]), default="table",
              help="Output format")
def status(config, format):
    """Show Discogs integration status and metrics."""
    try:
        # Load configuration
        cfg = load_config(config)
        
        # Check if Discogs is enabled
        if not cfg.data_sources.discogs_enabled:
            click.echo("❌ Discogs integration is disabled in configuration")
            return
        
        # Create a monitor instance to get current status
        monitor = DiscogsMonitor(cfg)
        
        # Get metrics and health status
        metrics = monitor.get_current_metrics()
        health = monitor.get_health_status()
        
        if format == "json":
            output = {
                "discogs_status": {
                    "enabled": cfg.data_sources.discogs_enabled,
                    "token_configured": bool(cfg.data_sources.discogs_token),
                    "health": health,
                    "metrics": metrics
                }
            }
            click.echo(json.dumps(output, indent=2))
        else:
            # Table format
            _display_status_table(cfg, health, metrics)
            
    except Exception as e:
        click.echo(f"❌ Error checking Discogs status: {e}", err=True)


@discogs_cli.command()
@click.option("--config", "-c", help="Configuration file path")
@click.option("--output", "-o", help="Output file path", 
              default="discogs_metrics.json")
def export_metrics(config, output):
    """Export Discogs metrics to JSON file."""
    try:
        cfg = load_config(config)
        monitor = DiscogsMonitor(cfg)
        
        monitor.export_metrics(output)
        click.echo(f"✅ Metrics exported to {output}")
        
    except Exception as e:
        click.echo(f"❌ Error exporting metrics: {e}", err=True)


@discogs_cli.command()
@click.option("--config", "-c", help="Configuration file path")
def test_connection(config):
    """Test connection to Discogs API."""
    try:
        cfg = load_config(config)
        
        if not cfg.data_sources.discogs_enabled:
            click.echo("❌ Discogs integration is disabled")
            return
        
        if not cfg.data_sources.discogs_token:
            click.echo("❌ No Discogs token configured")
            click.echo("Set DISCOGS_TOKEN environment variable or configure in settings")
            return
        
        # Test basic connectivity
        import asyncio
        import os
        from .passes.discogs_search_pass import DiscogsClient
        from .utils import DiscogsRateLimiter
        
        # Get token from env or config
        token = os.getenv("DISCOGS_TOKEN") or os.getenv("DISCOGS-TOKEN") or cfg.data_sources.discogs_token
        
        if not token:
            click.echo("❌ No token available for testing")
            return
        
        rate_limiter = DiscogsRateLimiter(cfg.data_sources.discogs_requests_per_minute)
        client = DiscogsClient(token, rate_limiter, cfg.data_sources.discogs_user_agent)
        
        async def test_api():
            click.echo("🔄 Testing Discogs API connection...")
            results = await client.search_release("Adele", "Hello", max_results=1)
            return len(results) > 0
        
        success = asyncio.run(test_api())
        
        if success:
            click.echo("✅ Discogs API connection successful")
        else:
            click.echo("⚠️  Discogs API connection test returned no results")
            click.echo("   This might be normal depending on search parameters")
            
    except Exception as e:
        click.echo(f"❌ Error testing connection: {e}", err=True)


@discogs_cli.command()
@click.option("--config", "-c", help="Configuration file path")
def health_check(config):
    """Perform comprehensive health check of Discogs integration."""
    try:
        cfg = load_config(config)
        monitor = DiscogsMonitor(cfg)
        health = monitor.get_health_status()
        
        click.echo("🏥 Discogs Integration Health Check")
        click.echo("=" * 40)
        
        # Overall status
        status_emoji = {
            "healthy": "✅",
            "degraded": "⚠️ ",
            "unhealthy": "❌"
        }
        
        click.echo(f"Overall Status: {status_emoji.get(health['status'], '❓')} {health['status'].upper()}")
        
        # Alerts
        if health["alerts"]:
            click.echo("\n🚨 ALERTS:")
            for alert in health["alerts"]:
                click.echo(f"  • {alert}")
        
        # Warnings
        if health["warnings"]:
            click.echo("\n⚠️  WARNINGS:")
            for warning in health["warnings"]:
                click.echo(f"  • {warning}")
        
        # Recommendations
        if health["recommendations"]:
            click.echo("\n💡 RECOMMENDATIONS:")
            for rec in health["recommendations"]:
                click.echo(f"  • {rec}")
        
        if not any([health["alerts"], health["warnings"], health["recommendations"]]):
            click.echo("\n✅ No issues detected")
            
    except Exception as e:
        click.echo(f"❌ Error performing health check: {e}", err=True)


def _display_status_table(config, health, metrics):
    """Display status in table format."""
    click.echo("🎵 Discogs Integration Status")
    click.echo("=" * 50)
    
    # Configuration
    click.echo("📋 Configuration:")
    click.echo(f"  Enabled: {'✅ Yes' if config.data_sources.discogs_enabled else '❌ No'}")
    click.echo(f"  Token: {'✅ Configured' if config.data_sources.discogs_token else '❌ Missing'}")
    click.echo(f"  Rate Limit: {config.data_sources.discogs_requests_per_minute} req/min")
    click.echo(f"  Timeout: {config.data_sources.discogs_timeout}s")
    click.echo(f"  Use as Fallback: {'✅ Yes' if config.data_sources.discogs_use_as_fallback else '❌ No'}")
    
    # Health Status
    status_emoji = {"healthy": "✅", "degraded": "⚠️ ", "unhealthy": "❌"}
    click.echo(f"\n🏥 Health: {status_emoji.get(health['status'], '❓')} {health['status'].upper()}")
    
    # API Metrics
    api_metrics = metrics["api_metrics"]
    click.echo("\n📊 API Metrics:")
    click.echo(f"  Total Calls: {api_metrics['total_calls']}")
    click.echo(f"  Success Rate: {api_metrics['success_rate']:.1%}")
    click.echo(f"  Avg Response Time: {api_metrics['avg_response_time_ms']:.0f}ms")
    click.echo(f"  Timeouts: {api_metrics['timeouts']}")
    click.echo(f"  Rate Limit Hits: {api_metrics['rate_limit_hits']}")
    
    # Search Metrics
    search_metrics = metrics["search_metrics"]
    click.echo("\n🔍 Search Metrics:")
    click.echo(f"  Total Searches: {search_metrics['total_searches']}")
    click.echo(f"  Match Rate: {search_metrics['match_rate']:.1%}")
    click.echo(f"  High Confidence Rate: {search_metrics['high_confidence_rate']:.1%}")
    click.echo(f"  Avg Confidence: {search_metrics['avg_confidence']:.3f}")
    click.echo(f"  Fallback Activations: {search_metrics['fallback_activations']}")
    
    # Data Quality
    quality_metrics = metrics["data_quality_metrics"]
    click.echo("\n📈 Data Quality:")
    click.echo(f"  Year Coverage: {quality_metrics['year_coverage']:.1%}")
    click.echo(f"  Genre Coverage: {quality_metrics['genre_coverage']:.1%}")
    click.echo(f"  Label Coverage: {quality_metrics['label_coverage']:.1%}")
    
    # Database
    db_metrics = metrics["database_metrics"]
    click.echo(f"\n💾 Database:")
    click.echo(f"  Records Saved: {db_metrics['records_saved']}")
    click.echo(f"  Save Success Rate: {db_metrics['save_success_rate']:.1%}")


if __name__ == "__main__":
    discogs_cli()