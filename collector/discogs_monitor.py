"""Monitoring and metrics collection for Discogs integration."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DiscogsMetrics:
    """Metrics tracking for Discogs integration."""

    # API Metrics
    total_api_calls: int = 0
    successful_api_calls: int = 0
    failed_api_calls: int = 0
    api_timeouts: int = 0
    rate_limit_hits: int = 0

    # Search Metrics
    total_searches: int = 0
    successful_matches: int = 0
    high_confidence_matches: int = 0  # >0.8 confidence
    fallback_activations: int = 0  # When MB confidence low

    # Data Quality Metrics
    records_with_year: int = 0
    records_with_genres: int = 0
    records_with_label: int = 0
    avg_confidence: float = 0.0

    # Performance Metrics
    avg_response_time_ms: float = 0.0
    total_response_time_ms: float = 0.0

    # Database Metrics
    records_saved: int = 0
    save_errors: int = 0

    # Rate Limiting Metrics
    tokens_used: int = 0
    wait_time_total_ms: float = 0.0

    # Timestamps
    last_reset: datetime = field(default_factory=datetime.now)
    last_api_call: Optional[datetime] = None


class DiscogsMonitor:
    """Monitor and track Discogs integration performance."""

    def __init__(self, config):
        self.config = config
        self.metrics = DiscogsMetrics()
        self.hourly_metrics: List[DiscogsMetrics] = []
        self.last_hourly_reset = datetime.now()

        # Alert thresholds
        self.error_rate_threshold = 0.15  # 15% error rate triggers alert
        self.response_time_threshold = 5000  # 5 second response time alert
        self.success_rate_threshold = 0.70  # 70% minimum success rate

        logger.info("Discogs monitoring initialized")

    def record_api_call(
        self,
        success: bool,
        response_time_ms: float,
        timeout: bool = False,
        rate_limited: bool = False,
    ):
        """Record an API call with its metrics."""
        self.metrics.total_api_calls += 1
        self.metrics.last_api_call = datetime.now()
        self.metrics.total_response_time_ms += response_time_ms

        if success:
            self.metrics.successful_api_calls += 1
        else:
            self.metrics.failed_api_calls += 1

        if timeout:
            self.metrics.api_timeouts += 1

        if rate_limited:
            self.metrics.rate_limit_hits += 1

        # Update average response time
        if self.metrics.total_api_calls > 0:
            self.metrics.avg_response_time_ms = (
                self.metrics.total_response_time_ms / self.metrics.total_api_calls
            )

        # Check for alerts
        self._check_api_alerts()

    def record_search_attempt(self, success: bool, confidence: float = 0.0, fallback: bool = False):
        """Record a search attempt with its results."""
        self.metrics.total_searches += 1

        if success:
            self.metrics.successful_matches += 1

            if confidence >= 0.8:
                self.metrics.high_confidence_matches += 1

            # Update average confidence
            current_avg = self.metrics.avg_confidence
            total_successful = self.metrics.successful_matches
            self.metrics.avg_confidence = (
                current_avg * (total_successful - 1) + confidence
            ) / total_successful

        if fallback:
            self.metrics.fallback_activations += 1

    def record_data_quality(
        self, has_year: bool = False, has_genres: bool = False, has_label: bool = False
    ):
        """Record data quality metrics."""
        if has_year:
            self.metrics.records_with_year += 1
        if has_genres:
            self.metrics.records_with_genres += 1
        if has_label:
            self.metrics.records_with_label += 1

    def record_database_save(self, success: bool):
        """Record database save operation."""
        if success:
            self.metrics.records_saved += 1
        else:
            self.metrics.save_errors += 1

    def record_rate_limiting(self, wait_time_ms: float, tokens_consumed: int = 1):
        """Record rate limiting metrics."""
        self.metrics.wait_time_total_ms += wait_time_ms
        self.metrics.tokens_used += tokens_consumed

    def get_current_metrics(self) -> Dict:
        """Get current metrics as dictionary."""
        return {
            "api_metrics": {
                "total_calls": self.metrics.total_api_calls,
                "successful_calls": self.metrics.successful_api_calls,
                "failed_calls": self.metrics.failed_api_calls,
                "success_rate": self._calculate_success_rate(),
                "error_rate": self._calculate_error_rate(),
                "avg_response_time_ms": round(self.metrics.avg_response_time_ms, 2),
                "timeouts": self.metrics.api_timeouts,
                "rate_limit_hits": self.metrics.rate_limit_hits,
            },
            "search_metrics": {
                "total_searches": self.metrics.total_searches,
                "successful_matches": self.metrics.successful_matches,
                "match_rate": self._calculate_match_rate(),
                "high_confidence_matches": self.metrics.high_confidence_matches,
                "high_confidence_rate": self._calculate_high_confidence_rate(),
                "avg_confidence": round(self.metrics.avg_confidence, 3),
                "fallback_activations": self.metrics.fallback_activations,
            },
            "data_quality_metrics": {
                "records_with_year": self.metrics.records_with_year,
                "records_with_genres": self.metrics.records_with_genres,
                "records_with_label": self.metrics.records_with_label,
                "year_coverage": self._calculate_year_coverage(),
                "genre_coverage": self._calculate_genre_coverage(),
                "label_coverage": self._calculate_label_coverage(),
            },
            "database_metrics": {
                "records_saved": self.metrics.records_saved,
                "save_errors": self.metrics.save_errors,
                "save_success_rate": self._calculate_save_success_rate(),
            },
            "rate_limiting_metrics": {
                "tokens_used": self.metrics.tokens_used,
                "total_wait_time_ms": round(self.metrics.wait_time_total_ms, 2),
                "avg_wait_time_ms": self._calculate_avg_wait_time(),
            },
            "timestamps": {
                "last_reset": self.metrics.last_reset.isoformat(),
                "last_api_call": (
                    self.metrics.last_api_call.isoformat() if self.metrics.last_api_call else None
                ),
                "uptime_hours": self._calculate_uptime_hours(),
            },
        }

    def get_health_status(self) -> Dict:
        """Get health status with alerts."""
        metrics = self.get_current_metrics()
        health = {"status": "healthy", "alerts": [], "warnings": [], "recommendations": []}

        # Check error rate
        error_rate = metrics["api_metrics"]["error_rate"]
        if error_rate > self.error_rate_threshold:
            health["status"] = "unhealthy"
            health["alerts"].append(
                f"High error rate: {error_rate:.1%} (threshold: {self.error_rate_threshold:.1%})"
            )

        # Check response time
        avg_response_time = metrics["api_metrics"]["avg_response_time_ms"]
        if avg_response_time > self.response_time_threshold:
            health["status"] = "degraded" if health["status"] == "healthy" else health["status"]
            health["warnings"].append(
                f"High response time: {avg_response_time:.0f}ms (threshold: {self.response_time_threshold}ms)"
            )

        # Check success rate
        success_rate = metrics["api_metrics"]["success_rate"]
        if success_rate < self.success_rate_threshold:
            health["status"] = "unhealthy"
            health["alerts"].append(
                f"Low success rate: {success_rate:.1%} (threshold: {self.success_rate_threshold:.1%})"
            )

        # Check rate limiting
        if self.metrics.rate_limit_hits > 0:
            health["warnings"].append(
                f"Rate limiting detected: {self.metrics.rate_limit_hits} hits"
            )

        # Performance recommendations
        match_rate = metrics["search_metrics"]["match_rate"]
        if match_rate < 0.5:
            health["recommendations"].append(
                f"Low match rate ({match_rate:.1%}). Consider adjusting confidence thresholds."
            )

        high_conf_rate = metrics["search_metrics"]["high_confidence_rate"]
        if high_conf_rate < 0.3:
            health["recommendations"].append(
                f"Low high-confidence rate ({high_conf_rate:.1%}). Review search patterns."
            )

        return health

    def reset_metrics(self):
        """Reset metrics for new monitoring period."""
        logger.info("Resetting Discogs metrics")

        # Save current metrics to hourly history
        self.hourly_metrics.append(self.metrics)

        # Keep only last 24 hours of hourly metrics
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.hourly_metrics = [m for m in self.hourly_metrics if m.last_reset > cutoff_time]

        # Reset current metrics
        self.metrics = DiscogsMetrics()
        self.last_hourly_reset = datetime.now()

    def log_metrics_summary(self):
        """Log a summary of current metrics."""
        metrics = self.get_current_metrics()

        logger.info("=== Discogs Integration Metrics Summary ===")
        logger.info(
            f"API Calls: {metrics['api_metrics']['total_calls']} "
            f"(Success: {metrics['api_metrics']['success_rate']:.1%})"
        )
        logger.info(
            f"Searches: {metrics['search_metrics']['total_searches']} "
            f"(Matches: {metrics['search_metrics']['match_rate']:.1%})"
        )
        logger.info(f"Avg Response Time: {metrics['api_metrics']['avg_response_time_ms']:.0f}ms")
        logger.info(f"Avg Confidence: {metrics['search_metrics']['avg_confidence']:.3f}")
        logger.info(f"Records Saved: {metrics['database_metrics']['records_saved']}")

        # Log health status
        health = self.get_health_status()
        logger.info(f"Health Status: {health['status'].upper()}")

        if health["alerts"]:
            logger.warning(f"Alerts: {'; '.join(health['alerts'])}")
        if health["warnings"]:
            logger.info(f"Warnings: {'; '.join(health['warnings'])}")

    def export_metrics(self, file_path: str):
        """Export metrics to JSON file."""
        export_data = {
            "current_metrics": self.get_current_metrics(),
            "health_status": self.get_health_status(),
            "hourly_history": [
                {
                    "timestamp": m.last_reset.isoformat(),
                    "api_calls": m.total_api_calls,
                    "success_rate": m.successful_api_calls / max(m.total_api_calls, 1),
                    "avg_response_time": m.avg_response_time_ms,
                    "matches": m.successful_matches,
                    "avg_confidence": m.avg_confidence,
                }
                for m in self.hourly_metrics
            ],
            "exported_at": datetime.now().isoformat(),
        }

        try:
            with open(file_path, "w") as f:
                json.dump(export_data, f, indent=2)
            logger.info(f"Metrics exported to {file_path}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

    # Helper methods for calculations
    def _calculate_success_rate(self) -> float:
        if self.metrics.total_api_calls == 0:
            return 0.0
        return self.metrics.successful_api_calls / self.metrics.total_api_calls

    def _calculate_error_rate(self) -> float:
        if self.metrics.total_api_calls == 0:
            return 0.0
        return self.metrics.failed_api_calls / self.metrics.total_api_calls

    def _calculate_match_rate(self) -> float:
        if self.metrics.total_searches == 0:
            return 0.0
        return self.metrics.successful_matches / self.metrics.total_searches

    def _calculate_high_confidence_rate(self) -> float:
        if self.metrics.successful_matches == 0:
            return 0.0
        return self.metrics.high_confidence_matches / self.metrics.successful_matches

    def _calculate_year_coverage(self) -> float:
        if self.metrics.records_saved == 0:
            return 0.0
        return self.metrics.records_with_year / self.metrics.records_saved

    def _calculate_genre_coverage(self) -> float:
        if self.metrics.records_saved == 0:
            return 0.0
        return self.metrics.records_with_genres / self.metrics.records_saved

    def _calculate_label_coverage(self) -> float:
        if self.metrics.records_saved == 0:
            return 0.0
        return self.metrics.records_with_label / self.metrics.records_saved

    def _calculate_save_success_rate(self) -> float:
        total_saves = self.metrics.records_saved + self.metrics.save_errors
        if total_saves == 0:
            return 0.0
        return self.metrics.records_saved / total_saves

    def _calculate_avg_wait_time(self) -> float:
        if self.metrics.tokens_used == 0:
            return 0.0
        return self.metrics.wait_time_total_ms / self.metrics.tokens_used

    def _calculate_uptime_hours(self) -> float:
        delta = datetime.now() - self.metrics.last_reset
        return delta.total_seconds() / 3600

    def _check_api_alerts(self):
        """Check for immediate API alerts."""
        # Only check if we have enough data
        if self.metrics.total_api_calls < 10:
            return

        error_rate = self._calculate_error_rate()
        if error_rate > self.error_rate_threshold:
            logger.warning(
                f"ALERT: High Discogs API error rate: {error_rate:.1%} "
                f"({self.metrics.failed_api_calls}/{self.metrics.total_api_calls})"
            )

        if self.metrics.avg_response_time_ms > self.response_time_threshold:
            logger.warning(
                f"ALERT: High Discogs API response time: {self.metrics.avg_response_time_ms:.0f}ms"
            )
