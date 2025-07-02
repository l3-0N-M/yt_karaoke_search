"""Unit tests for utils.py - focusing on DiscogsRateLimiter improvements."""

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.utils import DiscogsRateLimiter


class TestDiscogsRateLimiter:
    """Test cases for DiscogsRateLimiter with focus on rate limiting improvements."""

    @pytest.fixture
    def rate_limiter(self):
        """Create a DiscogsRateLimiter instance."""
        return DiscogsRateLimiter(requests_per_minute=60)

    def test_initialization_with_conservative_rate(self, rate_limiter):
        """Test that rate limiter uses conservative 80% of allowed rate."""
        # Should use 80% of 60 = 48 requests per minute
        assert rate_limiter.requests_per_minute == 48
        assert rate_limiter.requests_per_second == 48 / 60.0

        # Check reduced burst tokens
        assert rate_limiter.tokens == 3.0
        assert rate_limiter.max_tokens == 3.0

        # Check minimum interval
        assert rate_limiter.min_request_interval == 1.0

    @pytest.mark.asyncio
    async def test_wait_for_request_basic(self, rate_limiter):
        """Test basic rate limiting behavior."""
        start_time = time.time()

        # First request should be immediate (using burst token)
        await rate_limiter.wait_for_request()
        first_duration = time.time() - start_time
        assert first_duration < 0.1  # Should be nearly instant

        # Check token was consumed
        assert rate_limiter.tokens < 3.0

    @pytest.mark.asyncio
    async def test_minimum_interval_enforcement(self, rate_limiter):
        """Test that minimum 1-second interval is enforced between requests."""
        # Make first request
        await rate_limiter.wait_for_request()
        first_time = time.time()

        # Second request should wait for minimum interval
        await rate_limiter.wait_for_request()
        second_time = time.time()

        interval = second_time - first_time
        assert interval >= 0.9  # Allow small timing variance

    @pytest.mark.asyncio
    async def test_burst_tokens_limit(self, rate_limiter):
        """Test that burst tokens are limited to 3."""
        # Consume all burst tokens quickly
        for i in range(3):
            await rate_limiter.wait_for_request()
            if i < 2:  # Don't wait after last request
                await asyncio.sleep(0.1)  # Small delay between requests

        # Next request should require waiting
        start_time = time.time()
        await rate_limiter.wait_for_request()
        duration = time.time() - start_time

        # Should have to wait since burst tokens are exhausted
        assert duration >= 0.5  # Some wait time expected

    def test_handle_429_error_basic(self, rate_limiter):
        """Test handling of 429 errors with exponential backoff."""
        # First 429 error
        rate_limiter.handle_429_error()
        assert rate_limiter.consecutive_429_errors == 1
        assert rate_limiter.backoff_until > time.time()

        # Second 429 error - backoff should increase
        first_backoff = rate_limiter.backoff_until
        rate_limiter.handle_429_error()
        assert rate_limiter.consecutive_429_errors == 2
        assert rate_limiter.backoff_until > first_backoff

    def test_handle_429_with_retry_after(self, rate_limiter):
        """Test handling 429 with Retry-After header."""
        retry_after_seconds = 30
        current_time = time.time()

        rate_limiter.handle_429_error(retry_after_seconds)

        # Should respect the retry-after value
        assert rate_limiter.retry_after is not None
        assert rate_limiter.retry_after >= current_time + retry_after_seconds - 1
        assert rate_limiter.retry_after <= current_time + retry_after_seconds + 1

    def test_exponential_backoff_calculation(self, rate_limiter):
        """Test exponential backoff increases correctly."""
        backoff_times = []

        for i in range(5):
            # Reset state before each test
            rate_limiter.consecutive_429_errors = 0
            rate_limiter.backoff_until = 0

            # Simulate i+1 consecutive errors
            for _ in range(i + 1):
                rate_limiter.handle_429_error()

            # Calculate expected backoff (without jitter)
            # Implementation uses (consecutive_429_errors - 1) as exponent
            expected = min(
                rate_limiter.max_backoff,
                rate_limiter.min_backoff * (rate_limiter.backoff_multiplier**i),
            )

            # Get actual backoff time
            actual_backoff = rate_limiter.backoff_until - time.time()
            backoff_times.append(actual_backoff)

            # Should be close to expected (accounting for jitter which is ±10%)
            assert abs(actual_backoff - expected) < expected * 0.15

    def test_backoff_max_limit(self, rate_limiter):
        """Test that backoff doesn't exceed maximum."""
        # Simulate many consecutive 429 errors
        rate_limiter.consecutive_429_errors = 20
        rate_limiter.handle_429_error()

        backoff_duration = rate_limiter.backoff_until - time.time()

        # Should not exceed max_backoff (5 minutes = 300 seconds)
        # Allow some margin for jitter
        assert backoff_duration <= rate_limiter.max_backoff * 1.1

    def test_handle_success_resets_backoff(self, rate_limiter):
        """Test that successful request resets backoff state."""
        # Simulate some 429 errors
        rate_limiter.handle_429_error()
        rate_limiter.handle_429_error()

        assert rate_limiter.consecutive_429_errors == 2
        assert rate_limiter.backoff_until > 0

        # Successful request
        rate_limiter.handle_success()

        # Should reset backoff state
        assert rate_limiter.consecutive_429_errors == 0
        assert rate_limiter.backoff_until == 0

    @pytest.mark.asyncio
    async def test_wait_during_backoff_period(self, rate_limiter):
        """Test that requests wait during backoff period."""
        # Set backoff period
        rate_limiter.backoff_until = time.time() + 0.5

        start_time = time.time()
        await rate_limiter.wait_for_request()
        duration = time.time() - start_time

        # Should have waited for backoff period
        assert duration >= 0.45  # Allow small variance

    @pytest.mark.asyncio
    async def test_wait_with_retry_after(self, rate_limiter):
        """Test that requests respect Retry-After header."""
        # Set retry-after
        rate_limiter.retry_after = time.time() + 0.3

        start_time = time.time()
        await rate_limiter.wait_for_request()
        duration = time.time() - start_time

        # Should have waited for retry-after period
        assert duration >= 0.25  # Allow small variance
        # Retry-after should be cleared
        assert rate_limiter.retry_after is None

    def test_update_rate_limit_info(self, rate_limiter):
        """Test updating rate limit information from headers."""
        rate_limiter.update_rate_limit_info(remaining=10, reset_time=1234567890)

        assert rate_limiter.rate_limit_remaining == 10
        assert rate_limiter.rate_limit_reset == 1234567890

    def test_get_remaining_tokens(self, rate_limiter):
        """Test getting remaining tokens calculation."""
        initial_tokens = rate_limiter.get_remaining_tokens()
        assert initial_tokens == 3.0

        # Consume a token
        rate_limiter.tokens -= 1.0

        # Wait a bit
        time.sleep(0.1)

        # Should have regenerated some tokens
        current_tokens = rate_limiter.get_remaining_tokens()
        assert current_tokens > 2.0
        assert current_tokens < 3.0

    def test_reset_rate_limiter(self, rate_limiter):
        """Test resetting rate limiter to initial state."""
        # Modify state
        rate_limiter.tokens = 0
        rate_limiter.consecutive_429_errors = 5
        rate_limiter.backoff_until = time.time() + 100
        rate_limiter.retry_after = time.time() + 50

        # Reset
        rate_limiter.reset()

        # Should be back to initial state
        assert rate_limiter.tokens == rate_limiter.max_tokens
        assert rate_limiter.consecutive_429_errors == 0
        assert rate_limiter.backoff_until == 0
        assert rate_limiter.retry_after is None

    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, rate_limiter):
        """Test that rate limiter handles concurrent requests properly."""

        async def make_request(limiter, request_id):
            start_time = time.time()
            await limiter.wait_for_request()
            duration = time.time() - start_time
            return (request_id, duration)

        # Launch multiple concurrent requests
        tasks = [make_request(rate_limiter, i) for i in range(5)]

        results = await asyncio.gather(*tasks)

        # Check that requests were properly serialized
        # First 3 should use burst tokens, rest should wait
        fast_requests = sum(1 for _, duration in results if duration < 0.1)
        assert fast_requests <= 3  # Only burst tokens should be fast

        # Check minimum intervals were enforced
        completion_times = []
        for i, (_, duration) in enumerate(results):
            completion_time = duration
            if i > 0:
                # Account for previous completions
                completion_time += sum(d for _, d in results[:i])
            completion_times.append(completion_time)

        # Verify minimum intervals between completions
        for i in range(1, len(completion_times)):
            interval = completion_times[i] - completion_times[i - 1]
            # Should maintain minimum interval (with some tolerance)
            assert interval >= 0.8  # 80% of min_request_interval

    @pytest.mark.asyncio
    async def test_token_regeneration_rate(self, rate_limiter):
        """Test that tokens regenerate at the correct rate."""
        # Use all tokens
        rate_limiter.tokens = 0
        rate_limiter.last_update = time.time()

        # Wait for tokens to regenerate
        await asyncio.sleep(1.5)

        # Calculate expected tokens
        # 48 requests/minute = 0.8 requests/second
        expected_tokens = 1.5 * rate_limiter.requests_per_second
        actual_tokens = rate_limiter.get_remaining_tokens()

        # Should be close to expected (capped at max)
        assert actual_tokens == pytest.approx(
            min(expected_tokens, rate_limiter.max_tokens), rel=0.1
        )
