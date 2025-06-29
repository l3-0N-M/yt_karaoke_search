"""Tests for the Discogs rate limiter."""

import asyncio
import time

import pytest

from collector.utils import DiscogsRateLimiter


class TestDiscogsRateLimiter:
    """Test the DiscogsRateLimiter class."""

    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization with default values."""
        limiter = DiscogsRateLimiter()
        
        assert limiter.requests_per_minute == 60
        assert limiter.requests_per_second == 1.0
        assert limiter.tokens == 5.0  # Initial burst
        assert limiter.max_tokens == 5.0

    def test_rate_limiter_custom_rate(self):
        """Test rate limiter with custom requests per minute."""
        limiter = DiscogsRateLimiter(requests_per_minute=30)
        
        assert limiter.requests_per_minute == 30
        assert limiter.requests_per_second == 0.5

    @pytest.mark.asyncio
    async def test_burst_tokens_available(self):
        """Test that burst tokens are available initially."""
        limiter = DiscogsRateLimiter(requests_per_minute=60)
        
        # Should be able to make 5 requests immediately
        for i in range(5):
            start_time = time.time()
            await limiter.wait_for_request()
            elapsed = time.time() - start_time
            
            # Should not wait for burst requests
            assert elapsed < 0.1, f"Request {i+1} took too long: {elapsed}s"

    @pytest.mark.asyncio
    async def test_rate_limiting_after_burst(self):
        """Test that rate limiting kicks in after burst tokens are exhausted."""
        limiter = DiscogsRateLimiter(requests_per_minute=60)  # 1 request per second
        
        # Exhaust burst tokens
        for _ in range(5):
            await limiter.wait_for_request()
        
        # Next request should wait
        start_time = time.time()
        await limiter.wait_for_request()
        elapsed = time.time() - start_time
        
        # Should wait approximately 1 second (with some tolerance)
        assert 0.9 <= elapsed <= 1.5, f"Wait time was {elapsed}s, expected ~1s"

    @pytest.mark.asyncio
    async def test_token_refill_over_time(self):
        """Test that tokens are refilled over time."""
        limiter = DiscogsRateLimiter(requests_per_minute=120)  # 2 requests per second
        
        # Exhaust burst tokens
        for _ in range(5):
            await limiter.wait_for_request()
        
        # Wait for tokens to refill
        await asyncio.sleep(1.5)  # Should refill 3 tokens in 1.5 seconds
        
        # Should be able to make 3 more requests quickly
        for i in range(3):
            start_time = time.time()
            await limiter.wait_for_request()
            elapsed = time.time() - start_time
            
            assert elapsed < 0.1, f"Request {i+1} should not wait, took {elapsed}s"

    def test_get_remaining_tokens_initial(self):
        """Test getting remaining tokens initially."""
        limiter = DiscogsRateLimiter()
        
        remaining = limiter.get_remaining_tokens()
        assert remaining == 5.0

    def test_get_remaining_tokens_after_use(self):
        """Test getting remaining tokens after use."""
        limiter = DiscogsRateLimiter()
        
        # Simulate consuming tokens
        limiter.tokens = 3.0
        
        remaining = limiter.get_remaining_tokens()
        assert remaining == 3.0

    def test_get_remaining_tokens_with_refill(self):
        """Test getting remaining tokens with time-based refill."""
        limiter = DiscogsRateLimiter(requests_per_minute=60)  # 1 per second
        
        # Consume tokens and update last_update to simulate passage of time
        limiter.tokens = 0.0
        limiter.last_update = time.time() - 2.0  # 2 seconds ago
        
        remaining = limiter.get_remaining_tokens()
        
        # Should have refilled 2 tokens in 2 seconds, but capped at max_tokens
        assert remaining == 2.0

    def test_get_remaining_tokens_capped_at_max(self):
        """Test that remaining tokens are capped at max_tokens."""
        limiter = DiscogsRateLimiter(requests_per_minute=60)
        
        # Simulate long time passage
        limiter.tokens = 0.0
        limiter.last_update = time.time() - 100.0  # 100 seconds ago
        
        remaining = limiter.get_remaining_tokens()
        
        # Should be capped at max_tokens (5.0)
        assert remaining == 5.0

    def test_reset_rate_limiter(self):
        """Test resetting the rate limiter."""
        limiter = DiscogsRateLimiter()
        
        # Consume tokens
        limiter.tokens = 1.0
        original_time = limiter.last_update
        
        # Wait a bit and reset
        time.sleep(0.1)
        limiter.reset()
        
        assert limiter.tokens == 5.0  # Reset to max
        assert limiter.last_update > original_time  # Updated timestamp

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test that concurrent requests are properly rate limited."""
        limiter = DiscogsRateLimiter(requests_per_minute=120)  # 2 per second
        
        async def make_request(request_id):
            start_time = time.time()
            await limiter.wait_for_request()
            end_time = time.time()
            return request_id, end_time - start_time
        
        # Make 8 concurrent requests (3 more than burst)
        tasks = [make_request(i) for i in range(8)]
        results = await asyncio.gather(*tasks)
        
        # First 5 should be fast (burst), rest should be rate limited
        fast_requests = sum(1 for _, duration in results if duration < 0.1)
        slow_requests = sum(1 for _, duration in results if duration >= 0.1)
        
        assert fast_requests <= 5, "Too many fast requests"
        assert slow_requests >= 3, "Not enough rate-limited requests"

    @pytest.mark.asyncio
    async def test_very_low_rate_limit(self):
        """Test with very low rate limit."""
        limiter = DiscogsRateLimiter(requests_per_minute=6)  # 0.1 per second
        
        # Exhaust burst
        for _ in range(5):
            await limiter.wait_for_request()
        
        # Next request should wait 10 seconds
        start_time = time.time()
        await limiter.wait_for_request()
        elapsed = time.time() - start_time
        
        # Should wait approximately 10 seconds (with tolerance)
        assert 9.0 <= elapsed <= 11.0, f"Wait time was {elapsed}s, expected ~10s"

    @pytest.mark.asyncio
    async def test_fractional_token_consumption(self):
        """Test that fractional tokens work correctly."""
        limiter = DiscogsRateLimiter(requests_per_minute=90)  # 1.5 per second
        
        # Make several requests quickly to test fractional calculations
        start_time = time.time()
        
        for i in range(7):  # 5 burst + 2 rate limited
            await limiter.wait_for_request()
        
        total_time = time.time() - start_time
        
        # Should take approximately: 0 (burst) + 2/1.5 = 1.33 seconds
        assert 1.0 <= total_time <= 2.0, f"Total time was {total_time}s"

    def test_rate_limiter_thread_safety_setup(self):
        """Test that rate limiter has proper async lock setup."""
        limiter = DiscogsRateLimiter()
        
        # Check that asyncio.Lock is created
        assert limiter.lock is not None
        assert isinstance(limiter.lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_zero_tokens_edge_case(self):
        """Test edge case when tokens reach exactly zero."""
        limiter = DiscogsRateLimiter(requests_per_minute=60)
        
        # Manually set tokens to exactly zero
        limiter.tokens = 0.0
        
        start_time = time.time()
        await limiter.wait_for_request()
        elapsed = time.time() - start_time
        
        # Should wait for 1 full token (1 second at 60 rpm)
        assert 0.9 <= elapsed <= 1.5, f"Wait time was {elapsed}s, expected ~1s"

    def test_rate_limiter_properties(self):
        """Test rate limiter calculated properties."""
        # Test different rates
        test_cases = [
            (60, 1.0),    # 60 per minute = 1 per second
            (120, 2.0),   # 120 per minute = 2 per second
            (30, 0.5),    # 30 per minute = 0.5 per second
            (6, 0.1),     # 6 per minute = 0.1 per second
        ]
        
        for rpm, expected_rps in test_cases:
            limiter = DiscogsRateLimiter(requests_per_minute=rpm)
            assert limiter.requests_per_minute == rpm
            assert abs(limiter.requests_per_second - expected_rps) < 0.001