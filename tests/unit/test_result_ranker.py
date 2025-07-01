"""Unit tests for result_ranker.py."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.search.providers.base import SearchResult
from collector.search.result_ranker import RankingResult, RankingWeights, ResultRanker


class TestResultRanker:
    """Test cases for ResultRanker."""

    @pytest.fixture
    def ranker(self):
        """Create a ResultRanker instance."""
        config = {
            "ranking_weights": {
                "relevance": 0.4,
                "quality": 0.3,
                "popularity": 0.2,
                "metadata": 0.1,
            }
        }
        return ResultRanker(config)

    @pytest.fixture
    def sample_results(self):
        """Create sample search results."""
        return [
            SearchResult(
                video_id="test1",
                url="https://youtube.com/watch?v=test1",
                title="Artist Name - Song Title (Karaoke Version)",
                channel="Professional Karaoke Channel",
                channel_id="UCpro",
                provider="youtube",
                duration=240,
                view_count=100000,
                upload_date="20240101",
                description="High quality karaoke with lyrics",
            ),
            SearchResult(
                video_id="test2",
                url="https://youtube.com/watch?v=test2",
                title="song title by artist name karaoke",
                channel="Home Karaoke",
                channel_id="UChome",
                provider="youtube",
                duration=180,
                view_count=500,
                upload_date="20230615",
                description="My karaoke cover",
            ),
            SearchResult(
                video_id="test3",
                url="https://youtube.com/watch?v=test3",
                title="ARTIST NAME - SONG TITLE [4K KARAOKE]",
                channel="Sing King Karaoke",
                channel_id="UCsing",
                provider="youtube",
                duration=210,
                view_count=1000000,
                upload_date="20240301",
                description="Professional studio karaoke backing track",
            ),
        ]

    def test_ranking_weights_normalization(self):
        """Test that ranking weights normalize correctly."""
        weights = RankingWeights(relevance=2.0, quality=1.0, popularity=1.0, metadata=1.0)
        weights.normalize()

        total = weights.relevance + weights.quality + weights.popularity + weights.metadata
        assert abs(total - 1.0) < 0.001

    def test_rank_results_basic(self, ranker, sample_results):
        """Test basic result ranking."""
        ranked = ranker.rank_results(sample_results, "artist name song title")

        assert len(ranked) == 3
        assert all(isinstance(r, RankingResult) for r in ranked)
        # Results should be sorted by score
        assert ranked[0].final_score >= ranked[1].final_score
        assert ranked[1].final_score >= ranked[2].final_score

    def test_relevance_score_calculation(self, ranker):
        """Test relevance score calculation."""
        result = SearchResult(
            video_id="test",
            url="test",
            title="Exact Query Match Karaoke",
            channel="Test Channel",
            channel_id="UCtest",
            description="This contains the exact query match",
            provider="test",
        )

        score = ranker._calculate_relevance_score(result, "exact query match", {})

        assert score > 0.5  # Should have high relevance

    def test_relevance_score_no_match(self, ranker):
        """Test relevance score with no query match."""
        result = SearchResult(
            video_id="test",
            url="test",
            title="Completely Different Content",
            channel="Test Channel",
            channel_id="UCtest",
            description="Nothing related",
            provider="test",
        )

        score = ranker._calculate_relevance_score(result, "artist song karaoke", {})

        assert score < 0.3  # Should have low relevance

    def test_quality_score_calculation(self, ranker):
        """Test quality score calculation."""
        # High quality result
        hq_result = SearchResult(
            video_id="test",
            url="https://youtube.com/watch?v=test",
            title="Song Title (4K HD Karaoke with Lyrics)",
            channel="Sing King Karaoke",
            channel_id="UCsing",
            duration=180,
            provider="test",
        )

        hq_score = ranker._calculate_quality_score(hq_result)

        # Low quality result
        lq_result = SearchResult(
            video_id="test",
            url="test",
            title="song low quality broken",
            channel="my bedroom covers",
            channel_id="UCbed",
            duration=30,
            provider="test",
        )

        lq_score = ranker._calculate_quality_score(lq_result)

        assert hq_score > lq_score

    def test_popularity_score_calculation(self, ranker):
        """Test popularity score calculation."""
        # Popular video
        popular = SearchResult(
            video_id="test",
            title="Test",
            view_count=5000000,
            upload_date="20240101",
            url="test",
            provider="test",
            channel="Test Channel",
            channel_id="UCtest",
        )

        popular_score = ranker._calculate_popularity_score(popular)

        # Unpopular video
        unpopular = SearchResult(
            video_id="test",
            title="Test",
            view_count=100,
            upload_date="20200101",
            url="test",
            provider="test",
            channel="Test Channel",
            channel_id="UCtest",
        )

        unpopular_score = ranker._calculate_popularity_score(unpopular)

        assert popular_score > unpopular_score

    def test_metadata_score_calculation(self, ranker):
        """Test metadata completeness score."""
        # Complete metadata
        complete = SearchResult(
            video_id="test",
            title="Complete Title",
            channel="Test Channel",
            channel_id="UCcomplete",
            description="Detailed description of the karaoke video",
            duration=180,
            view_count=1000,
            upload_date="20240101",
            url="test",
            provider="test",
        )

        complete_score = ranker._calculate_metadata_score(complete)

        # Incomplete metadata
        incomplete = SearchResult(
            video_id="test",
            title="Title",
            url="test",
            provider="test",
            channel="Test Channel",
            channel_id="UCtest",
        )

        incomplete_score = ranker._calculate_metadata_score(incomplete)

        assert complete_score > incomplete_score

    def test_quality_indicators(self, ranker, sample_results):
        """Test quality indicator detection."""
        ranked = ranker.rank_results(sample_results, "test")

        # Professional channel should rank higher
        sing_king_idx = next(i for i, r in enumerate(ranked) if "Sing King" in r.result.channel)
        home_karaoke_idx = next(
            i for i, r in enumerate(ranked) if "Home Karaoke" in r.result.channel
        )

        assert sing_king_idx < home_karaoke_idx

    def test_negative_quality_indicators(self, ranker):
        """Test negative quality indicators."""
        bad_result = SearchResult(
            video_id="test",
            title="Low Quality Broken Karaoke Out of Sync",
            url="test",
            provider="test",
            channel="Test Channel",
            channel_id="UCtest",
        )

        score = ranker._calculate_quality_score(bad_result)

        assert score < 0.5  # Should be penalized

    def test_duration_quality_impact(self, ranker):
        """Test duration's impact on quality score."""
        # Good duration (3-4 minutes)
        good_duration = SearchResult(
            video_id="test",
            title="Test",
            duration=210,  # 3.5 minutes
            url="https://youtube.com/watch?v=test",
            provider="test",
            channel="Test Channel",
            channel_id="UCtest",
        )
        good_score = ranker._calculate_quality_score(good_duration)

        # Too short
        too_short = SearchResult(
            video_id="test",
            title="Test",
            duration=30,  # 30 seconds
            url="https://youtube.com/watch?v=test",
            provider="test",
            channel="Test Channel",
            channel_id="channel123",
        )
        short_score = ranker._calculate_quality_score(too_short)

        # Too long
        too_long = SearchResult(
            video_id="test",
            title="Test",
            duration=600,  # 10 minutes
            url="https://youtube.com/watch?v=test",
            provider="test",
            channel="Test Channel",
            channel_id="channel123",
        )
        long_score = ranker._calculate_quality_score(too_long)

        assert good_score > short_score
        assert good_score > long_score

    def test_contextual_adjustments(self, ranker):
        """Test contextual adjustments to scores."""
        result = SearchResult(
            video_id="test",
            title="Test",
            url="test",
            provider="youtube",
            search_method="exact_match",
            upload_date="20240315",
            channel="Test Channel",
            channel_id="UCtest",
        )

        base_score = 0.7

        # With exact match bonus
        adjusted = ranker._apply_contextual_adjustments(base_score, result, {})
        assert adjusted > base_score

        # With recency preference
        adjusted_recent = ranker._apply_contextual_adjustments(
            base_score, result, {"prefer_recent": True}
        )
        assert adjusted_recent > base_score

    def test_empty_results(self, ranker):
        """Test ranking empty results list."""
        ranked = ranker.rank_results([], "test query")
        assert ranked == []

    def test_single_result(self, ranker):
        """Test ranking single result."""
        result = SearchResult(
            video_id="test",
            title="Single Result",
            url="test",
            provider="test",
            channel="Test Channel",
            channel_id="UCtest",
        )

        ranked = ranker.rank_results([result], "single")

        assert len(ranked) == 1
        assert ranked[0].result == result

    def test_channel_reputation(self, ranker):
        """Test channel reputation scoring."""
        verified_channel = SearchResult(
            video_id="test",
            title="Test",
            channel="Karaoke Mugen Official",
            channel_id="UCverified",
            url="test",
            provider="test",
        )

        unknown_channel = SearchResult(
            video_id="test",
            title="Test",
            channel="Random User 123",
            channel_id="UCunknown",
            url="test",
            provider="test",
        )

        verified_score = ranker._calculate_quality_score(verified_channel)
        unknown_score = ranker._calculate_quality_score(unknown_channel)

        assert verified_score > unknown_score

    def test_statistics_tracking(self, ranker, sample_results):
        """Test that statistics are tracked."""
        initial_total = ranker.ranking_stats["total_ranked"]

        ranker.rank_results(sample_results, "test")

        assert ranker.ranking_stats["total_ranked"] == initial_total + 3
        assert len(ranker.ranking_stats["score_distribution"]) > 0

    def test_get_statistics(self, ranker):
        """Test statistics retrieval."""
        stats = ranker.get_statistics()

        assert "weights" in stats
        assert "statistics" in stats
        assert abs(stats["weights"]["relevance"] - 0.4) < 0.001

    def test_ranking_factors_details(self, ranker, sample_results):
        """Test detailed ranking factor information."""
        ranked = ranker.rank_results(sample_results, "artist song")

        for result in ranked:
            assert "query_match" in result.ranking_factors
            assert "quality_indicators" in result.ranking_factors
            assert "popularity_metrics" in result.ranking_factors
            assert "metadata_completeness" in result.ranking_factors

    def test_age_calculation(self, ranker):
        """Test video age calculation."""
        # Recent video
        recent_date = datetime.now() - timedelta(days=7)
        recent = recent_date.strftime("%Y%m%d")

        age_days = ranker._calculate_age_days(recent)
        assert age_days is not None
        assert 6 <= age_days <= 8  # Allow for date boundaries

    def test_duplicate_url_handling(self, ranker):
        """Test handling of duplicate URLs."""
        results = [
            SearchResult(
                video_id="test",
                url="https://youtube.com/watch?v=same",
                title="First Version",
                channel="Test Channel",
                channel_id="UCtest",
                provider="youtube",
                view_count=1000,
            ),
            SearchResult(
                video_id="test2",
                url="https://youtube.com/watch?v=same",
                title="Second Version",
                channel="Test Channel",
                channel_id="UCtest",
                provider="youtube",
                view_count=2000,
            ),
        ]

        ranked = ranker.rank_results(results, "test")

        # Both should be ranked despite same URL
        assert len(ranked) == 2
