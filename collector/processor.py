"""Video processing for comprehensive metadata extraction."""

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, cast
from urllib.parse import quote_plus

try:
    import yt_dlp  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yt_dlp = None

try:  # Optional dependency for tests without network modules
    import httpx  # type: ignore
except ImportError:  # pragma: no cover - optional dependency

    class _DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def get(self, *args, **kwargs):  # pragma: no cover - network disabled
            raise RuntimeError("httpx not available")

        async def aclose(self):  # pragma: no cover - network disabled
            pass

    class _DummyLimits:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyTimeout:
        def __init__(self, *args, **kwargs):
            pass

    class httpx:
        AsyncClient = _DummyAsyncClient
        Limits = _DummyLimits
        Timeout = _DummyTimeout


try:
    from tenacity import retry, stop_after_attempt, wait_exponential  # type: ignore
except ImportError:  # pragma: no cover - optional dependency

    def retry(*dargs, **dkwargs):
        def _decorator(fn):
            return fn

        return _decorator

    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_exponential(*args, **kwargs):
        return None


from .config import CollectorConfig

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of video processing with detailed metadata."""

    video_data: Dict
    confidence_score: float
    processing_time: float
    errors: List[str]
    warnings: List[str]

    @property
    def is_success(self) -> bool:
        return bool(self.video_data and not self.errors)


class VideoProcessor:
    """Advanced video processor with confidence scoring."""

    def __init__(self, config: CollectorConfig):
        self.config = config
        self.yt_dlp_opts = self._setup_yt_dlp()

        # Enhanced HTTP client with proper timeout and resource management
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,  # Connection timeout
                read=config.scraping.timeout_seconds,  # Read timeout
                write=10.0,  # Write timeout
                pool=5.0,  # Pool timeout
            ),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=30.0,  # Close idle connections
            ),
        )
        self._cleanup_completed = False

    def _setup_yt_dlp(self) -> Dict:
        """Configure yt-dlp for comprehensive metadata extraction."""
        return {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "ignoreerrors": True,
            "socket_timeout": self.config.scraping.timeout_seconds,
            "user_agent": random.choice(self.config.scraping.user_agents),
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def process_video(self, video_url: str) -> ProcessingResult:
        """Process a single video comprehensively."""
        start_time = time.time()
        errors = []
        warnings = []

        try:
            # Rotate UA for each video extraction to stay under radar
            self.yt_dlp_opts["user_agent"] = random.choice(self.config.scraping.user_agents)

            # Extract basic metadata
            video_data = await self._extract_basic_metadata(video_url)
            if not video_data:
                return ProcessingResult(
                    {}, 0.0, time.time() - start_time, ["Failed to extract basic metadata"], []
                )

            # Extract karaoke features first so external lookups have artist info
            features = self._extract_karaoke_features(video_data)
            video_data["features"] = features

            # Enhance with external data (uses extracted features)
            await self._enhance_with_external_data(video_data, errors, warnings)

            # Calculate quality scores
            quality_scores = self._calculate_quality_scores(video_data)
            video_data["quality_scores"] = quality_scores

            # Calculate confidence
            confidence = self._calculate_confidence_score(video_data, errors, warnings)

            processing_time = time.time() - start_time

            return ProcessingResult(
                video_data=video_data,
                confidence_score=confidence,
                processing_time=processing_time,
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            errors.append(f"Processing failed: {str(e)}")
            logger.error(f"Error processing {video_url}: {e}")
            return ProcessingResult({}, 0.0, time.time() - start_time, errors, warnings)

    async def _extract_basic_metadata(self, video_url: str) -> Dict:
        """Extract metadata using yt-dlp."""
        try:
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, self._extract_with_ytdlp, video_url)

            if not info:
                return {}

            return {
                "video_id": info["id"],
                "url": video_url,
                "title": info.get("title", ""),
                "description": info.get("description", ""),
                "duration_seconds": info.get("duration", 0),
                "view_count": info.get("view_count", 0),
                "like_count": info.get("like_count", 0),
                "comment_count": info.get("comment_count", 0),
                "upload_date": info.get("upload_date", ""),
                "uploader": info.get("uploader", ""),
                "uploader_id": info.get("uploader_id", ""),
                "thumbnail": info.get("thumbnail", ""),
                "tags": info.get("tags", []),
                "formats": info.get("formats", []),
            }
        except Exception as e:
            logger.error(f"yt-dlp extraction failed for {video_url}: {e}")
            return {}

    def _extract_with_ytdlp(self, video_url: str) -> Dict:
        """Execute yt-dlp extraction in thread."""
        if yt_dlp is None:
            raise RuntimeError("yt-dlp not available")
        with yt_dlp.YoutubeDL(self.yt_dlp_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            return cast(Dict, info)

    async def _enhance_with_external_data(
        self, video_data: Dict, errors: List[str], warnings: List[str]
    ):
        """Enhance video data with external sources."""
        if self.config.data_sources.ryd_api_enabled:
            ryd_data = await self._get_ryd_data(video_data["video_id"])
            video_data.update(ryd_data)

        # Get music metadata if we have artist info
        features = video_data.get("features", {})
        if (
            self.config.data_sources.musicbrainz_enabled
            and features.get("original_artist")
            and features.get("song_title")
        ):
            music_data = await self._get_music_metadata(
                features["original_artist"], features["song_title"]
            )
            video_data.update(music_data)

    async def _get_ryd_data(self, video_id: str) -> Dict:
        """Get Return YouTube Dislike data with confidence scoring."""
        try:
            url = f"{self.config.data_sources.ryd_api_url}?videoId={video_id}"
            response = await self.http_client.get(url, timeout=self.config.data_sources.ryd_timeout)

            if response.status_code == 200:
                data = response.json()
                confidence = self._calculate_ryd_confidence(data)

                return {
                    "estimated_dislikes": data.get("dislikes", 0),
                    "ryd_likes": data.get("likes", 0),
                    "ryd_rating": data.get("rating", 0),
                    "ryd_confidence": confidence,
                    "ryd_deleted": data.get("deleted", False),
                }
        except Exception as e:
            logger.debug(f"RYD API failed for {video_id}: {e}")

        return {"ryd_confidence": 0.0}

    def _calculate_ryd_confidence(self, ryd_data: Dict) -> float:
        """Calculate confidence score for RYD data."""
        if not ryd_data:
            return 0.0

        confidence_factors = []

        if ryd_data.get("dislikes", 0) > 0:
            confidence_factors.append(0.3)

        likes = ryd_data.get("likes", 0)
        dislikes = ryd_data.get("dislikes", 0)
        if likes > 0 and dislikes > 0:
            ratio = likes / (likes + dislikes)
            if 0.1 <= ratio <= 0.99:
                confidence_factors.append(0.4)

        if ryd_data.get("rating", 0) > 0:
            confidence_factors.append(0.2)

        if not ryd_data.get("deleted", False):
            confidence_factors.append(0.1)

        return min(sum(confidence_factors), 1.0)

    async def _get_music_metadata(self, artist: str, song: str) -> Dict:
        """Get release year from MusicBrainz API."""
        try:
            # Simple MusicBrainz search
            raw_query = f"artist:{artist} AND recording:{song}"
            query = quote_plus(raw_query)
            url = f"https://musicbrainz.org/ws/2/recording/?query={query}&limit=1&fmt=json"

            headers = {
                "User-Agent": self.config.data_sources.musicbrainz_user_agent,
                "Accept": "application/json",
            }

            response = await self.http_client.get(
                url, headers=headers, timeout=self.config.data_sources.musicbrainz_timeout
            )

            if response.status_code == 200:
                data = response.json()
                recordings = data.get("recordings", [])

                if recordings:
                    recording = recordings[0]
                    releases = recording.get("releases", [])

                    if releases:
                        # Get earliest release date
                        release_date = releases[0].get("date")
                        if release_date and len(release_date) >= 4:
                            year = int(release_date[:4])
                            return {"estimated_release_year": year, "release_year_confidence": 0.8}
        except Exception as e:
            logger.debug(f"MusicBrainz lookup failed for {artist} - {song}: {e}")

        return {"release_year_confidence": 0.0}

    def _extract_karaoke_features(self, video_data: Dict) -> Dict:
        """Extract karaoke-specific features with confidence scoring."""
        title = video_data.get("title", "").lower()
        description = video_data.get("description", "").lower()
        tags = " ".join(video_data.get("tags", [])).lower()
        combined_text = f"{title} {description} {tags}"

        features = {}

        feature_patterns = {
            "has_guide_vocals": ["guide vocal", "guide melody", "demo vocal", "vocal guide"],
            "has_scrolling_lyrics": ["scrolling lyrics", "moving lyrics", "karaoke style"],
            "has_backing_vocals": ["backing vocals", "background vocals", "harmony"],
            "is_instrumental_only": ["instrumental only", "no vocals", "backing track only"],
            "is_piano_only": ["piano only", "piano version", "piano karaoke"],
            "is_acoustic": ["acoustic", "unplugged", "acoustic version"],
        }

        # Assign a confidence score based on how many patterns match
        for feat, patterns in feature_patterns.items():
            hits = sum(p in combined_text for p in patterns)
            features[feat] = bool(hits)
            features[f"{feat}_confidence"] = min(0.6 + 0.1 * (hits - 1), 1.0) if hits else 0.0

        # Extract artist and song info
        artist_info = self._extract_artist_song_info(title, description, tags)
        features.update(artist_info)

        # Extract featured artists
        featured_artists = self._extract_featured_artists(title, description, tags)
        if featured_artists:
            features["featured_artists"] = featured_artists

        return features

    def _extract_featured_artists(self, title: str, description: str, tags: str) -> Optional[str]:
        """Extract featured artists from title, description, or tags."""
        combined_text = f"{title} {description} {tags}".lower()

        # Common patterns for featured artists
        patterns = [
            r"feat\.?\s+([^(\[]+)",
            r"featuring\s+([^(\[]+)",
            r"ft\.?\s+([^(\[]+)",
            r"with\s+([^(\[]+)",
            r"&\s+([^(\[]+)",
        ]

        featured = []
        for pattern in patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches:
                # Clean up the match and handle multiple artists
                cleaned = re.sub(r"[^\w\s,&]", "", match.strip())
                parts = re.split(r"\s*&\s*|\s+and\s+|,\s*", cleaned)
                for part in parts:
                    artist = part.strip()
                    if artist and 2 < len(artist) < 50:
                        featured.append(artist.title())

        # Remove duplicates while preserving order
        unique_featured = []
        for artist in featured:
            if artist not in unique_featured:
                unique_featured.append(artist)

        return ", ".join(unique_featured) if unique_featured else None

    def _extract_artist_song_info(self, title: str, description: str, tags: str = "") -> Dict:
        """Extract artist and song information from title, with fallbacks."""

        default_patterns = [
            (r"^(.+?)\s*-\s*(.+?)\s*\((?:karaoke|instrumental)", "artist_first"),
            (r"^(.+?)\s+by\s+(.+?)\s*\((?:karaoke|instrumental)", "title_first"),
            (r"^(.+?)\s*\(.*karaoke.*\)\s*-\s*(.+)$", "title_first"),
            (r"^karaoke:?\s*(.+?)\s*-\s*(.+?)(?:\s|$)", "title_first"),
            (r"^(.+?)\s*-\s*(.+)$", "artist_first"),
        ]

        patterns = default_patterns + [(p, None) for p in self.config.search.title_patterns]

        for pattern, orientation in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                part1, part2 = match.group(1).strip(), match.group(2).strip()

                if orientation == "artist_first":
                    return {"song_title": part2, "original_artist": part1, "artist_confidence": 0.8}
                if orientation == "title_first":
                    return {"song_title": part1, "original_artist": part2, "artist_confidence": 0.8}

                if len(part2) < len(part1) and len(part2) < 50:
                    return {"song_title": part1, "original_artist": part2, "artist_confidence": 0.7}
                else:
                    return {"song_title": part2, "original_artist": part1, "artist_confidence": 0.7}

        desc_match = re.search(r"(?:by|artist):?\s*(.+?)(?:\n|$)", description, re.IGNORECASE)
        if desc_match:
            return {"original_artist": desc_match.group(1).strip(), "artist_confidence": 0.6}

        tag_match = re.search(r"(?:by|artist):?\s*(.+?)(?:\n|$)", tags, re.IGNORECASE)
        if tag_match:
            return {"original_artist": tag_match.group(1).strip(), "artist_confidence": 0.5}

        return {"artist_confidence": 0.0}

    def _calculate_quality_scores(self, video_data: Dict) -> Dict:
        """Calculate comprehensive quality scores."""
        scores = {}

        # Technical quality score
        technical_factors = []

        formats = video_data.get("formats", [])
        if formats:
            max_height = max((f.get("height", 0) for f in formats), default=0)
            if max_height >= 1080:
                technical_factors.append(0.4)
            elif max_height >= 720:
                technical_factors.append(0.3)
            elif max_height >= 480:
                technical_factors.append(0.2)

        max_abr = max((f.get("abr", 0) for f in formats), default=0)
        if max_abr >= 192:
            technical_factors.append(0.3)
        elif max_abr >= 128:
            technical_factors.append(0.2)

        duration = video_data.get("duration_seconds", 0)
        if 120 <= duration <= 360:
            technical_factors.append(0.3)
        elif 60 <= duration <= 480:
            technical_factors.append(0.2)

        scores["technical_score"] = min(sum(technical_factors), 1.0)

        # Engagement quality score
        views = video_data.get("view_count", 0)
        likes = video_data.get("like_count", 0)

        engagement_factors = []
        if views > 1000:
            engagement_factors.append(0.3)
        if views > 10000:
            engagement_factors.append(0.2)
        if likes > 50:
            engagement_factors.append(0.3)
        if views > 0 and likes / views > 0.01:
            engagement_factors.append(0.2)

        # Apply a penalty when the dislike ratio is high
        if video_data.get("ryd_confidence", 0) > self.config.data_sources.ryd_confidence_threshold:
            dl = video_data.get("estimated_dislikes", 0)
            lk = video_data.get("like_count", 0)
            if (lk + dl) > 0 and dl / (lk + dl) > 0.3:  # >30% dislikes
                engagement_factors.append(-0.2)

        scores["engagement_score"] = max(0.0, min(sum(engagement_factors), 1.0))

        # Metadata completeness score
        metadata_factors = []
        if video_data.get("title"):
            metadata_factors.append(0.2)
        if video_data.get("description"):
            metadata_factors.append(0.2)
        if video_data.get("tags"):
            metadata_factors.append(0.2)
        if video_data.get("features", {}).get("original_artist"):
            metadata_factors.append(0.2)
        if video_data.get("features", {}).get("song_title"):
            metadata_factors.append(0.2)

        scores["metadata_completeness"] = min(sum(metadata_factors), 1.0)

        # Overall score (weighted average)
        scores["overall_score"] = (
            scores["technical_score"] * 0.3
            + scores["engagement_score"] * 0.4
            + scores["metadata_completeness"] * 0.3
        )

        return scores

    def _calculate_confidence_score(
        self, video_data: Dict, errors: List[str], warnings: List[str]
    ) -> float:
        """Calculate overall confidence in the extracted data."""
        confidence_factors = []

        # Basic data availability
        if video_data.get("title"):
            confidence_factors.append(0.2)
        if video_data.get("view_count", 0) > 0:
            confidence_factors.append(0.2)
        if video_data.get("duration_seconds", 0) > 0:
            confidence_factors.append(0.2)

        # Incorporate the confidence values from individual karaoke features
        features = video_data.get("features", {})
        confidence_keys = [
            "has_guide_vocals_confidence",
            "has_scrolling_lyrics_confidence",
            "has_backing_vocals_confidence",
            "is_instrumental_only_confidence",
            "is_piano_only_confidence",
            "is_acoustic_confidence",
        ]
        valid_confidences = [
            features.get(key, 0) for key in confidence_keys if features.get(key, 0) > 0
        ]
        feature_confidence = (
            sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0
        )
        confidence_factors.append(feature_confidence * 0.3)

        # RYD data confidence
        ryd_confidence = video_data.get("ryd_confidence", 0)
        confidence_factors.append(ryd_confidence * 0.1)

        # Penalty for errors
        error_penalty = min(len(errors) * 0.1, 0.3)

        return max(min(sum(confidence_factors) - error_penalty, 1.0), 0.0)

    async def cleanup(self):
        """Comprehensive cleanup of all network resources."""
        if self._cleanup_completed:
            return

        try:
            # Close HTTP client with timeout
            if hasattr(self, "http_client") and self.http_client:
                await asyncio.wait_for(self.http_client.aclose(), timeout=5.0)
                logger.debug("HTTP client closed successfully")
        except asyncio.TimeoutError:
            logger.warning("HTTP client cleanup timed out")
        except Exception as e:
            logger.error(f"HTTP client cleanup error: {e}")
        finally:
            self._cleanup_completed = True

        # Force garbage collection to clean up any remaining resources
        import gc

        gc.collect()
