"""Video processing for comprehensive metadata extraction."""

import asyncio
import logging
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, cast
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

    class AsyncClientWithForceClose(httpx.AsyncClient):  # type: ignore
        async def force_close(self) -> None:  # pragma: no cover - httpx missing
            pass

else:

    class AsyncClientWithForceClose(httpx.AsyncClient):  # type: ignore
        async def force_close(self) -> None:  # pragma: no cover - depends on httpx internals
            transport = getattr(self, "_transport", None)
            if transport and hasattr(transport, "aclose"):
                await transport.aclose()


try:
    from tenacity import retry, stop_after_attempt, wait_exponential  # type: ignore
except ImportError:  # pragma: no cover - optional dependency

    class _DummyWait:
        def __add__(self, other):
            return self

    def retry(*dargs, **dkwargs):
        def _decorator(fn):
            return fn

        return _decorator

    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_exponential(*args, **kwargs):
        return _DummyWait()


from typing import TYPE_CHECKING

from .advanced_parser import AdvancedTitleParser, ParseResult
from .config import CollectorConfig
from .validation_corrector import ValidationCorrector

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .multi_pass_controller import MultiPassParsingController

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

    def __init__(
        self,
        config: CollectorConfig,
        multi_pass_controller: Optional["MultiPassParsingController"] = None,
    ):
        self.config = config
        self.yt_dlp_opts = self._setup_yt_dlp()

        # Enhanced HTTP client with proper timeout and resource management
        self.http_client: Optional[AsyncClientWithForceClose] = AsyncClientWithForceClose(
            timeout=httpx.Timeout(
                connect=10.0,  # Connection timeout
                read=config.scraping.timeout_seconds,  # Read timeout
                write=10.0,  # Write timeout
                pool=5.0,  # Pool timeout
            ),
            limits=httpx.Limits(
                max_connections=50,  # Increased for concurrent operations
                max_keepalive_connections=20,  # Increased keepalive pool
                keepalive_expiry=60.0,  # Longer keepalive for efficiency
            ),
        )
        self._cleanup_completed = False

        # Rate limiting for MusicBrainz (1 request per second)
        self._musicbrainz_last_request = 0
        self._musicbrainz_min_interval = 1.0  # MusicBrainz requires 1 req/sec limit

        # Initialize advanced parser if enabled
        if config.search.use_advanced_parser or config.search.multi_pass.enabled:
            self.advanced_parser = AdvancedTitleParser(config)
        else:
            self.advanced_parser = None

        # Multi-pass parsing controller (optional)
        self.multi_pass_controller = multi_pass_controller

        # Validator for post-processing metadata checks
        self.validator = ValidationCorrector()

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

    def _safe_int(self, value):
        """Safely convert value to integer, handling None and invalid values."""
        try:
            return int(value) if value is not None else 0
        except (ValueError, TypeError):
            return 0

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def process_video(self, video_info: Dict) -> ProcessingResult:
        """Process a single video comprehensively."""
        start_time = time.time()
        errors = []
        warnings = []
        video_url = video_info.get("url")

        if not video_url:
            return ProcessingResult(
                {}, 0.0, time.time() - start_time, ["No video URL provided"], []
            )

        try:
            # Rotate UA for each video extraction to stay under radar
            self.yt_dlp_opts["user_agent"] = random.choice(self.config.scraping.user_agents)

            # Extract basic metadata
            video_data = await self._extract_basic_metadata(video_url)
            if not video_data:
                return ProcessingResult(
                    {}, 0.0, time.time() - start_time, ["Failed to extract basic metadata"], []
                )

            # Preserve channel_id from the initial scan
            if "channel_id" in video_info:
                video_data["channel_id"] = video_info["channel_id"]

            # Extract artist/song using multi-pass controller if enabled
            parse_result = None
            if self.multi_pass_controller and self.config.search.multi_pass.enabled:
                try:
                    mp_res = await self.multi_pass_controller.parse_video(
                        video_id=video_data.get("video_id", ""),
                        title=video_data.get("title", ""),
                        description=video_data.get("description", ""),
                        tags=" ".join(video_data.get("tags", [])),
                        channel_name=video_data.get("uploader", ""),
                        channel_id=video_data.get("channel_id", ""),
                    )
                    parse_result = mp_res.final_result
                    # Transfer metadata from parse result to video data
                    if parse_result and parse_result.metadata:
                        video_data["metadata"] = parse_result.metadata
                except Exception as e:
                    warnings.append(f"Multi-pass parsing failed: {e}")

            # Extract karaoke features first so external lookups have artist info
            features = self._extract_karaoke_features(video_data, parse_result)
            video_data["features"] = features

            # Promote key features to top-level video_data for database storage
            if features.get("original_artist"):
                video_data["artist"] = features["original_artist"]
            if features.get("song_title"):
                video_data["song_title"] = features["song_title"]
            if features.get("featured_artists"):
                video_data["featured_artists"] = features["featured_artists"]
            if features.get("artist_confidence"):
                video_data["parse_confidence"] = features["artist_confidence"]
            # Only use fallback release_year if we don't have one from metadata
            if (
                features.get("release_year")
                and not video_data.get("metadata", {}).get("release_year")
                and not video_data.get("metadata", {}).get("year")
            ):
                video_data["release_year"] = features["release_year"]

            # Promote release_year from metadata if available
            if video_data.get("metadata", {}).get("release_year"):
                video_data["release_year"] = video_data["metadata"]["release_year"]
            elif video_data.get("metadata", {}).get("year"):
                video_data["release_year"] = video_data["metadata"]["year"]

            # Promote genre from metadata if available
            if video_data.get("metadata", {}).get("genre"):
                video_data["genre"] = video_data["metadata"]["genre"]
            elif video_data.get("metadata", {}).get("genres"):
                genres = video_data["metadata"]["genres"]
                video_data["genre"] = genres[0] if isinstance(genres, list) and genres else genres

            # Promote MusicBrainz data from metadata if available
            metadata = video_data.get("metadata", {})
            if metadata.get("musicbrainz_recording_id"):
                video_data["musicbrainz_recording_id"] = metadata["musicbrainz_recording_id"]
            if metadata.get("musicbrainz_artist_id"):
                video_data["musicbrainz_artist_id"] = metadata["musicbrainz_artist_id"]
            if metadata.get("musicbrainz_score"):
                video_data["musicbrainz_score"] = metadata["musicbrainz_score"]
            if metadata.get("musicbrainz_confidence"):
                video_data["musicbrainz_confidence"] = metadata["musicbrainz_confidence"]

            # Promote Discogs data from metadata if available
            if metadata.get("discogs_release_id"):
                video_data["discogs_release_id"] = metadata["discogs_release_id"]
            if metadata.get("discogs_master_id"):
                video_data["discogs_master_id"] = metadata["discogs_master_id"]
            if metadata.get("discogs_confidence"):
                video_data["discogs_confidence"] = metadata["discogs_confidence"]
            if metadata.get("discogs_url"):
                video_data["discogs_url"] = metadata["discogs_url"]

            # Enhance with external data (uses extracted features)
            await self._enhance_with_external_data(video_data, errors, warnings)

            # Calculate quality scores
            quality_scores = self._calculate_quality_scores(video_data)
            video_data["quality_scores"] = quality_scores

            # Calculate engagement ratio
            engagement_ratio = self._calculate_engagement_ratio(video_data)
            if engagement_ratio is not None:
                video_data["engagement_ratio"] = engagement_ratio

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
                "title": info.get("title") or "",
                "description": info.get("description") or "",
                "duration_seconds": info.get("duration") or 0,
                "view_count": info.get("view_count") or 0,
                "like_count": info.get("like_count") or 0,
                "comment_count": info.get("comment_count") or 0,
                "upload_date": info.get("upload_date") or "",
                "uploader": info.get("uploader") or "",
                "uploader_id": info.get("uploader_id") or None,
                "thumbnail": info.get("thumbnail") or "",
                "tags": info.get("tags") or [],
                "formats": info.get("formats") or [],
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
            and features.get("artist")
            and features.get("song_title")
        ):
            music_data, recording = await self._get_music_metadata(
                features["artist"], features["song_title"]
            )
            video_data.update(music_data)
            if recording:
                validation, suggestion = self.validator.validate(
                    features["artist"], features["song_title"], recording
                )
                video_data["validation"] = asdict(validation)
                if suggestion:
                    video_data["correction_suggestion"] = asdict(suggestion)
                if self.advanced_parser:
                    self.advanced_parser.apply_validation_feedback(
                        features["artist"],
                        features["song_title"],
                        validation,
                    )
            else:
                # Fallback if MusicBrainz lookup fails
                warnings.append("MusicBrainz lookup failed, using parsed data only.")
                video_data["parse_confidence"] = features.get("artist_confidence", 0.0)

    async def _get_ryd_data(self, video_id: str) -> Dict:
        """Get Return YouTube Dislike data with confidence scoring."""
        try:
            if not self.http_client:
                return {"ryd_confidence": 0.0}
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

    async def _get_music_metadata(self, artist: str, song: str) -> Tuple[Dict, Optional[Dict]]:
        """Get comprehensive music metadata from MusicBrainz API with rate limiting."""
        try:
            if not self.http_client:
                return {"release_year_confidence": 0.0}, None

            # Enforce MusicBrainz rate limiting (1 request per second)
            current_time = time.time()
            time_since_last = current_time - self._musicbrainz_last_request
            if time_since_last < self._musicbrainz_min_interval:
                sleep_time = self._musicbrainz_min_interval - time_since_last
                await asyncio.sleep(sleep_time)

            self._musicbrainz_last_request = time.time()

            # Enhanced MusicBrainz search with comprehensive data inclusion
            raw_query = f"artist:{artist} AND recording:{song}"
            query = quote_plus(raw_query)
            # Include credits and relationship info for richer context
            url = (
                "https://musicbrainz.org/ws/2/recording/?query="
                f"{query}&limit=3&fmt=json&inc="
                "artist-credits+artist-rels+recording-rels+releases+tags+genres"
            )

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
                    # Process the best matching recording
                    best_recording = self._select_best_recording(recordings, artist, song)
                    if best_recording:
                        return (
                            self._extract_recording_metadata(best_recording, artist, song),
                            best_recording,
                        )

        except Exception as e:
            logger.warning(f"MusicBrainz lookup failed for {artist} - {song}: {e}")

        return {"parse_confidence": 0.0}, None

    def _select_best_recording(
        self, recordings: List[Dict], target_artist: str, target_song: str
    ) -> Optional[Dict]:
        """Select the best matching recording from MusicBrainz results."""
        if not recordings:
            return None

        best_recording = None
        best_score = -1

        for recording in recordings:
            score = 0

            # Score based on artist name similarity
            artist_credits = recording.get("artist-credit", [])
            if artist_credits:
                recording_artist = artist_credits[0].get("name", "").lower()
                if (
                    target_artist.lower() in recording_artist
                    or recording_artist in target_artist.lower()
                ):
                    score += 50
                elif self._fuzzy_match(target_artist.lower(), recording_artist):
                    score += 30

            # Score based on song title similarity
            recording_title = recording.get("title", "").lower()
            if target_song.lower() in recording_title or recording_title in target_song.lower():
                score += 50
            elif self._fuzzy_match(target_song.lower(), recording_title):
                score += 30

            # Prefer recordings with more releases (indicates popularity)
            releases = recording.get("releases", [])
            score += min(len(releases), 10)  # Cap at 10 bonus points

            # Prefer recordings with tags/genres
            if recording.get("tags") or recording.get("genres"):
                score += 5

            if score > best_score:
                best_score = score
                best_recording = recording

        return best_recording if best_score >= 50 else None

    def _fuzzy_match(self, str1: str, str2: str) -> bool:
        """Simple fuzzy string matching."""
        # Remove common words and punctuation
        stop_words = {"the", "a", "an", "and", "or", "but", "feat", "ft"}

        def clean_string(s):
            import re

            s = re.sub(r"[^\w\s]", "", s.lower())
            words = [w for w in s.split() if w not in stop_words]
            return " ".join(words)

        clean1 = clean_string(str1)
        clean2 = clean_string(str2)

        # Check if 70% of words match
        words1 = set(clean1.split())
        words2 = set(clean2.split())

        if not words1 or not words2:
            return False

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) >= 0.7

    def _extract_recording_metadata(
        self, recording: Dict, target_artist: str, target_song: str
    ) -> Dict:
        """Extract comprehensive metadata from a MusicBrainz recording."""
        metadata = {}

        # Extract MusicBrainz IDs
        metadata["musicbrainz_recording_id"] = recording.get("id")

        # Extract artist information
        artist_credits = recording.get("artist-credit", [])
        if artist_credits:
            metadata["musicbrainz_artist_id"] = artist_credits[0].get("artist", {}).get("id")

        # Extract release information and year
        releases = recording.get("releases", [])
        if releases:
            # Find earliest release date
            earliest_year = None
            record_labels = set()

            for release in releases:
                # Extract release date
                date = release.get("date")
                if date and len(date) >= 4:
                    try:
                        year = int(date[:4])
                        if earliest_year is None or year < earliest_year:
                            earliest_year = year
                    except ValueError:
                        pass

                # Extract record labels
                label_info = release.get("label-info", [])
                for label in label_info:
                    label_name = label.get("label", {}).get("name")
                    if label_name:
                        record_labels.add(label_name)

            if earliest_year:
                metadata["release_year"] = earliest_year

            if record_labels:
                metadata["record_label"] = ", ".join(sorted(record_labels)[:3])  # Limit to top 3

        # Extract recording length
        length_ms = recording.get("length")
        if length_ms:
            metadata["recording_length_ms"] = length_ms

        # Extract and classify genres/tags
        genre_info = self._extract_and_classify_genres(recording)
        metadata.update(genre_info)

        # Calculate confidence score
        metadata["parse_confidence"] = self._calculate_parse_confidence(
            recording, target_artist, target_song, metadata
        )

        return metadata

    def _extract_and_classify_genres(self, recording: Dict) -> Dict:
        """Extract and classify genres from MusicBrainz tags and genres."""
        all_tags = []

        # Extract genres (newer MusicBrainz field)
        genres = recording.get("genres", [])
        for genre in genres:
            tag_name = genre.get("name", "").lower()
            if tag_name:
                all_tags.append(tag_name)

        # Extract tags (older but more comprehensive field)
        tags = recording.get("tags", [])
        for tag in tags:
            tag_name = tag.get("name", "").lower()
            count = tag.get("count", 0)
            if tag_name and count > 0:  # Only include tags with positive count
                all_tags.append(tag_name)

        # Classify into primary genre
        primary_genre = self._classify_primary_genre(all_tags)

        # Store all tags as JSON string for future analysis
        import json

        tags_json = json.dumps(all_tags[:20])  # Limit to top 20 tags

        return {"musicbrainz_genre": primary_genre, "musicbrainz_tags": tags_json}

    def _classify_primary_genre(self, tags: List[str]) -> Optional[str]:
        """Classify tags into primary genre categories."""
        genre_mapping = {
            # Rock and derivatives
            "rock": [
                "rock",
                "hard rock",
                "soft rock",
                "classic rock",
                "alternative rock",
                "indie rock",
            ],
            "pop": ["pop", "pop rock", "dance-pop", "electropop", "teen pop", "synth-pop"],
            "jazz": ["jazz", "smooth jazz", "bebop", "swing", "fusion", "big band"],
            "classical": [
                "classical",
                "orchestral",
                "opera",
                "symphony",
                "chamber music",
                "baroque",
            ],
            "blues": [
                "blues",
                "electric blues",
                "chicago blues",
                "delta blues",
                "rhythm and blues",
            ],
            "country": ["country", "bluegrass", "americana", "folk country", "country rock"],
            "folk": ["folk", "folk rock", "traditional folk", "contemporary folk", "acoustic"],
            "electronic": ["electronic", "techno", "house", "ambient", "drum and bass", "dubstep"],
            "hip-hop": ["hip hop", "rap", "hip-hop", "gangsta rap", "conscious hip hop"],
            "r&b": ["r&b", "rhythm and blues", "contemporary r&b", "neo soul"],
            "soul": ["soul", "northern soul", "southern soul", "motown"],
            "reggae": ["reggae", "ska", "dub", "dancehall"],
            "metal": ["metal", "heavy metal", "death metal", "black metal", "thrash metal"],
            "punk": ["punk", "punk rock", "hardcore punk", "pop punk"],
            "funk": ["funk", "p-funk", "funk rock"],
            "gospel": ["gospel", "contemporary gospel", "traditional gospel"],
            "world": ["world music", "ethnic", "traditional", "regional"],
        }

        # Count matches for each genre category
        genre_scores = {}
        for tag in tags:
            for genre, keywords in genre_mapping.items():
                for keyword in keywords:
                    if keyword in tag:
                        genre_scores[genre] = genre_scores.get(genre, 0) + 1
                        break

        # Return the genre with highest score
        if genre_scores:
            return max(genre_scores.items(), key=lambda x: x[1])[0].title()

        return None

    def _calculate_parse_confidence(
        self, recording: Dict, target_artist: str, target_song: str, metadata: Dict
    ) -> float:
        """Calculate confidence score for MusicBrainz match."""
        confidence = 0.0

        # Base confidence for having a match
        confidence += 0.3

        # Artist name match quality
        artist_credits = recording.get("artist-credit", [])
        if artist_credits:
            recording_artist = artist_credits[0].get("name", "").lower()
            if target_artist.lower() == recording_artist:
                confidence += 0.3
            elif (
                target_artist.lower() in recording_artist
                or recording_artist in target_artist.lower()
            ):
                confidence += 0.2
            elif self._fuzzy_match(target_artist.lower(), recording_artist):
                confidence += 0.1

        # Song title match quality
        recording_title = recording.get("title", "").lower()
        if target_song.lower() == recording_title:
            confidence += 0.3
        elif target_song.lower() in recording_title or recording_title in target_song.lower():
            confidence += 0.2
        elif self._fuzzy_match(target_song.lower(), recording_title):
            confidence += 0.1

        # Bonus for rich metadata
        if metadata.get("release_year"):
            confidence += 0.05
        if metadata.get("musicbrainz_genre"):
            confidence += 0.05
        if metadata.get("record_label"):
            confidence += 0.05
        if recording.get("releases") and len(recording["releases"]) > 1:
            confidence += 0.05

        return min(confidence, 1.0)

    def _extract_karaoke_features(
        self, video_data: Dict, parse_result: Optional[ParseResult] = None
    ) -> Dict:
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

        # Extract release year from titles/descriptions (fallback when MusicBrainz unavailable)
        release_year = self._extract_release_year_fallback(
            video_data.get("title", ""), video_data.get("description", "")
        )
        if release_year:
            features["release_year"] = release_year

        # Extract artist and song info
        if parse_result is None and self.advanced_parser:
            uploader = video_data.get("uploader", "")
            parse_result = self.advanced_parser.parse_title(title, description, tags, uploader)

        if parse_result is not None:
            artist_info = {
                "original_artist": parse_result.artist,
                "song_title": parse_result.song_title,
                "artist_confidence": parse_result.confidence,
                "parsing_method": parse_result.method,
                "pattern_used": parse_result.pattern_used,
            }

            if parse_result.featured_artists:
                artist_info["featured_artists"] = parse_result.featured_artists
            if parse_result.alternative_results:
                artist_info["alternative_extractions"] = parse_result.alternative_results
        else:
            artist_info = self._extract_artist_song_info(title, description, tags)

        features.update(artist_info)

        # Extract featured artists
        featured_artists = self._extract_featured_artists(title, description, tags)
        if featured_artists:
            features["featured_artists"] = featured_artists

        # Detect genre
        genre = self._detect_genre(title, description, tags)
        if genre:
            features["genre"] = genre

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
                    if artist and 2 < len(artist) < 50 and self._is_valid_featured_artist(artist):
                        featured.append(artist.title())

        # Remove duplicates while preserving order
        unique_featured = []
        for artist in featured:
            if artist not in unique_featured:
                unique_featured.append(artist)

        return ", ".join(unique_featured) if unique_featured else None

    def _is_valid_featured_artist(self, artist: str) -> bool:
        """Validate that a featured artist name is legitimate and not metadata noise."""
        artist_lower = artist.lower().strip()

        # Filter out common metadata terms that shouldn't be featured artists
        invalid_terms = {
            "rights holders",
            "rights holder",
            "copyright",
            "all rights reserved",
            "record label",
            "music group",
            "publishing",
            "records",
            "entertainment",
            "distribution",
            "ltd",
            "inc",
            "corp",
            "llc",
            "gmbh",
            "karaoke version",
            "instrumental",
            "melody",
            "backing track",
            "original artist",
            "various artists",
            "soundtrack",
            "compilation",
            "unknown artist",
            "remix",
            "edit",
            "version",
            "remaster",
        }

        # Check if the artist name is just an invalid term
        if artist_lower in invalid_terms:
            return False

        # Check if the artist name contains invalid terms (more flexible)
        for term in invalid_terms:
            if term in artist_lower:
                return False

        # Additional validation: check for reasonable artist name patterns
        # Reject if it's all numbers or special characters
        if re.match(r"^[\d\s\-_\.]+$", artist):
            return False

        # Reject single letters or too short names
        if len(artist.strip()) < 2:
            return False

        return True

    def _extract_release_year_fallback(self, title: str, description: str) -> Optional[int]:
        """Extract release year from title and description using pattern matching."""
        combined_text = f"{title} {description}"
        current_year = datetime.now().year

        # Don't accept current year as it's likely the upload year
        max_acceptable_year = current_year - 1

        # First, remove false positive patterns
        # Remove playlist references like "2000's Karaoke Playlist"
        combined_text = re.sub(r"\b\d{4}'s\s+\w+\s+[Pp]laylist", "", combined_text)
        # Remove decade references like "1990's hits"
        combined_text = re.sub(r"\b\d{4}'s\b", "", combined_text)

        # Year patterns (prioritized by reliability)
        year_patterns = [
            r"\((\d{4})\)",  # (1985) - most reliable
            r"\[(\d{4})\]",  # [1985] - second most reliable
            r"from\s+(\d{4})",  # "from 1985"
            r"released\s+(\d{4})",  # "released 1985"
            r"(\d{4})\s+version",  # "1985 version"
            r"circa\s+(\d{4})",  # "circa 1985"
            # Removed standalone \b(\d{4})\b pattern as it's too prone to false positives
            # Removed (\d{4})s pattern as it was matching decade references
        ]

        found_years = []

        for pattern in year_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches:
                try:
                    year = int(match)
                    # Validate reasonable year range (1900 to last year)
                    # Reject current year as it's likely the upload year
                    if 1900 <= year <= max_acceptable_year:
                        found_years.append(year)
                    elif year >= current_year:
                        logger.debug(
                            f"Rejected year {year} as it's current/future year (current year is {current_year})"
                        )
                except ValueError:
                    continue

        # Return the earliest valid year found (most likely to be original release)
        return min(found_years) if found_years else None

    def _detect_genre(self, title: str, description: str, tags: str) -> Optional[str]:
        """Detect music genre based on title, description and tags."""
        combined_text = f"{title} {description} {tags}".lower()

        # Christmas/Holiday genre detection
        christmas_keywords = {
            # Traditional Christmas songs
            "silent night",
            "jingle bells",
            "white christmas",
            "let it snow",
            "winter wonderland",
            "silver bells",
            "deck the halls",
            "rudolph",
            "santa claus",
            "sleigh ride",
            "mary had a baby",
            "o holy night",
            "the first noel",
            "hark the herald",
            "joy to the world",
            "o come all ye faithful",
            "away in a manger",
            "what child is this",
            "auld lang syne",
            "feliz navidad",
            "mary's boy child",
            "little drummer boy",
            "blue christmas",
            "chestnuts roasting",
            "rockin around the christmas tree",
            "have yourself a merry little christmas",
            "i'll be home for christmas",
            "its beginning to look a lot like christmas",
            "walking in a winter wonderland",
            # General Christmas/Holiday terms
            "christmas",
            "holiday",
            "xmas",
            "winter",
            "santa",
            "sleigh",
            "carol",
            "noel",
            "nativity",
            "advent",
            "bethlehem",
            "star of bethlehem",
            "christmas tree",
            "mistletoe",
            "holly",
            "ivy",
            "wreath",
            "christmas morning",
            "christmas eve",
            "christmas day",
            "holiday special",
            "christmas traditional",
            "winter song",
            "holiday song",
            "christmas carol",
            "holiday classic",
        }

        # Check for Christmas/Holiday keywords
        for keyword in christmas_keywords:
            if keyword in combined_text:
                return "Christmas/Holiday"

        # Could extend this to detect other genres in the future
        # e.g., Rock, Pop, Jazz, Classical, etc.

        return None

    def _extract_artist_song_info(self, title: str, description: str, tags: str = "") -> Dict:
        """Extract artist and song information from title using comprehensive pattern matching."""

        # Comprehensive patterns for various karaoke naming schemes
        # Each pattern has: (regex, artist_group, title_group, confidence)
        karaoke_patterns = [
            # Quoted patterns - High confidence due to explicit formatting
            # Pattern: "Artist" - "Title" "(Karaoke Version)"
            (r'^"([^"]+)"\s*-\s*"([^"]+)"\s*\([^)]*[Kk]araoke[^)]*\)', 1, 2, 0.95),
            # Pattern: Karaoke "Title" - "Artist" "*"
            (r'^[Kk]araoke\s+"([^"]+)"\s*-\s*"([^"]+)"', 2, 1, 0.95),
            # Pattern: Karaoke - Song Title - Artist Name (common format)
            (r"^[Kk]araoke\s*-\s*([^-]+?)\s*-\s*(.+)$", 2, 1, 0.95),
            # Pattern: "Title" "(in the Style of "Artist")"
            (r'^"([^"]+)"\s*\([^)]*[Ss]tyle\s+of\s+"([^"]+)"[^)]*\)', 2, 1, 0.9),
            # Pattern: "Sing King Karaoke - "Title" "(in the Style of "Artist")"
            (r'^[^-]+-\s*"([^"]+)"\s*\([^)]*[Ss]tyle\s+of\s+"([^"]+)"[^)]*\)', 2, 1, 0.9),
            # Pattern: "Artist" - "Title" - "Karaoke Version from Zoom Karaoke"
            (r'^"([^"]+)"\s*-\s*"([^"]+)"\s*-\s*"[^"]*[Kk]araoke[^"]*"', 1, 2, 0.9),
            # Pattern: "Artist"-"Title" ... "(Karaoke Version)" (complex format)
            (r'^"([^"]+)"-"([^"]+)"\s*(?:"[^"]*")*\s*\([^)]*[Kk]araoke[^)]*\)', 1, 2, 0.85),
            # Pattern: "Movie" - "Title" "("Artist")" "(Karaoke Version)"
            (
                r'^"[^"]+"\s*-\s*"([^"]+)"\s*"?\(?"?([^")]+)"?\)?"\s*\([^)]*[Kk]araoke[^)]*\)',
                2,
                1,
                0.85,
            ),
            # Pattern: "Kids Karaoke" "Title" "Karaoke Version from Zoom Karaoke"
            (r'^"[^"]*[Kk]araoke[^"]*"\s*"([^"]+)"\s*"[^"]*[Kk]araoke[^"]*"', None, 1, 0.7),
            # Bracket patterns
            # Pattern: [짱가라오케/원키/노래방] Artist-Title [ZZang KARAOKE] (Korean style)
            (
                r"^\[[^\]]*[Kk]araoke[^\]]*\]\s*([^-]+)-([^[\]]+)\s*\[[^\]]*[Kk]araoke[^\]]*\]",
                1,
                2,
                0.9,
            ),
            # Standard patterns without quotes
            # Pattern: Artist - Title (Karaoke Version)
            (r"^([^-]+?)\s*-\s*([^(]+?)\s*\([^)]*[Kk]araoke[^)]*\)", 1, 2, 0.85),
            # Pattern: Title (in the Style of Artist)
            (
                r"^([^(]+?)\s*\([^)]*[Ss]tyle\s+of\s+([^)]+?)\)(?:[^(]*\([^)]*[Kk]araoke[^)]*\))?",
                2,
                1,
                0.8,
            ),
            # Pattern: Channel Name - Title (in the Style of Artist)
            (r"^[^-]+-\s*([^(]+?)\s*\([^)]*[Ss]tyle\s+of\s+([^)]+?)\)", 2, 1, 0.8),
            # Pattern: Title by Artist (Karaoke)
            (r"^([^(]+?)\s+by\s+([^(]+?)\s*\([^)]*[Kk]araoke[^)]*\)", 2, 1, 0.8),
            # Pattern: Artist - Title [Karaoke]
            (r"^([^-\[]+?)\s*-\s*([^\[]+?)\s*\[[^\]]*[Kk]araoke[^\]]*\]", 1, 2, 0.8),
            # Pattern: Artist - Title (no karaoke indicators)
            (r"^([^-–—(\[]+?)\s*[-–—]\s*([^(\[]+?)$", 1, 2, 0.8),
            # Pattern: Title - Artist (with various separators and optional karaoke indicator)
            (
                r"^([^-–—]+?)\s*[-–—]\s*([^(\[]+)(?:\s*[\(\[][^)\]]*[Kk]araoke[^)\]]*[\)\]])?",
                2,
                1,
                0.7,
            ),
            # Generic karaoke pattern with title first
            (r"^([^(]+?)\s*\([^)]*[Kk]araoke[^)]*\)", None, 1, 0.6),
        ]

        # Clean title for better matching
        clean_title = self._clean_title_for_parsing(title)

        # Try each pattern
        for pattern, artist_group, title_group, confidence in karaoke_patterns:
            match = re.search(pattern, clean_title, re.IGNORECASE | re.UNICODE)
            if match:
                result = {}

                if artist_group and artist_group <= len(match.groups()):
                    artist = self._clean_extracted_text(match.group(artist_group))
                    if self._is_valid_artist_name(artist):
                        result["artist"] = artist

                if title_group and title_group <= len(match.groups()):
                    song_title = self._clean_extracted_text(match.group(title_group))
                    if self._is_valid_song_title(song_title):
                        result["song_title"] = song_title

                if result:
                    result["artist_confidence"] = confidence
                    result["extraction_pattern"] = pattern
                    return result

        # Fallback to description and tags
        desc_match = re.search(
            r"(?:by|artist|performed by):?\s*([^\n]+)", description, re.IGNORECASE
        )
        if desc_match:
            artist = self._clean_extracted_text(desc_match.group(1))
            if self._is_valid_artist_name(artist):
                return {"artist": artist, "artist_confidence": 0.5}

        # Try custom patterns from config
        for custom_pattern in self.config.search.title_patterns:
            try:
                match = re.search(custom_pattern, clean_title, re.IGNORECASE | re.UNICODE)
                if match and len(match.groups()) >= 2:
                    artist = self._clean_extracted_text(match.group(1))
                    song_title = self._clean_extracted_text(match.group(2))
                    result = {}

                    if self._is_valid_artist_name(artist):
                        result["artist"] = artist
                    if self._is_valid_song_title(song_title):
                        result["song_title"] = song_title

                    if result:
                        result["artist_confidence"] = 0.6
                        result["extraction_pattern"] = "custom"
                        return result
            except re.error:
                logger.warning(f"Invalid regex pattern in config: {custom_pattern}")

        return {"artist_confidence": 0.0}

    def _clean_title_for_parsing(self, title: str) -> str:
        """Clean title to improve pattern matching."""
        # Remove common prefixes that interfere with parsing
        cleaned = title.strip()

        # Remove leading content before actual title
        prefixes_to_remove = [
            r"^\[[^\]]*\]\s*",  # Remove [brackets] at start
            r"^【[^】]*】\s*",  # Remove 【brackets】 at start
            r"^.*?[Kk]araoke[^:]*:\s*",  # Remove "Something Karaoke: "
            r"^.*?presents\s*:?\s*",  # Remove "Channel presents: "
        ]

        for prefix in prefixes_to_remove:
            cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE)

        return cleaned.strip()

    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted artist/title text."""
        if not text:
            return ""

        # Remove extra quotes and brackets
        cleaned = re.sub(r'^["\'`]+|["\'`]+$', "", text.strip())
        cleaned = re.sub(r"^\([^)]*\)|^\[[^\]]*\]", "", cleaned).strip()

        # Remove trailing noise
        noise_patterns = [
            r"\s*\([^)]*(?:[Kk]araoke|[Ii]nstrumental|[Mm]inus|[Mm][Rr])[^)]*\)$",
            r"\s*\[[^\]]*(?:[Kk]araoke|[Ii]nstrumental|[Mm]inus|[Mm][Rr])[^\]]*\]$",
            r"\s*-\s*[Kk]araoke.*$",
            r"\s*[Mm][Rr]$",
            r"\s*[Ii]nst\.?$",
            r"\s*\([^)]*[Kk]ey\)$",  # Remove (Key) variations
        ]

        for pattern in noise_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

        # Clean up whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    def _is_valid_artist_name(self, artist: str) -> bool:
        """Check if extracted text looks like a valid artist name."""
        if not artist or len(artist.strip()) < 1:
            return False

        # Filter out common non-artist terms (more restrictive list)
        invalid_terms = {
            "karaoke",
            "instrumental",
            "backing track",
            "minus one",
            "mr",
            "inst",
            "version",
            "sing along",
            "cover",
            "remix",
            "quality",
            "audio",
            "video",
        }

        artist_lower = artist.lower().strip()
        if artist_lower in invalid_terms:
            return False

        # Allow more diverse characters (for international artists)
        # Check that at least 20% of characters are word characters
        word_chars = len(re.findall(r"\w", artist))
        if word_chars < len(artist) * 0.2:
            return False

        return len(artist.strip()) <= 100  # Reasonable length limit

    def _is_valid_song_title(self, title: str) -> bool:
        """Check if extracted text looks like a valid song title."""
        if not title or len(title.strip()) < 2:
            return False

        # Similar validation as artist but more lenient
        invalid_terms = {"karaoke", "instrumental", "backing track", "minus one", "mr", "inst"}

        title_lower = title.lower().strip()
        if title_lower in invalid_terms:
            return False

        return len(title.strip()) <= 200  # Reasonable length limit

    def _calculate_quality_scores(self, video_data: Dict) -> Dict:
        """Calculate comprehensive quality scores."""
        scores = {}

        # Technical quality score
        technical_factors = []

        formats = video_data.get("formats", [])
        if formats:
            max_height = max((self._safe_int(f.get("height")) for f in formats), default=0)
            if max_height >= 1080:
                technical_factors.append(0.4)
            elif max_height >= 720:
                technical_factors.append(0.3)
            elif max_height >= 480:
                technical_factors.append(0.2)

        max_abr = max((self._safe_int(f.get("abr")) for f in formats), default=0)
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
        if video_data.get("features", {}).get("artist"):
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

    def _calculate_engagement_ratio(self, video_data: Dict) -> Optional[float]:
        """Calculate engagement ratio as percentage."""
        view_count = video_data.get("view_count", 0)
        like_count = video_data.get("like_count", 0)
        estimated_dislikes = video_data.get("estimated_dislikes", 0)

        if view_count <= 0 or like_count is None:
            return None

        # Use net engagement if we have dislike data
        if estimated_dislikes > 0:
            net_engagement = like_count - estimated_dislikes
            ratio = net_engagement / view_count
        else:
            ratio = like_count / view_count

        # Convert to percentage and round to 3 decimal places
        return round(ratio * 100, 3)

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

        # Validation result from MusicBrainz cross-check
        validation_score = video_data.get("validation", {}).get("validation_score", 0.0)
        confidence_factors.append(validation_score * 0.2)

        # Penalty for errors
        error_penalty = min(len(errors) * 0.1, 0.3)

        base = max(min(sum(confidence_factors) - error_penalty, 1.0), 0.0)
        return min(base * (0.7 + 0.3 * validation_score), 1.0)

    async def cleanup(self):
        """Comprehensive cleanup of all network resources."""
        if self._cleanup_completed:
            return

        try:
            # Close HTTP client with timeout and force close on failure
            if hasattr(self, "http_client") and self.http_client:
                try:
                    await asyncio.wait_for(self.http_client.aclose(), timeout=10.0)
                    logger.debug("HTTP client closed successfully")
                except asyncio.TimeoutError:
                    logger.warning("HTTP client graceful close timed out, forcing close")
                    # Force close if graceful close fails
                    try:
                        if hasattr(self.http_client, "force_close"):
                            await self.http_client.force_close()
                    except Exception as fe:
                        logger.error(f"Force close failed: {fe}")
                except Exception as e:
                    logger.error(f"HTTP client cleanup error: {e}")
                finally:
                    # Ensure we don't leak references
                    self.http_client = None
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
        finally:
            self._cleanup_completed = True

        # Force garbage collection to clean up any remaining resources
        import gc

        gc.collect()

    def __del__(self):
        """Ensure cleanup is called when object is destroyed."""
        if not self._cleanup_completed and hasattr(self, "http_client"):
            # Schedule cleanup in event loop if possible
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.cleanup())
            except RuntimeError:
                # No event loop available, force close synchronously
                if hasattr(self, "http_client") and self.http_client:
                    try:
                        import asyncio

                        asyncio.run(self.http_client.aclose())
                    except Exception:
                        pass  # Best effort cleanup

        import gc

        gc.collect()
