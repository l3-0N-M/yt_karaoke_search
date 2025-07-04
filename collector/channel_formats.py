"""Channel-specific format configurations for karaoke title parsing.

This module contains mappings of known karaoke channels to their consistent
title formats, enabling more accurate artist/song extraction.
"""

from enum import Enum
from typing import Dict, Optional


class TitleFormat(Enum):
    """Enumeration of known title formats."""

    ARTIST_SONG = "artist-song"  # Format: "Artist - Song (Karaoke)"
    SONG_ARTIST = "song-artist"  # Format: "Song - Artist (Karaoke)"
    CUSTOM = "custom"  # Channel uses unique format
    UNKNOWN = "unknown"  # Format not yet determined


# Channel format mappings based on database analysis
# These channels have shown >80% consistency in their format
CHANNEL_FORMATS: Dict[str, TitleFormat] = {
    # Channels that consistently use Song - Artist format
    "KaraFun Karaoke": TitleFormat.SONG_ARTIST,  # Song - Artist | Karaoke by KaraFun
    "KaraokeyTV": TitleFormat.SONG_ARTIST,  # Song Artist (Karaoke) - no dash!
    "Musisi Karaoke": TitleFormat.SONG_ARTIST,  # Song - Artist (Karaoke with Lyrics)
    "ZZang KARAOKE": TitleFormat.SONG_ARTIST,  # Song (Karaoke) - Artist
    "Atomic Karaoke...": TitleFormat.SONG_ARTIST,  # Song - Artist | Karaoke
    "Mi Balmz Karaoke Tracks": TitleFormat.SONG_ARTIST,  # Song - Artist (Karaoke Instrumental)
    "Quantum Karaoke": TitleFormat.SONG_ARTIST,  # Song by Artist (Karaoke) - uses "by"
    "Lugn": TitleFormat.SONG_ARTIST,  # Song - Artist (Karaoke)
    "FrauKnoblauch": TitleFormat.SONG_ARTIST,  # Song - Artist (Karaoke/Instrumental)
    "Theo''s Music": TitleFormat.SONG_ARTIST,  # Song - Artist (Various) - mixed but mostly this
    # Channels that consistently use Artist - Song format
    "Sing King": TitleFormat.ARTIST_SONG,  # Artist - Song (Karaoke Version)
    "FakeyOke": TitleFormat.ARTIST_SONG,  # Artist - Song (FakeyOke Karaoke)
    "Sing Karaoke": TitleFormat.ARTIST_SONG,  # Artist - Song Karaoke
    "Sing2Piano | Piano Karaoke Instrumentals": TitleFormat.ARTIST_SONG,  # Artist - Song | Piano Instrumental
    "PARTY TYME KARAOKE CHANNEL": TitleFormat.ARTIST_SONG,  # Artist - Song (Party Tyme Karaoke)
    "Stingray Karaoke": TitleFormat.ARTIST_SONG,  # Artist - Song (Stingray Karaoke)
    "The Proper Volume Karaoke Studio": TitleFormat.ARTIST_SONG,  # Artist - Song (Instrumental)
    "Songjam: Official Karaoke": TitleFormat.ARTIST_SONG,  # Artist - Song (Backing Track)
    "AVD Karaoke": TitleFormat.ARTIST_SONG,  # Artist - Song (Karaoke Version)
    "EdKara": TitleFormat.ARTIST_SONG,  # Artist - Song Karaoke
    "KaraFun Deutschland - Karaoke": TitleFormat.ARTIST_SONG,  # Artist - Song Karaoke (NOT DE format!)
    # Channels with custom formats
    "BandaisuanKaraoke001": TitleFormat.CUSTOM,  # Song / Artist (Karaoke) - uses slash
}


# Confidence boost for known channel formats
CHANNEL_FORMAT_CONFIDENCE_BOOST = 0.15


def get_channel_format(channel_name: str) -> Optional[TitleFormat]:
    """Get the known format for a channel.

    Args:
        channel_name: The name of the YouTube channel

    Returns:
        The TitleFormat enum value if known, None otherwise
    """
    return CHANNEL_FORMATS.get(channel_name)


def is_channel_format_known(channel_name: str) -> bool:
    """Check if a channel's format is known.

    Args:
        channel_name: The name of the YouTube channel

    Returns:
        True if the channel format is known and not UNKNOWN
    """
    format_type = get_channel_format(channel_name)
    return format_type is not None and format_type != TitleFormat.UNKNOWN


def should_trust_channel_format(channel_name: str) -> bool:
    """Determine if we should trust the channel's known format over validation.

    Args:
        channel_name: The name of the YouTube channel

    Returns:
        True if we should use the channel's known format without validation
    """
    format_type = get_channel_format(channel_name)
    return format_type in [TitleFormat.ARTIST_SONG, TitleFormat.SONG_ARTIST]
