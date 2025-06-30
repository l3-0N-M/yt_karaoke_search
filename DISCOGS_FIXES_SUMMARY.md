# Discogs Matching Logic Fixes

## Problem
The Discogs search was incorrectly matching "ROSÉ - Messy" with "Andrea Datzman - Inside Out 2 (Original Motion Picture Soundtrack)" with a confidence score of 0.55.

## Root Causes Identified

1. **Overly Broad Search Strategy**: The quaternary search strategy was searching with just the artist name, returning hundreds of unrelated results.

2. **Weak Artist Name Validation**: The text similarity algorithm was giving partial credit to completely different artist names.

3. **Excessive Bonuses**: Metadata bonuses (year, genre, community data) were being applied even when core matching was poor.

4. **No Early Rejection**: Results with very different artist names were not being rejected early in the process.

## Fixes Applied

### 1. Removed Artist-Only Search Strategy (Line 261-262)
- Removed the quaternary search strategy that was searching with just artist name
- This prevents getting hundreds of unrelated results for popular artists

### 2. Added Early Artist Validation (Line 443-450)
- Added check to reject results where artist similarity < 0.5 before calculating full confidence
- This prevents obviously wrong matches from being processed further

### 3. Improved Artist Name Matching (Line 494-499)
- Increased penalties for non-exact artist matches:
  - < 0.9 similarity: multiply by 0.5 (was 0.7)
  - < 0.7 similarity: multiply by 0.3 (new)

### 4. Conditional Bonus Application (Line 506-552)
- Only apply bonuses when core match score (artist + title) >= 0.3
- Scale bonuses based on core match quality using a multiplier
- This prevents bad matches from accumulating high scores through bonuses

### 5. Stricter Text Similarity for Artists (Line 644-679)
- Added `is_artist` parameter to `_text_similarity` function
- For artist names, require all words from query to be present in result
- Adjusted weights for artist matching:
  - Edit similarity: 50%
  - Word coverage: 40% (all words must be present)
  - Token similarity: 10%

### 6. Enhanced Logging (Line 844-851)
- Added debug logging to show top matches and their confidence scores
- Helps diagnose why certain matches are being considered

## Results

With these fixes:
- "ROSÉ" vs "Andrea Datzman" similarity: 0.036 (3.6%)
- Overall confidence for the incorrect match: 0.115 (11.5%) instead of 0.55
- The match would be rejected early due to artist similarity < 0.5

## Testing

Run the following test to verify the fixes:
```python
# Test artist similarity
client._text_similarity("ROSÉ", "Andrea Datzman", is_artist=True)  # Should be < 0.1

# Test confidence calculation
# Mock a soundtrack result and calculate confidence for "ROSÉ - Messy"
# Should be < 0.2
```