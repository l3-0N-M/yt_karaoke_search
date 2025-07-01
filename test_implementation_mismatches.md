# Test-Implementation Mismatches Report

## Summary

This report identifies all mismatches between test files and their corresponding implementation files. The main issues are:

1. **Method name changes**: `parse()` → `parse_title()` in AdvancedTitleParser
2. **Field renames**: `original_artist` → `artist`, `source` → `method` 
3. **Data type changes**: `featured_artists` from list to Optional[str]
4. **Enum usage**: PassType enum values instead of strings
5. **Moved fields**: Many ParseResult fields moved to metadata dict
6. **Missing methods**: Several helper methods referenced in tests don't exist
7. **Different result structures**: SearchResult vs dict returns

## 1. Advanced Parser Mismatches

### test_advanced_parser.py Issues:

**Line 22: `featured_artists` parameter mismatch**
- Test expects: `featured_artists=["Feat1", "Feat2"]` as a list parameter
- Implementation: `featured_artists` is an Optional[str] field, not a list
- Fix: Change test to use string format: `featured_artists="Feat1, Feat2"`

**Lines 36-41: Missing/incorrect metadata fields**
- Test expects: `version`, `language`, `year`, `is_cover` fields in metadata
- Implementation: These are not fields in ParseResult, should be in metadata dict
- Fix: Access via `result.metadata.get('version')` etc.

**Line 54: Missing `parse` method**
- Test calls: `parser.parse(title=..., description=...)`
- Implementation has: `parse_title(title, description, tags, channel_name)`
- Fix: Change all `parser.parse()` calls to `parser.parse_title()`

**Lines 100, 118, 130, 167, 247: Non-existent attributes**
- Test expects: `result.version`, `result.remix_info`, `result.is_cover`, `result.original_artist`, `result.year`, `result.language`, `result.genre`
- Implementation: These should be in `result.metadata` dict
- Fix: Access as `result.metadata.get('version')` etc.

**Line 233-235: Incorrect None handling**
- Test expects: `parse()` to return None for empty input
- Implementation: `parse_title()` likely returns a ParseResult with low confidence
- Fix: Check for `result.confidence < 0.1` instead of None

**Line 326: Non-existent field**
- Test expects: `result.additional_metadata`
- Implementation: Only has `result.metadata`
- Fix: Use `result.metadata` instead

## 2. Multi-Pass Controller Mismatches

### test_multi_pass_controller.py Issues:

**Lines 29, 33: Incorrect PassResult fields**
- Test uses: `source="test"` in ParseResult
- Implementation: ParseResult doesn't have a `source` field
- Fix: Remove `source` parameter or use `method` field

**Line 33: Incorrect PassType**
- Test uses: `PassType.BASIC`
- Implementation: PassType enum doesn't have `BASIC`, has `CHANNEL_TEMPLATE`, `MUSICBRAINZ_SEARCH`, etc.
- Fix: Use valid PassType values like `PassType.CHANNEL_TEMPLATE`

**Lines 57-75: Incorrect MultiPassResult structure**
- Test uses: `pass_results`, `passes_attempted` as list of strings, `improvements` dict
- Implementation: Has `passes_attempted` as List[PassResult], no `improvements` field
- Fix: Update test to match actual MultiPassResult structure

**Lines 84-102: Incorrect config structure**
- Test uses: `pass_progression`, `confidence_thresholds` dict, `parallel_execution`, etc.
- Implementation: MultiPassConfig has different structure with pass-specific configs
- Fix: Create proper MultiPassPassConfig instances

**Line 208: Incorrect PassResult field access**
- Test uses: `pr.pass_type == 'basic'`
- Implementation: `pass_type` is PassType enum, not string
- Fix: Use `pr.pass_type == PassType.CHANNEL_TEMPLATE`

**Lines 264, 375, 402: Missing methods**
- Test calls: `controller._merge_results()`, `controller._should_continue_parsing()`, `controller.get_statistics()`
- Implementation: These methods don't exist or have different signatures
- Fix: Remove these test cases or implement the methods

## 3. Channel Template Pass Mismatches

### test_channel_template_pass.py Issues:

**Lines 62-64: Incorrect parse method signature**
- Test calls: `parse(title=..., channel_id=...)`
- Implementation expects: `parse(title, description, tags, channel_name, channel_id, metadata)`
- Fix: Add all required parameters

**Lines 69, 98: Incorrect source field**
- Test expects: `result.source == "channel_template"`
- Implementation: ParseResult uses `method` field, not `source`
- Fix: Check `result.method` instead

**Lines 94, 134, 231: Missing methods**
- Test calls: `_learn_channel_patterns()`, `_extract_with_template()`, `_calculate_confidence()`
- Implementation: These are likely private or have different names
- Fix: Check actual implementation for correct method names

**Line 329: Incorrect featured_artists handling**
- Test expects: `"Guest" in result.featured_artists or result.featured_artists == ["Guest"]`
- Implementation: `featured_artists` is Optional[str], not list
- Fix: Check string containment only

## 4. DB Optimized Mismatches

### test_db_optimized.py Issues:

**Lines 22-25: Incorrect connection handling**
- Test expects: `get_connection()` to work with sqlite3 directly
- Implementation: Uses connection pooling and custom connection manager
- Fix: Update tests to use proper connection context manager

**Lines 54-68: Schema field mismatches**
- Test expects specific Discogs fields
- Implementation: May have different field names or structure
- Fix: Verify actual schema and update field names

## 5. YouTube Provider Mismatches

### test_youtube_provider.py Issues:

**Line 21: Incorrect mock path**
- Test mocks: `AsyncMock()` for YoutubeDL
- Implementation: yt_dlp uses synchronous methods
- Fix: Use regular Mock instead of AsyncMock

**Lines 38-61: Incorrect return structure**
- Test expects: Results as list of dicts with specific fields like `results[0]['id']`
- Implementation: YouTubeSearchProvider should return List[SearchResult] objects with fields like `video_id`, `url`, `title`, `channel`, etc.
- Fix: Update test to access `results[0].video_id` instead of `results[0]['id']`

**Line 72: Incorrect field name**
- Test expects: `results[0]['source'] == 'youtube'`
- Implementation: SearchResult has `provider` field, not `source`
- Fix: Check `results[0].provider == 'youtube'` instead

## 6. Discogs Search Pass Mismatches

### test_discogs_search_pass.py Issues:

**Lines 12-16: Import mismatches**
- Test imports: `DiscogsSearchPass, DiscogsClient, DiscogsMatch`
- Issue: DiscogsMatch exists but other imports need verification
- Fix: Verify all imports match actual module structure

**Lines 24-34: DiscogsMatch field order**
- Test creates: DiscogsMatch with specific field order
- Implementation: Has fields in different order and includes `metadata` field
- Fix: Update test to match actual dataclass field order

**Lines 42-64: Removed SearchCandidate class**
- Test comments indicate: SearchCandidate class no longer exists
- Implementation: This class was removed
- Fix: Good - test correctly identified and commented out obsolete code

**Line 16: DiscogsRateLimiter import**
- Test imports: `from collector.utils import DiscogsRateLimiter`
- Implementation: Need to verify if this is the correct import path
- Fix: Check actual location of DiscogsRateLimiter class

## 7. General Pattern Issues

### Common Problems Across Tests:

1. **Parse Method Signatures**: Tests often use keyword arguments while implementations expect positional arguments
2. **Async/Sync Confusion**: Some tests use AsyncMock where sync Mock is needed
3. **Enum vs String**: Tests use strings where implementation uses enums (PassType)
4. **Result Object Structure**: Tests expect different fields than what ParseResult/PassResult actually have
5. **Missing Imports**: Some tests may be missing proper imports for enums and dataclasses
6. **Field Name Changes**: Many fields were renamed (e.g., `original_artist` → `artist`, `source` → `method`)
7. **Data Type Mismatches**: Lists vs strings (e.g., `featured_artists`), missing fields moved to metadata dict

## Recommended Fixes Priority:

1. **High Priority**: 
   - Fix method signatures (parse vs parse_title)
   - Fix PassType enum usage
   - Fix missing/renamed methods

2. **Medium Priority**: 
   - Fix field access and data structure mismatches
   - Update ParseResult/PassResult field access
   - Fix metadata dict access patterns

3. **Low Priority**: 
   - Update mocking strategies
   - Fix test assertions for confidence scores
   - Clean up obsolete test code

Each test file needs to be updated to match the actual implementation's API and data structures.