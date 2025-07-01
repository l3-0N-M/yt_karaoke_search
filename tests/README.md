# Karaoke Collector Unit Tests

This directory contains comprehensive unit tests for the fixes implemented in the karaoke collector, specifically addressing the critical issues found during the collection run.

## Test Coverage

### 1. Database Tests (`test_db_optimized.py`)
Tests for the Discogs schema integration and database operations:
- ✅ Schema creation includes all Discogs fields
- ✅ Migration adds missing Discogs columns to existing databases
- ✅ Saving video data with complete Discogs information
- ✅ Year validation rejects future years (2025+)
- ✅ Discogs year is preferred over parsed year
- ✅ Null value handling in all fields
- ✅ Parameter 15 (featured_artists) special handling
- ✅ Connection pool management
- ✅ Database error retry logic

### 2. Web Search Pass Tests (`test_web_search_pass.py`)
Tests for null safety fixes in web search operations:
- ✅ FillerWordProcessor handles None inputs
- ✅ FillerWordProcessor handles non-string types
- ✅ Karaoke term removal from queries
- ✅ Word filtering with None values
- ✅ Safe string conversion for all types
- ✅ Unicode and encoding issues handling
- ✅ Parse method error handling
- ✅ SERP cache functionality
- ✅ Cache expiration and size limits

### 3. Rate Limiter Tests (`test_utils.py`)
Tests for improved Discogs rate limiting:
- ✅ Conservative 80% rate limiting
- ✅ Minimum 1-second interval enforcement
- ✅ Burst token limiting (3 tokens)
- ✅ 429 error handling with exponential backoff
- ✅ Retry-After header respect
- ✅ Maximum backoff limit (5 minutes)
- ✅ Success resets backoff state
- ✅ Concurrent request handling
- ✅ Token regeneration rate

### 4. Year Extraction Tests (`test_processor.py`)
Tests for release year validation:
- ✅ Current year rejection (prevents upload year confusion)
- ✅ Future year rejection
- ✅ Valid historical year acceptance
- ✅ Earlier year prioritization (original release)
- ✅ Reasonable year range validation (1900+)
- ✅ Decade indicator handling
- ✅ Pattern priority (parentheses > brackets > standalone)
- ✅ Description field extraction
- ✅ Multiple year handling (earliest selected)

## Running the Tests

### Setup
```bash
# Install test dependencies
pip install -r tests/requirements-test.txt
```

### Run All Tests
```bash
# From project root
python tests/run_tests.py all

# Or using pytest directly
pytest tests/unit -v --cov=collector
```

### Run Specific Test Modules
```bash
# Database tests only
python tests/run_tests.py db

# Web search tests only
python tests/run_tests.py web

# Rate limiter tests only
python tests/run_tests.py utils

# Year extraction tests only
python tests/run_tests.py processor
```

### Run Failed Tests Only
```bash
python tests/run_tests.py failed
```

### Generate Coverage Report
```bash
# Terminal report
pytest tests/unit --cov=collector --cov-report=term-missing

# HTML report
pytest tests/unit --cov=collector --cov-report=html:tests/htmlcov
```

## Test Fixtures

The `fixtures/test_data.py` file provides:
- Sample video data with various edge cases
- Problematic strings that previously caused issues
- Rate limit test scenarios
- Database migration test cases
- Mock objects for external dependencies

## Key Test Scenarios

### Database Schema Migration
- Tests migration from old schema without Discogs fields
- Verifies all new columns are added correctly
- Ensures backward compatibility

### Null Safety
- Every string operation tested with None input
- Complex data types (lists, dicts) handled gracefully
- Unicode and encoding edge cases covered

### Rate Limiting
- Simulates burst traffic scenarios
- Tests sustained load handling
- Verifies 429 error recovery

### Year Validation
- Rejects current year (2025) as likely upload year
- Prioritizes Discogs data over parsed years
- Handles multiple year formats correctly

## Integration with CI/CD

These tests can be integrated into your CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
- name: Run Unit Tests
  run: |
    pip install -r tests/requirements-test.txt
    pytest tests/unit --cov=collector --cov-fail-under=80
```

## Debugging Failed Tests

1. Run with verbose output:
   ```bash
   pytest tests/unit -vv -s
   ```

2. Run specific test:
   ```bash
   pytest tests/unit/test_db_optimized.py::TestOptimizedDatabaseManager::test_save_video_with_discogs_data -v
   ```

3. Use pdb debugger:
   ```bash
   pytest tests/unit --pdb
   ```

## Future Improvements

- Add integration tests for full collection workflow
- Add performance benchmarks for rate limiting
- Add stress tests for concurrent database access
- Mock external API calls more comprehensively