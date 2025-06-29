-- Migration 007: Add cache tables for search optimization
CREATE TABLE IF NOT EXISTS search_cache (
    key TEXT PRIMARY KEY,
    value BLOB NOT NULL,
    created_at TIMESTAMP NOT NULL,
    last_accessed TIMESTAMP NOT NULL,
    access_count INTEGER DEFAULT 0,
    ttl_seconds INTEGER,
    size_bytes INTEGER DEFAULT 0,
    metadata TEXT,
    namespace TEXT,
    query_hash TEXT
);

CREATE INDEX IF NOT EXISTS idx_search_cache_created ON search_cache(created_at);
CREATE INDEX IF NOT EXISTS idx_search_cache_accessed ON search_cache(last_accessed);
CREATE INDEX IF NOT EXISTS idx_search_cache_namespace ON search_cache(namespace);
CREATE INDEX IF NOT EXISTS idx_search_cache_query_hash ON search_cache(query_hash);

-- Table for tracking search query performance
CREATE TABLE IF NOT EXISTS search_analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    query_normalized TEXT,
    provider TEXT,
    result_count INTEGER DEFAULT 0,
    response_time_ms INTEGER,
    cache_hit BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_search_analytics_query ON search_analytics(query);
CREATE INDEX IF NOT EXISTS idx_search_analytics_provider ON search_analytics(provider);
CREATE INDEX IF NOT EXISTS idx_search_analytics_created ON search_analytics(created_at);

-- Table for fuzzy matching reference data
CREATE TABLE IF NOT EXISTS fuzzy_reference_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    data_type TEXT NOT NULL, -- 'artist' or 'song'
    original_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    popularity_score REAL DEFAULT 0.0,
    confidence REAL DEFAULT 1.0,
    source TEXT, -- 'musicbrainz', 'manual', 'extracted'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fuzzy_ref_type ON fuzzy_reference_data(data_type);
CREATE INDEX IF NOT EXISTS idx_fuzzy_ref_normalized ON fuzzy_reference_data(normalized_text);
CREATE INDEX IF NOT EXISTS idx_fuzzy_ref_popularity ON fuzzy_reference_data(popularity_score);

-- Trigger for fuzzy reference data updated_at
CREATE TRIGGER IF NOT EXISTS trg_fuzzy_ref_updated
AFTER UPDATE ON fuzzy_reference_data
FOR EACH ROW
WHEN OLD.updated_at = NEW.updated_at
BEGIN
  UPDATE fuzzy_reference_data SET updated_at = CURRENT_TIMESTAMP
  WHERE id = NEW.id;
END;

INSERT OR REPLACE INTO schema_info(version) VALUES (7);
