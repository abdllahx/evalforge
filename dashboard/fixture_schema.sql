-- SQLite-flavored mirror of the Postgres schema, used by the read-only dashboard
-- when deployed on Streamlit Cloud. Arrays + JSONB collapse to TEXT (JSON-encoded).

CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY,
    ingested_at TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    feature TEXT NOT NULL,
    user_prompt TEXT NOT NULL,
    system_prompt TEXT,
    model TEXT NOT NULL,
    response TEXT NOT NULL,
    latency_ms INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    user_feedback TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    content_hash TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    config TEXT NOT NULL DEFAULT '{}',
    notes TEXT
);

CREATE TABLE IF NOT EXISTS samples (
    run_id INTEGER NOT NULL,
    log_id INTEGER NOT NULL,
    sample_strategy TEXT NOT NULL,
    PRIMARY KEY (run_id, log_id)
);

CREATE TABLE IF NOT EXISTS clusters (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL,
    cluster_idx INTEGER NOT NULL,
    name TEXT,
    description TEXT,
    size INTEGER NOT NULL,
    representative_log_ids TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS log_cluster_assignment (
    run_id INTEGER NOT NULL,
    log_id INTEGER NOT NULL,
    cluster_id INTEGER NOT NULL,
    is_outlier INTEGER NOT NULL DEFAULT 0,
    outlier_reasons TEXT NOT NULL DEFAULT '[]',
    embedding_2d_x REAL,
    embedding_2d_y REAL,
    PRIMARY KEY (run_id, log_id)
);

CREATE TABLE IF NOT EXISTS label_runs (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL,
    log_id INTEGER NOT NULL,
    pass_idx INTEGER NOT NULL,
    quality_score INTEGER,
    difficulty TEXT,
    expected_behavior TEXT,
    raw_response TEXT
);

CREATE TABLE IF NOT EXISTS labels (
    log_id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL,
    quality_score INTEGER,
    difficulty TEXT,
    expected_behavior TEXT,
    golden_answer TEXT,
    must_contain TEXT NOT NULL DEFAULT '[]',
    must_not_contain TEXT NOT NULL DEFAULT '[]',
    confidence REAL,
    needs_review INTEGER NOT NULL DEFAULT 0,
    review_status TEXT NOT NULL DEFAULT 'pending',
    reviewer_notes TEXT,
    labeled_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_dataset (
    id INTEGER PRIMARY KEY,
    log_id INTEGER NOT NULL UNIQUE,
    added_at TEXT NOT NULL,
    category TEXT,
    difficulty TEXT,
    quality_score INTEGER,
    expected_behavior TEXT,
    user_prompt TEXT NOT NULL,
    golden_answer TEXT,
    rubric TEXT,
    must_contain TEXT NOT NULL DEFAULT '[]',
    must_not_contain TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    candidate_model TEXT NOT NULL,
    candidate_label TEXT,
    total_cases INTEGER NOT NULL DEFAULT 0,
    passed INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    config TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS eval_results (
    id INTEGER PRIMARY KEY,
    eval_run_id INTEGER NOT NULL,
    test_case_id INTEGER NOT NULL,
    candidate_response TEXT,
    passed INTEGER NOT NULL,
    score REAL,
    judge_reasoning TEXT,
    failure_reasons TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS claude_call_log (
    id INTEGER PRIMARY KEY,
    called_at TEXT NOT NULL,
    run_id INTEGER,
    purpose TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    cached INTEGER NOT NULL,
    duration_ms INTEGER,
    success INTEGER NOT NULL,
    error TEXT
);
