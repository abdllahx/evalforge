-- evalforge schema

CREATE TABLE IF NOT EXISTS logs (
    id BIGSERIAL PRIMARY KEY,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    occurred_at TIMESTAMPTZ NOT NULL,
    feature TEXT NOT NULL,
    user_prompt TEXT NOT NULL,
    system_prompt TEXT,
    model TEXT NOT NULL,
    response TEXT NOT NULL,
    latency_ms INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    user_feedback TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    content_hash TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS logs_feature_idx ON logs (feature);
CREATE INDEX IF NOT EXISTS logs_occurred_at_idx ON logs (occurred_at);
CREATE INDEX IF NOT EXISTS logs_user_feedback_idx ON logs (user_feedback);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id BIGSERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running',
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS samples (
    run_id BIGINT NOT NULL REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    log_id BIGINT NOT NULL REFERENCES logs(id) ON DELETE CASCADE,
    sample_strategy TEXT NOT NULL,
    PRIMARY KEY (run_id, log_id)
);

CREATE TABLE IF NOT EXISTS clusters (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    cluster_idx INTEGER NOT NULL,
    name TEXT,
    description TEXT,
    size INTEGER NOT NULL,
    representative_log_ids BIGINT[] NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS log_cluster_assignment (
    run_id BIGINT NOT NULL,
    log_id BIGINT NOT NULL REFERENCES logs(id) ON DELETE CASCADE,
    cluster_id BIGINT NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
    is_outlier BOOLEAN NOT NULL DEFAULT FALSE,
    outlier_reasons TEXT[] NOT NULL DEFAULT '{}',
    embedding_2d_x DOUBLE PRECISION,
    embedding_2d_y DOUBLE PRECISION,
    PRIMARY KEY (run_id, log_id)
);

CREATE TABLE IF NOT EXISTS label_runs (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    log_id BIGINT NOT NULL REFERENCES logs(id) ON DELETE CASCADE,
    pass_idx INTEGER NOT NULL,
    quality_score INTEGER,
    difficulty TEXT,
    expected_behavior TEXT,
    raw_response JSONB
);

CREATE TABLE IF NOT EXISTS labels (
    log_id BIGINT PRIMARY KEY REFERENCES logs(id) ON DELETE CASCADE,
    run_id BIGINT NOT NULL REFERENCES pipeline_runs(id),
    quality_score INTEGER,
    difficulty TEXT,
    expected_behavior TEXT,
    golden_answer TEXT,
    must_contain TEXT[] NOT NULL DEFAULT '{}',
    must_not_contain TEXT[] NOT NULL DEFAULT '{}',
    confidence DOUBLE PRECISION,
    needs_review BOOLEAN NOT NULL DEFAULT FALSE,
    review_status TEXT NOT NULL DEFAULT 'pending',
    reviewer_notes TEXT,
    labeled_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS eval_dataset (
    id BIGSERIAL PRIMARY KEY,
    log_id BIGINT NOT NULL REFERENCES logs(id) ON DELETE CASCADE,
    added_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    category TEXT,
    difficulty TEXT,
    quality_score INTEGER,
    expected_behavior TEXT,
    user_prompt TEXT NOT NULL,
    golden_answer TEXT,
    rubric JSONB,
    must_contain TEXT[] NOT NULL DEFAULT '{}',
    must_not_contain TEXT[] NOT NULL DEFAULT '{}',
    UNIQUE (log_id)
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id BIGSERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    candidate_model TEXT NOT NULL,
    candidate_label TEXT,
    total_cases INTEGER NOT NULL DEFAULT 0,
    passed INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    config JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS eval_results (
    id BIGSERIAL PRIMARY KEY,
    eval_run_id BIGINT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    test_case_id BIGINT NOT NULL REFERENCES eval_dataset(id) ON DELETE CASCADE,
    candidate_response TEXT,
    passed BOOLEAN NOT NULL,
    score DOUBLE PRECISION,
    judge_reasoning TEXT,
    failure_reasons TEXT[] NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS eval_results_run_idx ON eval_results (eval_run_id);

CREATE TABLE IF NOT EXISTS claude_call_log (
    id BIGSERIAL PRIMARY KEY,
    called_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id BIGINT,
    purpose TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    cached BOOLEAN NOT NULL,
    duration_ms INTEGER,
    success BOOLEAN NOT NULL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS claude_call_log_run_idx ON claude_call_log (run_id);
CREATE INDEX IF NOT EXISTS claude_call_log_purpose_idx ON claude_call_log (purpose);
