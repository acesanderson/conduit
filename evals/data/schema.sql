-- 1. Dataset Items: Immutable reference documents from your Parquet store
CREATE TABLE dataset_items (
    source_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    content JSONB NOT NULL, -- The full GoldStandardDatum
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Eval Configs: The Recipes (Fingerprinted)
CREATE TABLE eval_configs (
    id UUID PRIMARY KEY,
    params JSONB NOT NULL,
    checksum TEXT UNIQUE NOT NULL, -- Deterministic hash of params
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Run Groups: The Experiment Session
CREATE TABLE run_groups (
    id UUID PRIMARY KEY,
    project_name TEXT NOT NULL,
    name TEXT NOT NULL,
    git_commit TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Eval Runs: The Atomic Results
CREATE TABLE eval_runs (
    id UUID PRIMARY KEY,
    config_id UUID REFERENCES eval_configs(id),
    group_id UUID REFERENCES run_groups(id) ON DELETE CASCADE,
    source_id TEXT REFERENCES dataset_items(source_id),
    
    output_summary TEXT NOT NULL,
    metrics JSONB NOT NULL, -- Optimized for indexing/querying
    trace JSONB NOT NULL,   -- Heavy blob, excluded from aggregate queries
    
    latency_ms FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indices for your analytical "Pressure Testing"
CREATE INDEX idx_runs_config ON eval_runs(config_id);
CREATE INDEX idx_runs_source ON eval_runs(source_id);
CREATE INDEX idx_runs_total_loss ON eval_runs ((metrics->>'L_total'));
