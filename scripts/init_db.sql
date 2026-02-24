-- =============================================================================
-- Project Titan — PostgreSQL Database Initialization
-- =============================================================================
-- This script is executed automatically by the postgres Docker container
-- on first startup via /docker-entrypoint-initdb.d/init.sql
-- =============================================================================

-- Enable the pgcrypto extension for gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- =============================================================================
-- Trades table: records every trade from entry to exit
-- =============================================================================
CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker VARCHAR(10) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    entry_time TIMESTAMPTZ,
    exit_time TIMESTAMPTZ,
    entry_price DECIMAL(10,4),
    exit_price DECIMAL(10,4),
    quantity INTEGER NOT NULL,
    max_profit DECIMAL(10,2),
    max_loss DECIMAL(10,2),
    realized_pnl DECIMAL(10,2),
    commission DECIMAL(10,2),
    ml_confidence DECIMAL(5,4),
    regime VARCHAR(30),
    entry_iv_rank DECIMAL(5,2),
    entry_reasoning TEXT,
    exit_reasoning TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- Trade legs table: individual legs of multi-leg spread orders
-- =============================================================================
CREATE TABLE trade_legs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_id UUID REFERENCES trades(id) ON DELETE CASCADE,
    leg_type VARCHAR(10) NOT NULL,
    option_type VARCHAR(4) NOT NULL,
    strike DECIMAL(10,2) NOT NULL,
    expiry DATE NOT NULL,
    quantity INTEGER NOT NULL,
    fill_price DECIMAL(10,4),
    ib_order_id INTEGER,
    ib_con_id INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- Account snapshots: periodic snapshots of account state
-- =============================================================================
CREATE TABLE account_snapshots (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    net_liquidation DECIMAL(12,2),
    buying_power DECIMAL(12,2),
    excess_liquidity DECIMAL(12,2),
    realized_pnl_day DECIMAL(10,2),
    unrealized_pnl DECIMAL(10,2),
    total_positions INTEGER,
    regime VARCHAR(30)
);

-- =============================================================================
-- Circuit breaker state: persisted across restarts
-- =============================================================================
CREATE TABLE circuit_breaker_state (
    id SERIAL PRIMARY KEY,
    level VARCHAR(20) NOT NULL,
    triggered_at TIMESTAMPTZ,
    daily_pnl DECIMAL(10,2),
    weekly_pnl DECIMAL(10,2),
    monthly_pnl DECIMAL(10,2),
    total_drawdown_pct DECIMAL(5,4),
    high_water_mark DECIMAL(12,2),
    recovery_stage INTEGER DEFAULT 0,
    consecutive_winners INTEGER DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- ML model metadata: tracks trained model versions
-- =============================================================================
CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    version INTEGER NOT NULL,
    trained_at TIMESTAMPTZ DEFAULT NOW(),
    train_start DATE,
    train_end DATE,
    val_accuracy DECIMAL(5,4),
    val_sharpe DECIMAL(6,3),
    features_json JSONB,
    hyperparams_json JSONB,
    model_path VARCHAR(255),
    is_active BOOLEAN DEFAULT FALSE
);

-- =============================================================================
-- Agent decisions log: every AI agent action is recorded
-- =============================================================================
CREATE TABLE agent_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    agent VARCHAR(30) NOT NULL,
    trade_id UUID REFERENCES trades(id) ON DELETE SET NULL,
    decision VARCHAR(20) NOT NULL,
    reasoning TEXT,
    confidence DECIMAL(5,4),
    thinking_tokens INTEGER,
    latency_ms INTEGER,
    cost_usd DECIMAL(6,4)
);

-- =============================================================================
-- Indexes for query performance
-- =============================================================================
CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_trades_ticker ON trades(ticker);
CREATE INDEX idx_trades_strategy ON trades(strategy);
CREATE INDEX idx_trades_entry_time ON trades(entry_time);
CREATE INDEX idx_account_snapshots_ts ON account_snapshots(timestamp);
CREATE INDEX idx_agent_decisions_ts ON agent_decisions(timestamp);
CREATE INDEX idx_trade_legs_trade_id ON trade_legs(trade_id);
CREATE INDEX idx_model_versions_active ON model_versions(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_agent_decisions_agent ON agent_decisions(agent);
CREATE INDEX idx_agent_decisions_trade_id ON agent_decisions(trade_id);

-- =============================================================================
-- Insert initial circuit breaker state (NORMAL level)
-- =============================================================================
INSERT INTO circuit_breaker_state (level, daily_pnl, weekly_pnl, monthly_pnl, total_drawdown_pct, high_water_mark, recovery_stage, consecutive_winners)
VALUES ('NORMAL', 0.00, 0.00, 0.00, 0.0000, 150000.00, 0, 0);

-- =============================================================================
-- Function to auto-update updated_at timestamp on trades
-- =============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_trades_updated_at
    BEFORE UPDATE ON trades
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_circuit_breaker_updated_at
    BEFORE UPDATE ON circuit_breaker_state
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
