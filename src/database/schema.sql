-- Liquid Vision Database Schema
-- PostgreSQL schema for model management and flight telemetry

-- Create database if not exists
-- CREATE DATABASE liquid_vision_db;

-- Use the database
-- \c liquid_vision_db;

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Models table
CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    type VARCHAR(50) NOT NULL, -- LNN, CNN, RNN, etc.
    architecture JSONB NOT NULL, -- Network architecture details
    file_path TEXT NOT NULL,
    checksum VARCHAR(64), -- SHA256 hash of model file
    
    -- Performance metrics
    avg_inference_time_ms REAL,
    accuracy REAL,
    power_consumption_mw REAL,
    model_size_bytes BIGINT,
    
    -- Metadata
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    
    -- Constraints
    UNIQUE(name, version)
);

-- Create index for faster lookups
CREATE INDEX idx_models_name ON models(name);
CREATE INDEX idx_models_created_at ON models(created_at DESC);

-- Flight sessions table
CREATE TABLE IF NOT EXISTS flight_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES models(id) ON DELETE SET NULL,
    
    -- Time information
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    duration_seconds REAL GENERATED ALWAYS AS (
        EXTRACT(EPOCH FROM (end_time - start_time))
    ) STORED,
    
    -- Flight metrics
    total_distance_m REAL DEFAULT 0,
    max_altitude_m REAL DEFAULT 0,
    min_altitude_m REAL DEFAULT 0,
    average_speed_ms REAL DEFAULT 0,
    max_speed_ms REAL DEFAULT 0,
    
    -- Processing metrics
    total_frames_processed INTEGER DEFAULT 0,
    failed_frames INTEGER DEFAULT 0,
    avg_confidence REAL,
    
    -- Status and metadata
    status VARCHAR(50) DEFAULT 'active', -- active, completed, aborted, crashed
    abort_reason TEXT,
    metadata JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for flight sessions
CREATE INDEX idx_flight_sessions_model_id ON flight_sessions(model_id);
CREATE INDEX idx_flight_sessions_start_time ON flight_sessions(start_time DESC);
CREATE INDEX idx_flight_sessions_status ON flight_sessions(status);

-- Telemetry data table (partitioned by month for performance)
CREATE TABLE IF NOT EXISTS telemetry (
    id BIGSERIAL,
    session_id UUID NOT NULL REFERENCES flight_sessions(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Position data
    position_x REAL NOT NULL,
    position_y REAL NOT NULL,
    position_z REAL NOT NULL,
    
    -- Velocity data
    velocity_x REAL NOT NULL,
    velocity_y REAL NOT NULL,
    velocity_z REAL NOT NULL,
    
    -- Orientation (Euler angles in radians)
    roll REAL NOT NULL,
    pitch REAL NOT NULL,
    yaw REAL NOT NULL,
    
    -- Angular velocities
    roll_rate REAL,
    pitch_rate REAL,
    yaw_rate REAL,
    
    -- System status
    battery_voltage REAL,
    battery_percent REAL,
    armed BOOLEAN DEFAULT FALSE,
    flight_mode VARCHAR(50),
    
    -- Neural network output
    nn_forward_velocity REAL,
    nn_yaw_rate REAL,
    nn_target_altitude REAL,
    inference_time_ms REAL,
    confidence REAL,
    
    PRIMARY KEY (session_id, timestamp)
) PARTITION BY RANGE (timestamp);

-- Create partitions for telemetry (example for 2025)
CREATE TABLE telemetry_2025_01 PARTITION OF telemetry
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE telemetry_2025_02 PARTITION OF telemetry
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

CREATE TABLE telemetry_2025_03 PARTITION OF telemetry
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');

-- Add more partitions as needed...

-- Create indexes on telemetry partitions
CREATE INDEX idx_telemetry_session_timestamp ON telemetry(session_id, timestamp);

-- Training data table
CREATE TABLE IF NOT EXISTS training_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_name VARCHAR(255) NOT NULL,
    
    -- Data information
    num_samples INTEGER NOT NULL,
    data_type VARCHAR(50), -- images, video, sensor_data
    storage_path TEXT NOT NULL,
    
    -- Labels and annotations
    labels JSONB,
    annotation_format VARCHAR(50), -- COCO, YOLO, custom
    
    -- Metadata
    description TEXT,
    source VARCHAR(255),
    license VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(dataset_name)
);

-- Model training history
CREATE TABLE IF NOT EXISTS training_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES models(id) ON DELETE CASCADE,
    dataset_id UUID REFERENCES training_data(id) ON DELETE SET NULL,
    
    -- Training configuration
    config JSONB NOT NULL, -- hyperparameters, architecture, etc.
    
    -- Training metrics
    epochs_completed INTEGER DEFAULT 0,
    total_epochs INTEGER,
    best_loss REAL,
    best_accuracy REAL,
    training_time_seconds REAL,
    
    -- Status
    status VARCHAR(50) DEFAULT 'pending', -- pending, running, completed, failed
    error_message TEXT,
    
    -- Timestamps
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Performance benchmarks table
CREATE TABLE IF NOT EXISTS benchmarks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES models(id) ON DELETE CASCADE,
    
    -- Platform information
    platform VARCHAR(100) NOT NULL, -- cortex_m7, jetson_nano, x86, etc.
    hardware_details JSONB,
    
    -- Performance metrics
    avg_inference_time_ms REAL NOT NULL,
    min_inference_time_ms REAL NOT NULL,
    max_inference_time_ms REAL NOT NULL,
    std_dev_inference_time_ms REAL,
    
    -- Resource usage
    avg_power_consumption_mw REAL,
    peak_power_consumption_mw REAL,
    memory_usage_kb INTEGER,
    
    -- Accuracy metrics
    accuracy REAL,
    precision_score REAL,
    recall REAL,
    f1_score REAL,
    
    -- Test configuration
    test_dataset VARCHAR(255),
    num_test_samples INTEGER,
    test_conditions JSONB,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Waypoints for mission planning
CREATE TABLE IF NOT EXISTS waypoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mission_id UUID,
    sequence_number INTEGER NOT NULL,
    
    -- Position
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    altitude_m REAL,
    
    -- Or local coordinates
    local_x REAL,
    local_y REAL,
    local_z REAL,
    
    -- Navigation parameters
    speed_ms REAL DEFAULT 1.0,
    yaw_angle REAL,
    acceptance_radius_m REAL DEFAULT 2.0,
    hold_time_s REAL DEFAULT 0,
    
    -- Metadata
    name VARCHAR(255),
    description TEXT,
    actions JSONB, -- Special actions at waypoint
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- System events and logs
CREATE TABLE IF NOT EXISTS system_events (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(50) NOT NULL, -- info, warning, error, critical
    component VARCHAR(100), -- vision, control, telemetry, etc.
    message TEXT NOT NULL,
    details JSONB,
    session_id UUID REFERENCES flight_sessions(id) ON DELETE SET NULL
);

-- Create index for system events
CREATE INDEX idx_system_events_timestamp ON system_events(timestamp DESC);
CREATE INDEX idx_system_events_type ON system_events(event_type);
CREATE INDEX idx_system_events_session ON system_events(session_id);

-- Cache table for frequently accessed data
CREATE TABLE IF NOT EXISTS cache (
    key VARCHAR(255) PRIMARY KEY,
    value JSONB NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index for cache expiration
CREATE INDEX idx_cache_expires ON cache(expires_at);

-- Views for common queries

-- Recent flight sessions with statistics
CREATE OR REPLACE VIEW v_recent_flights AS
SELECT 
    fs.id,
    fs.start_time,
    fs.end_time,
    fs.duration_seconds,
    fs.total_distance_m,
    fs.max_altitude_m,
    fs.average_speed_ms,
    fs.total_frames_processed,
    fs.status,
    m.name as model_name,
    m.version as model_version
FROM flight_sessions fs
LEFT JOIN models m ON fs.model_id = m.id
ORDER BY fs.start_time DESC
LIMIT 100;

-- Model performance comparison
CREATE OR REPLACE VIEW v_model_performance AS
SELECT 
    m.id,
    m.name,
    m.version,
    m.type,
    COUNT(DISTINCT fs.id) as total_flights,
    AVG(fs.duration_seconds) as avg_flight_duration,
    AVG(fs.total_distance_m) as avg_distance,
    AVG(fs.avg_confidence) as avg_confidence,
    MIN(b.avg_inference_time_ms) as best_inference_time_ms,
    MIN(b.avg_power_consumption_mw) as best_power_consumption_mw
FROM models m
LEFT JOIN flight_sessions fs ON m.id = fs.model_id
LEFT JOIN benchmarks b ON m.id = b.model_id
GROUP BY m.id, m.name, m.version, m.type;

-- Functions and triggers

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update trigger to models table
CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to clean expired cache entries
CREATE OR REPLACE FUNCTION clean_expired_cache()
RETURNS void AS $$
BEGIN
    DELETE FROM cache WHERE expires_at < CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Function to get flight statistics
CREATE OR REPLACE FUNCTION get_flight_statistics(session_uuid UUID)
RETURNS TABLE (
    total_frames BIGINT,
    avg_inference_time REAL,
    min_inference_time REAL,
    max_inference_time REAL,
    avg_confidence REAL,
    max_altitude REAL,
    total_distance REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_frames,
        AVG(inference_time_ms)::REAL as avg_inference_time,
        MIN(inference_time_ms)::REAL as min_inference_time,
        MAX(inference_time_ms)::REAL as max_inference_time,
        AVG(confidence)::REAL as avg_confidence,
        MAX(position_z)::REAL as max_altitude,
        COALESCE(
            SUM(SQRT(
                POWER(position_x - LAG(position_x) OVER (ORDER BY timestamp), 2) +
                POWER(position_y - LAG(position_y) OVER (ORDER BY timestamp), 2) +
                POWER(position_z - LAG(position_z) OVER (ORDER BY timestamp), 2)
            ))::REAL, 0
        ) as total_distance
    FROM telemetry
    WHERE session_id = session_uuid;
END;
$$ LANGUAGE plpgsql;

-- Permissions (adjust as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO liquid_vision_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO liquid_vision_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO liquid_vision_user;