#pragma once

#include <string>
#include <vector>
#include <optional>
#include <memory>
#include <chrono>
#include <array>
#include "../database/connection.h"

namespace LiquidVision {
namespace Repository {

/**
 * Model entity representing a trained neural network model
 */
struct ModelEntity {
    std::string id;
    std::string name;
    std::string version;
    std::string type;  // LNN, CNN, etc.
    std::string architecture;  // JSON description
    std::string file_path;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point updated_at;
    std::string metadata;  // JSON metadata
    
    // Performance metrics
    float avg_inference_time_ms = 0.0f;
    float accuracy = 0.0f;
    float power_consumption_mw = 0.0f;
    size_t model_size_bytes = 0;
};

/**
 * Flight session entity
 */
struct FlightSession {
    std::string id;
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    float total_distance_m = 0.0f;
    float max_altitude_m = 0.0f;
    float average_speed_ms = 0.0f;
    int total_frames_processed = 0;
    std::string model_id;
    std::string status;  // completed, aborted, crashed
};

/**
 * Telemetry data point
 */
struct TelemetryData {
    std::string session_id;
    std::chrono::system_clock::time_point timestamp;
    std::array<float, 3> position;  // x, y, z
    std::array<float, 3> velocity;  // vx, vy, vz
    std::array<float, 3> orientation;  // roll, pitch, yaw
    float battery_voltage = 0.0f;
    float battery_percent = 0.0f;
    float inference_time_ms = 0.0f;
    float confidence = 0.0f;
};

/**
 * Flight statistics
 */
struct FlightStatistics {
    int total_frames = 0;
    float avg_inference_time_ms = 0.0f;
    float min_inference_time_ms = 0.0f;
    float max_inference_time_ms = 0.0f;
    float avg_confidence = 0.0f;
    float total_flight_time_s = 0.0f;
    float total_distance_m = 0.0f;
    float max_altitude_m = 0.0f;
    float avg_battery_percent = 0.0f;
};

/**
 * Repository for model management
 */
class ModelRepository {
private:
    std::shared_ptr<Database::DatabaseConnection> db_;
    
public:
    explicit ModelRepository(std::shared_ptr<Database::DatabaseConnection> db);
    
    bool save_model(const ModelEntity& model);
    std::optional<ModelEntity> find_by_id(const std::string& id);
    std::optional<ModelEntity> find_latest_by_name(const std::string& name);
    std::vector<ModelEntity> find_all();
    bool delete_model(const std::string& id);
    bool update_metadata(const std::string& id, const std::string& metadata);
    
private:
    std::optional<ModelEntity> parse_model_entity(const std::vector<std::string>& row);
    std::string format_timestamp(const std::chrono::system_clock::time_point& tp);
};

/**
 * Repository for flight data and telemetry
 */
class FlightDataRepository {
private:
    std::shared_ptr<Database::DatabaseConnection> db_;
    
public:
    explicit FlightDataRepository(std::shared_ptr<Database::DatabaseConnection> db);
    
    bool save_flight_session(const FlightSession& session);
    bool save_telemetry(const TelemetryData& telemetry);
    std::vector<FlightSession> get_recent_sessions(int limit = 10);
    std::vector<TelemetryData> get_telemetry_by_session(const std::string& session_id);
    FlightStatistics get_statistics(const std::string& session_id);
    
private:
    std::optional<FlightSession> parse_flight_session(const std::vector<std::string>& row);
    std::optional<TelemetryData> parse_telemetry_data(const std::vector<std::string>& row);
    std::string format_timestamp(const std::chrono::system_clock::time_point& tp);
};

/**
 * Base repository with common CRUD operations
 */
template<typename T>
class BaseRepository {
protected:
    std::shared_ptr<Database::DatabaseConnection> db_;
    std::string table_name_;
    
public:
    BaseRepository(std::shared_ptr<Database::DatabaseConnection> db, const std::string& table_name)
        : db_(db), table_name_(table_name) {}
    
    virtual bool create(const T& entity) = 0;
    virtual std::optional<T> read(const std::string& id) = 0;
    virtual bool update(const T& entity) = 0;
    virtual bool delete_by_id(const std::string& id) = 0;
    virtual std::vector<T> list(int limit = 100, int offset = 0) = 0;
    
    size_t count() {
        std::string sql = "SELECT COUNT(*) FROM " + table_name_;
        auto result = db_->query(sql);
        
        if (result.success && result.has_rows()) {
            return std::stoull(result.rows[0][0]);
        }
        
        return 0;
    }
    
    bool exists(const std::string& id) {
        std::string sql = "SELECT 1 FROM " + table_name_ + " WHERE id = '" + id + "' LIMIT 1";
        auto result = db_->query(sql);
        return result.success && result.has_rows();
    }
};

} // namespace Repository
} // namespace LiquidVision