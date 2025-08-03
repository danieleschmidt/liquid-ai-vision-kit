#include "model_repository.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>

namespace LiquidVision {
namespace Repository {

ModelRepository::ModelRepository(std::shared_ptr<Database::DatabaseConnection> db) 
    : db_(db) {
}

bool ModelRepository::save_model(const ModelEntity& model) {
    std::stringstream sql;
    sql << "INSERT INTO models (id, name, version, type, architecture, "
        << "file_path, created_at, updated_at, metadata) "
        << "VALUES ('" << model.id << "', "
        << "'" << model.name << "', "
        << "'" << model.version << "', "
        << "'" << model.type << "', "
        << "'" << model.architecture << "', "
        << "'" << model.file_path << "', "
        << "'" << format_timestamp(model.created_at) << "', "
        << "'" << format_timestamp(model.updated_at) << "', "
        << "'" << model.metadata << "') "
        << "ON CONFLICT (id) DO UPDATE SET "
        << "name = EXCLUDED.name, "
        << "version = EXCLUDED.version, "
        << "updated_at = EXCLUDED.updated_at, "
        << "metadata = EXCLUDED.metadata";
    
    return db_->execute(sql.str());
}

std::optional<ModelEntity> ModelRepository::find_by_id(const std::string& id) {
    std::stringstream sql;
    sql << "SELECT * FROM models WHERE id = '" << id << "'";
    
    auto result = db_->query(sql.str());
    
    if (!result.success || !result.has_rows()) {
        return std::nullopt;
    }
    
    return parse_model_entity(result.rows[0]);
}

std::optional<ModelEntity> ModelRepository::find_latest_by_name(const std::string& name) {
    std::stringstream sql;
    sql << "SELECT * FROM models WHERE name = '" << name << "' "
        << "ORDER BY created_at DESC LIMIT 1";
    
    auto result = db_->query(sql.str());
    
    if (!result.success || !result.has_rows()) {
        return std::nullopt;
    }
    
    return parse_model_entity(result.rows[0]);
}

std::vector<ModelEntity> ModelRepository::find_all() {
    std::vector<ModelEntity> models;
    
    auto result = db_->query("SELECT * FROM models ORDER BY created_at DESC");
    
    if (result.success) {
        for (const auto& row : result.rows) {
            auto model = parse_model_entity(row);
            if (model) {
                models.push_back(*model);
            }
        }
    }
    
    return models;
}

bool ModelRepository::delete_model(const std::string& id) {
    std::stringstream sql;
    sql << "DELETE FROM models WHERE id = '" << id << "'";
    
    return db_->execute(sql.str());
}

bool ModelRepository::update_metadata(const std::string& id, const std::string& metadata) {
    std::stringstream sql;
    sql << "UPDATE models SET metadata = '" << metadata << "', "
        << "updated_at = '" << format_timestamp(std::chrono::system_clock::now()) << "' "
        << "WHERE id = '" << id << "'";
    
    return db_->execute(sql.str());
}

std::optional<ModelEntity> ModelRepository::parse_model_entity(const std::vector<std::string>& row) {
    if (row.size() < 9) {
        return std::nullopt;
    }
    
    ModelEntity model;
    model.id = row[0];
    model.name = row[1];
    model.version = row[2];
    model.type = row[3];
    model.architecture = row[4];
    model.file_path = row[5];
    // Parse timestamps - simplified for demonstration
    model.created_at = std::chrono::system_clock::now();
    model.updated_at = std::chrono::system_clock::now();
    model.metadata = row[8];
    
    return model;
}

std::string ModelRepository::format_timestamp(const std::chrono::system_clock::time_point& tp) {
    auto time_t = std::chrono::system_clock::to_time_t(tp);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// FlightDataRepository implementation
FlightDataRepository::FlightDataRepository(std::shared_ptr<Database::DatabaseConnection> db) 
    : db_(db) {
}

bool FlightDataRepository::save_flight_session(const FlightSession& session) {
    std::stringstream sql;
    sql << "INSERT INTO flight_sessions (id, start_time, end_time, "
        << "total_distance_m, max_altitude_m, average_speed_ms, "
        << "total_frames_processed, model_id, status) "
        << "VALUES ('" << session.id << "', "
        << "'" << format_timestamp(session.start_time) << "', "
        << "'" << format_timestamp(session.end_time) << "', "
        << session.total_distance_m << ", "
        << session.max_altitude_m << ", "
        << session.average_speed_ms << ", "
        << session.total_frames_processed << ", "
        << "'" << session.model_id << "', "
        << "'" << session.status << "')";
    
    return db_->execute(sql.str());
}

bool FlightDataRepository::save_telemetry(const TelemetryData& telemetry) {
    std::stringstream sql;
    sql << "INSERT INTO telemetry (session_id, timestamp, "
        << "position_x, position_y, position_z, "
        << "velocity_x, velocity_y, velocity_z, "
        << "roll, pitch, yaw, "
        << "battery_voltage, battery_percent, "
        << "inference_time_ms, confidence) "
        << "VALUES ('" << telemetry.session_id << "', "
        << "'" << format_timestamp(telemetry.timestamp) << "', "
        << telemetry.position[0] << ", " << telemetry.position[1] << ", " << telemetry.position[2] << ", "
        << telemetry.velocity[0] << ", " << telemetry.velocity[1] << ", " << telemetry.velocity[2] << ", "
        << telemetry.orientation[0] << ", " << telemetry.orientation[1] << ", " << telemetry.orientation[2] << ", "
        << telemetry.battery_voltage << ", " << telemetry.battery_percent << ", "
        << telemetry.inference_time_ms << ", " << telemetry.confidence << ")";
    
    return db_->execute(sql.str());
}

std::vector<FlightSession> FlightDataRepository::get_recent_sessions(int limit) {
    std::vector<FlightSession> sessions;
    
    std::stringstream sql;
    sql << "SELECT * FROM flight_sessions ORDER BY start_time DESC LIMIT " << limit;
    
    auto result = db_->query(sql.str());
    
    if (result.success) {
        for (const auto& row : result.rows) {
            auto session = parse_flight_session(row);
            if (session) {
                sessions.push_back(*session);
            }
        }
    }
    
    return sessions;
}

std::vector<TelemetryData> FlightDataRepository::get_telemetry_by_session(const std::string& session_id) {
    std::vector<TelemetryData> telemetry_data;
    
    std::stringstream sql;
    sql << "SELECT * FROM telemetry WHERE session_id = '" << session_id << "' "
        << "ORDER BY timestamp ASC";
    
    auto result = db_->query(sql.str());
    
    if (result.success) {
        for (const auto& row : result.rows) {
            auto telemetry = parse_telemetry_data(row);
            if (telemetry) {
                telemetry_data.push_back(*telemetry);
            }
        }
    }
    
    return telemetry_data;
}

FlightStatistics FlightDataRepository::get_statistics(const std::string& session_id) {
    FlightStatistics stats;
    
    std::stringstream sql;
    sql << "SELECT "
        << "COUNT(*) as total_records, "
        << "AVG(inference_time_ms) as avg_inference_time, "
        << "MIN(inference_time_ms) as min_inference_time, "
        << "MAX(inference_time_ms) as max_inference_time, "
        << "AVG(confidence) as avg_confidence, "
        << "MAX(position_z) as max_altitude, "
        << "AVG(battery_percent) as avg_battery "
        << "FROM telemetry WHERE session_id = '" << session_id << "'";
    
    auto result = db_->query(sql.str());
    
    if (result.success && result.has_rows()) {
        const auto& row = result.rows[0];
        stats.total_frames = std::stoi(row[0]);
        stats.avg_inference_time_ms = std::stof(row[1]);
        stats.min_inference_time_ms = std::stof(row[2]);
        stats.max_inference_time_ms = std::stof(row[3]);
        stats.avg_confidence = std::stof(row[4]);
        stats.max_altitude_m = std::stof(row[5]);
        stats.avg_battery_percent = std::stof(row[6]);
    }
    
    return stats;
}

std::optional<FlightSession> FlightDataRepository::parse_flight_session(const std::vector<std::string>& row) {
    if (row.size() < 9) {
        return std::nullopt;
    }
    
    FlightSession session;
    session.id = row[0];
    // Parse timestamps - simplified
    session.start_time = std::chrono::system_clock::now();
    session.end_time = std::chrono::system_clock::now();
    session.total_distance_m = std::stof(row[3]);
    session.max_altitude_m = std::stof(row[4]);
    session.average_speed_ms = std::stof(row[5]);
    session.total_frames_processed = std::stoi(row[6]);
    session.model_id = row[7];
    session.status = row[8];
    
    return session;
}

std::optional<TelemetryData> FlightDataRepository::parse_telemetry_data(const std::vector<std::string>& row) {
    if (row.size() < 15) {
        return std::nullopt;
    }
    
    TelemetryData telemetry;
    telemetry.session_id = row[0];
    telemetry.timestamp = std::chrono::system_clock::now(); // Simplified
    telemetry.position[0] = std::stof(row[2]);
    telemetry.position[1] = std::stof(row[3]);
    telemetry.position[2] = std::stof(row[4]);
    telemetry.velocity[0] = std::stof(row[5]);
    telemetry.velocity[1] = std::stof(row[6]);
    telemetry.velocity[2] = std::stof(row[7]);
    telemetry.orientation[0] = std::stof(row[8]);
    telemetry.orientation[1] = std::stof(row[9]);
    telemetry.orientation[2] = std::stof(row[10]);
    telemetry.battery_voltage = std::stof(row[11]);
    telemetry.battery_percent = std::stof(row[12]);
    telemetry.inference_time_ms = std::stof(row[13]);
    telemetry.confidence = std::stof(row[14]);
    
    return telemetry;
}

std::string FlightDataRepository::format_timestamp(const std::chrono::system_clock::time_point& tp) {
    auto time_t = std::chrono::system_clock::to_time_t(tp);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

} // namespace Repository
} // namespace LiquidVision