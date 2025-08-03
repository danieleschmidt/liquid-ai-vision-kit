#pragma once

#include <string>
#include <memory>
#include <vector>
#include <queue>
#include <unordered_set>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>

namespace LiquidVision {
namespace Database {

/**
 * Database connection configuration
 */
struct ConnectionConfig {
    std::string host = "localhost";
    int port = 5432;
    std::string database = "liquid_vision";
    std::string username = "liquid_vision";
    std::string password = "";
    int connection_timeout = 30;  // seconds
    int max_retries = 3;
    bool auto_reconnect = true;
    bool debug = false;
};

/**
 * Query result structure
 */
struct QueryResult {
    bool success = false;
    std::string error_message;
    std::vector<std::string> columns;
    std::vector<std::vector<std::string>> rows;
    int rows_affected = 0;
    
    bool has_rows() const { return !rows.empty(); }
    size_t row_count() const { return rows.size(); }
    size_t column_count() const { return columns.size(); }
};

/**
 * Abstract database connection interface
 */
class DatabaseConnection {
public:
    virtual ~DatabaseConnection() = default;
    
    virtual bool connect() = 0;
    virtual bool disconnect() = 0;
    virtual bool is_connected() const = 0;
    
    virtual bool execute(const std::string& query) = 0;
    virtual QueryResult query(const std::string& sql) = 0;
    
    virtual bool begin_transaction() = 0;
    virtual bool commit() = 0;
    virtual bool rollback() = 0;
};

/**
 * PostgreSQL connection implementation
 */
class PostgreSQLConnection : public DatabaseConnection {
private:
    ConnectionConfig config_;
    bool connected_;
    std::chrono::steady_clock::time_point last_activity_;
    
public:
    explicit PostgreSQLConnection(const ConnectionConfig& config);
    ~PostgreSQLConnection() override;
    
    bool connect() override;
    bool disconnect() override;
    bool is_connected() const override { return connected_; }
    
    bool execute(const std::string& query) override;
    QueryResult query(const std::string& sql) override;
    
    bool begin_transaction() override;
    bool commit() override;
    bool rollback() override;
    
private:
    bool ensure_connected();
};

/**
 * Connection pool for efficient database access
 */
class ConnectionPool {
private:
    ConnectionConfig config_;
    size_t pool_size_;
    std::queue<std::shared_ptr<DatabaseConnection>> available_connections_;
    std::unordered_set<std::shared_ptr<DatabaseConnection>> in_use_connections_;
    
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread cleanup_thread_;
    bool running_;
    
public:
    ConnectionPool(const ConnectionConfig& config, size_t pool_size = 10);
    ~ConnectionPool();
    
    std::shared_ptr<DatabaseConnection> get_connection();
    void release_connection(std::shared_ptr<DatabaseConnection> conn);
    
    size_t get_available_connections() const;
    size_t get_in_use_connections() const;
    
    void shutdown();
    
private:
    void cleanup_loop();
};

} // namespace Database
} // namespace LiquidVision