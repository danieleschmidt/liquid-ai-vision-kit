#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <future>

namespace LiquidVision {
namespace Deployment {

/**
 * @brief Cloud regions for global deployment
 */
enum class CloudRegion {
    US_EAST_1,      // Virginia, USA
    US_WEST_1,      // California, USA  
    EU_WEST_1,      // Ireland
    EU_CENTRAL_1,   // Germany
    AP_SOUTHEAST_1, // Singapore
    AP_NORTHEAST_1, // Tokyo, Japan
    AP_NORTHEAST_3, // Seoul, South Korea
    SA_EAST_1,      // São Paulo, Brazil
    CA_CENTRAL_1,   // Canada
    ME_SOUTH_1      // Middle East
};

/**
 * @brief Deployment configuration for a region
 */
struct RegionConfig {
    CloudRegion region;
    std::string endpoint_url;
    std::string api_key;
    bool data_residency_required = false;
    bool encryption_at_rest = true;
    bool encryption_in_transit = true;
    uint32_t max_concurrent_users = 1000;
    uint32_t storage_limit_gb = 100;
    std::vector<std::string> allowed_features;
    std::string compliance_level; // "gdpr", "ccpa", "pdpa", "standard"
};

/**
 * @brief Health status of a regional deployment
 */
struct RegionHealth {
    CloudRegion region;
    bool online = false;
    float latency_ms = 0.0f;
    float cpu_usage_percent = 0.0f;
    float memory_usage_percent = 0.0f;
    float disk_usage_percent = 0.0f;
    uint32_t active_connections = 0;
    uint32_t error_rate_per_minute = 0;
    uint64_t last_health_check_us = 0;
};

/**
 * @brief Multi-region deployment manager
 */
class MultiRegionManager {
private:
    std::unordered_map<CloudRegion, RegionConfig> region_configs_;
    std::unordered_map<CloudRegion, RegionHealth> region_health_;
    CloudRegion primary_region_ = CloudRegion::US_EAST_1;
    std::vector<CloudRegion> active_regions_;
    
    // Load balancing
    std::unordered_map<CloudRegion, float> region_weights_;
    uint32_t round_robin_counter_ = 0;

public:
    MultiRegionManager();
    ~MultiRegionManager() = default;
    
    // Configuration
    bool add_region(const RegionConfig& config);
    bool remove_region(CloudRegion region);
    void set_primary_region(CloudRegion region);
    
    // Deployment operations
    std::future<bool> deploy_to_region(CloudRegion region);
    std::future<bool> deploy_to_all_regions();
    std::future<bool> update_region_config(CloudRegion region, const RegionConfig& config);
    
    // Health monitoring
    std::future<RegionHealth> check_region_health(CloudRegion region);
    std::vector<RegionHealth> get_all_health_status();
    bool is_region_healthy(CloudRegion region) const;
    
    // Load balancing
    CloudRegion select_best_region() const;
    CloudRegion select_region_for_user(const std::string& user_location) const;
    void update_region_weights();
    
    // Failover
    bool initiate_failover(CloudRegion failed_region);
    CloudRegion get_failover_region(CloudRegion primary) const;
    
    // Data compliance
    bool validate_data_residency(const std::string& data_type, CloudRegion region) const;
    std::vector<CloudRegion> get_compliant_regions(const std::string& compliance_requirement) const;
    
    // Utilities
    static std::string region_to_string(CloudRegion region);
    static CloudRegion string_to_region(const std::string& str);
    static std::string get_region_display_name(CloudRegion region);

private:
    void initialize_default_regions();
    float calculate_user_region_latency(const std::string& user_location, CloudRegion region) const;
    bool meets_compliance_requirements(CloudRegion region, const std::string& requirement) const;
};

/**
 * @brief Auto-scaling configuration
 */
struct AutoScalingConfig {
    uint32_t min_instances = 1;
    uint32_t max_instances = 10;
    uint32_t target_cpu_utilization = 70;
    uint32_t target_memory_utilization = 80;
    uint32_t scale_up_threshold = 85;
    uint32_t scale_down_threshold = 30;
    uint32_t scale_up_cooldown_minutes = 5;
    uint32_t scale_down_cooldown_minutes = 10;
};

/**
 * @brief Auto-scaling manager for regional deployments
 */
class AutoScaler {
private:
    std::unordered_map<CloudRegion, AutoScalingConfig> scaling_configs_;
    std::unordered_map<CloudRegion, uint32_t> current_instances_;
    std::unordered_map<CloudRegion, uint64_t> last_scaling_action_;

public:
    void configure_region(CloudRegion region, const AutoScalingConfig& config);
    bool should_scale_up(CloudRegion region, const RegionHealth& health) const;
    bool should_scale_down(CloudRegion region, const RegionHealth& health) const;
    std::future<bool> scale_region(CloudRegion region, uint32_t target_instances);
    uint32_t get_current_instances(CloudRegion region) const;
};

/**
 * @brief Global CDN integration for static assets
 */
class CDNManager {
private:
    std::string cdn_base_url_;
    std::unordered_map<CloudRegion, std::string> edge_locations_;

public:
    void set_cdn_base_url(const std::string& base_url);
    std::string get_asset_url(const std::string& asset_path, CloudRegion preferred_region) const;
    bool invalidate_cache(const std::string& asset_path);
    bool preload_assets(const std::vector<std::string>& asset_paths, CloudRegion region);
};

/**
 * @brief Global configuration manager
 */
class GlobalConfigManager {
private:
    std::unordered_map<std::string, std::string> global_settings_;
    std::unordered_map<CloudRegion, std::unordered_map<std::string, std::string>> region_overrides_;

public:
    void set_global_setting(const std::string& key, const std::string& value);
    void set_region_override(CloudRegion region, const std::string& key, const std::string& value);
    std::string get_setting(const std::string& key, CloudRegion region) const;
    bool load_from_file(const std::string& config_file_path);
    bool save_to_file(const std::string& config_file_path) const;
};

} // namespace Deployment
} // namespace LiquidVision