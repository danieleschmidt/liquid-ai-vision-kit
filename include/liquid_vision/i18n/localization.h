#pragma once

#include <string>
#include <unordered_map>
#include <memory>

namespace LiquidVision {
namespace I18n {

/**
 * @brief Language codes following ISO 639-1 standard
 */
enum class Language {
    EN = 0,  // English
    ES = 1,  // Spanish  
    FR = 2,  // French
    DE = 3,  // German
    JA = 4,  // Japanese
    ZH = 5,  // Chinese (Simplified)
    PT = 6,  // Portuguese
    RU = 7,  // Russian
    KO = 8,  // Korean
    IT = 9   // Italian
};

/**
 * @brief Region codes following ISO 3166-1 alpha-2 standard
 */
enum class Region {
    US = 0,  // United States
    EU = 1,  // European Union
    JP = 2,  // Japan
    CN = 3,  // China
    KR = 4,  // South Korea
    CA = 5,  // Canada
    AU = 6,  // Australia
    BR = 7,  // Brazil
    IN = 8,  // India
    GLOBAL = 9
};

/**
 * @brief Localization keys for system messages
 */
enum class MessageKey {
    SYSTEM_INIT,
    VISION_PROCESSING,
    NEURAL_INFERENCE,
    FLIGHT_CONTROL,
    ERROR_VISION_FAIL,
    ERROR_NETWORK_FAIL,
    WARNING_LOW_CONFIDENCE,
    WARNING_HIGH_LATENCY,
    INFO_SYSTEM_READY,
    INFO_LANDING_COMPLETE
};

/**
 * @brief Localization manager for multi-language support
 */
class LocalizationManager {
private:
    Language current_language_ = Language::EN;
    Region current_region_ = Region::US;
    
    // Language-specific message translations
    std::unordered_map<MessageKey, std::unordered_map<Language, std::string>> messages_;
    
    // Region-specific compliance settings
    std::unordered_map<Region, std::unordered_map<std::string, std::string>> compliance_;

public:
    LocalizationManager();
    ~LocalizationManager() = default;
    
    // Configuration
    void set_language(Language lang);
    void set_region(Region region);
    Language get_language() const { return current_language_; }
    Region get_region() const { return current_region_; }
    
    // Message translation
    std::string get_message(MessageKey key) const;
    std::string get_localized_string(const std::string& key) const;
    
    // Compliance helpers
    std::string get_privacy_notice() const;
    std::string get_data_retention_policy() const;
    bool is_feature_allowed(const std::string& feature) const;
    
    // Regional formatting
    std::string format_timestamp(uint64_t timestamp_us) const;
    std::string format_distance(float meters) const;
    std::string format_speed(float mps) const;
    std::string format_temperature(float celsius) const;
    
    // Utility functions
    static std::string language_to_string(Language lang);
    static std::string region_to_string(Region region);
    static Language string_to_language(const std::string& str);
    static Region string_to_region(const std::string& str);

private:
    void initialize_messages();
    void initialize_compliance();
};

/**
 * @brief Global localization instance
 */
extern std::unique_ptr<LocalizationManager> g_localization;

/**
 * @brief Convenience macros for localized strings
 */
#define LOC(key) (LiquidVision::I18n::g_localization->get_message(key))
#define LOCSTR(key) (LiquidVision::I18n::g_localization->get_localized_string(key))

/**
 * @brief GDPR/CCPA compliance checker
 */
class ComplianceChecker {
public:
    enum class Regulation {
        GDPR,    // EU General Data Protection Regulation
        CCPA,    // California Consumer Privacy Act
        PDPA,    // Singapore Personal Data Protection Act
        LGPD     // Brazil Lei Geral de Proteção de Dados
    };
    
    struct ComplianceStatus {
        bool data_processing_allowed = true;
        bool requires_explicit_consent = false;
        bool requires_data_anonymization = false;
        bool requires_data_localization = false;
        std::string privacy_notice_url;
        uint32_t data_retention_days = 365;
    };
    
    static ComplianceStatus check_compliance(Region region);
    static bool validate_data_processing(const std::string& data_type, Region region);
    static std::string get_required_consent_text(Region region);
};

} // namespace I18n
} // namespace LiquidVision