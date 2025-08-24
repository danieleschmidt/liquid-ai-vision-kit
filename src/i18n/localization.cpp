#include "../include/liquid_vision/i18n/localization.h"
#include <ctime>
#include <iomanip>
#include <sstream>

namespace LiquidVision {
namespace I18n {

// Global localization instance
std::unique_ptr<LocalizationManager> g_localization = nullptr;

LocalizationManager::LocalizationManager() {
    initialize_messages();
    initialize_compliance();
}

void LocalizationManager::set_language(Language lang) {
    current_language_ = lang;
}

void LocalizationManager::set_region(Region region) {
    current_region_ = region;
}

std::string LocalizationManager::get_message(MessageKey key) const {
    auto msg_it = messages_.find(key);
    if (msg_it == messages_.end()) {
        return "Message not found";
    }
    
    auto lang_it = msg_it->second.find(current_language_);
    if (lang_it == msg_it->second.end()) {
        // Fallback to English
        lang_it = msg_it->second.find(Language::EN);
        if (lang_it == msg_it->second.end()) {
            return "Translation not available";
        }
    }
    
    return lang_it->second;
}

std::string LocalizationManager::get_privacy_notice() const {
    auto region_it = compliance_.find(current_region_);
    if (region_it == compliance_.end()) {
        region_it = compliance_.find(Region::GLOBAL);
    }
    
    auto notice_it = region_it->second.find("privacy_notice");
    return (notice_it != region_it->second.end()) ? notice_it->second : "Privacy notice not available";
}

std::string LocalizationManager::format_timestamp(uint64_t timestamp_us) const {
    time_t seconds = timestamp_us / 1000000;
    struct tm* timeinfo = localtime(&seconds);
    
    std::ostringstream oss;
    switch (current_region_) {
        case Region::US:
            oss << std::put_time(timeinfo, "%m/%d/%Y %I:%M:%S %p");
            break;
        case Region::EU:
        case Region::CN:
        default:
            oss << std::put_time(timeinfo, "%d/%m/%Y %H:%M:%S");
            break;
    }
    
    return oss.str();
}

std::string LocalizationManager::format_distance(float meters) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    
    switch (current_region_) {
        case Region::US:
            oss << (meters * 3.28084f) << " ft";
            break;
        default:
            oss << meters << " m";
            break;
    }
    
    return oss.str();
}

std::string LocalizationManager::format_speed(float mps) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    
    switch (current_region_) {
        case Region::US:
            oss << (mps * 2.237f) << " mph";
            break;
        default:
            oss << (mps * 3.6f) << " km/h";
            break;
    }
    
    return oss.str();
}

void LocalizationManager::initialize_messages() {
    // English messages
    messages_[MessageKey::SYSTEM_INIT][Language::EN] = "System initializing...";
    messages_[MessageKey::VISION_PROCESSING][Language::EN] = "Processing camera feed";
    messages_[MessageKey::NEURAL_INFERENCE][Language::EN] = "Running neural inference";
    messages_[MessageKey::FLIGHT_CONTROL][Language::EN] = "Sending flight commands";
    messages_[MessageKey::ERROR_VISION_FAIL][Language::EN] = "Vision processing failed";
    messages_[MessageKey::ERROR_NETWORK_FAIL][Language::EN] = "Neural network error";
    messages_[MessageKey::WARNING_LOW_CONFIDENCE][Language::EN] = "Low confidence prediction";
    messages_[MessageKey::WARNING_HIGH_LATENCY][Language::EN] = "High processing latency detected";
    messages_[MessageKey::INFO_SYSTEM_READY][Language::EN] = "System ready for operation";
    messages_[MessageKey::INFO_LANDING_COMPLETE][Language::EN] = "Landing sequence completed";
    
    // Spanish messages
    messages_[MessageKey::SYSTEM_INIT][Language::ES] = "Iniciando sistema...";
    messages_[MessageKey::VISION_PROCESSING][Language::ES] = "Procesando alimentación de cámara";
    messages_[MessageKey::NEURAL_INFERENCE][Language::ES] = "Ejecutando inferencia neuronal";
    messages_[MessageKey::FLIGHT_CONTROL][Language::ES] = "Enviando comandos de vuelo";
    messages_[MessageKey::ERROR_VISION_FAIL][Language::ES] = "Falló el procesamiento de visión";
    messages_[MessageKey::ERROR_NETWORK_FAIL][Language::ES] = "Error de red neuronal";
    
    // French messages  
    messages_[MessageKey::SYSTEM_INIT][Language::FR] = "Initialisation du système...";
    messages_[MessageKey::VISION_PROCESSING][Language::FR] = "Traitement du flux caméra";
    messages_[MessageKey::NEURAL_INFERENCE][Language::FR] = "Exécution de l'inférence neuronale";
    messages_[MessageKey::FLIGHT_CONTROL][Language::FR] = "Envoi des commandes de vol";
    messages_[MessageKey::ERROR_VISION_FAIL][Language::FR] = "Échec du traitement de vision";
    messages_[MessageKey::ERROR_NETWORK_FAIL][Language::FR] = "Erreur de réseau neuronal";
    
    // German messages
    messages_[MessageKey::SYSTEM_INIT][Language::DE] = "System wird initialisiert...";
    messages_[MessageKey::VISION_PROCESSING][Language::DE] = "Kamerafeed wird verarbeitet";
    messages_[MessageKey::NEURAL_INFERENCE][Language::DE] = "Neurale Inferenz läuft";
    messages_[MessageKey::FLIGHT_CONTROL][Language::DE] = "Flugkommandos werden gesendet";
    messages_[MessageKey::ERROR_VISION_FAIL][Language::DE] = "Bildverarbeitung fehlgeschlagen";
    messages_[MessageKey::ERROR_NETWORK_FAIL][Language::DE] = "Neuronaler Netzwerkfehler";
    
    // Japanese messages
    messages_[MessageKey::SYSTEM_INIT][Language::JA] = "システム初期化中...";
    messages_[MessageKey::VISION_PROCESSING][Language::JA] = "カメラフィード処理中";
    messages_[MessageKey::NEURAL_INFERENCE][Language::JA] = "ニューラル推論実行中";
    messages_[MessageKey::FLIGHT_CONTROL][Language::JA] = "フライトコマンド送信中";
    messages_[MessageKey::ERROR_VISION_FAIL][Language::JA] = "画像処理が失敗しました";
    messages_[MessageKey::ERROR_NETWORK_FAIL][Language::JA] = "ニューラルネットワークエラー";
    
    // Chinese messages
    messages_[MessageKey::SYSTEM_INIT][Language::ZH] = "系统初始化中...";
    messages_[MessageKey::VISION_PROCESSING][Language::ZH] = "处理相机画面";
    messages_[MessageKey::NEURAL_INFERENCE][Language::ZH] = "运行神经推理";
    messages_[MessageKey::FLIGHT_CONTROL][Language::ZH] = "发送飞行指令";
    messages_[MessageKey::ERROR_VISION_FAIL][Language::ZH] = "视觉处理失败";
    messages_[MessageKey::ERROR_NETWORK_FAIL][Language::ZH] = "神经网络错误";
}

void LocalizationManager::initialize_compliance() {
    // EU GDPR compliance
    compliance_[Region::EU]["privacy_notice"] = "We process personal data in accordance with GDPR. You have the right to access, rectify, and erase your data.";
    compliance_[Region::EU]["data_retention"] = "30";
    compliance_[Region::EU]["requires_consent"] = "true";
    compliance_[Region::EU]["data_localization"] = "true";
    
    // US CCPA compliance
    compliance_[Region::US]["privacy_notice"] = "We collect and use personal information as described in our privacy policy. California residents have additional rights.";
    compliance_[Region::US]["data_retention"] = "365";
    compliance_[Region::US]["requires_consent"] = "false";
    compliance_[Region::US]["data_localization"] = "false";
    
    // China data localization
    compliance_[Region::CN]["privacy_notice"] = "Personal data is processed and stored within China in accordance with local regulations.";
    compliance_[Region::CN]["data_retention"] = "180";
    compliance_[Region::CN]["requires_consent"] = "true";
    compliance_[Region::CN]["data_localization"] = "true";
    
    // Global default
    compliance_[Region::GLOBAL]["privacy_notice"] = "We respect your privacy and process data according to applicable laws.";
    compliance_[Region::GLOBAL]["data_retention"] = "365";
    compliance_[Region::GLOBAL]["requires_consent"] = "true";
    compliance_[Region::GLOBAL]["data_localization"] = "false";
}

ComplianceChecker::ComplianceStatus ComplianceChecker::check_compliance(Region region) {
    ComplianceStatus status;
    
    switch (region) {
        case Region::EU:
            status.requires_explicit_consent = true;
            status.requires_data_anonymization = true;
            status.requires_data_localization = true;
            status.data_retention_days = 30;
            status.privacy_notice_url = "https://liquid-ai-vision.com/privacy/eu";
            break;
            
        case Region::US:
            status.requires_explicit_consent = false;
            status.requires_data_anonymization = false;
            status.requires_data_localization = false;
            status.data_retention_days = 365;
            status.privacy_notice_url = "https://liquid-ai-vision.com/privacy/us";
            break;
            
        case Region::CN:
            status.requires_explicit_consent = true;
            status.requires_data_anonymization = false;
            status.requires_data_localization = true;
            status.data_retention_days = 180;
            status.privacy_notice_url = "https://liquid-ai-vision.cn/privacy";
            break;
            
        default:
            status.requires_explicit_consent = true;
            status.requires_data_anonymization = true;
            status.requires_data_localization = false;
            status.data_retention_days = 365;
            status.privacy_notice_url = "https://liquid-ai-vision.com/privacy";
            break;
    }
    
    return status;
}

std::string LocalizationManager::language_to_string(Language lang) {
    switch (lang) {
        case Language::EN: return "en";
        case Language::ES: return "es";
        case Language::FR: return "fr";
        case Language::DE: return "de";
        case Language::JA: return "ja";
        case Language::ZH: return "zh";
        case Language::PT: return "pt";
        case Language::RU: return "ru";
        case Language::KO: return "ko";
        case Language::IT: return "it";
        default: return "en";
    }
}

std::string LocalizationManager::region_to_string(Region region) {
    switch (region) {
        case Region::US: return "US";
        case Region::EU: return "EU";
        case Region::JP: return "JP";
        case Region::CN: return "CN";
        case Region::KR: return "KR";
        case Region::CA: return "CA";
        case Region::AU: return "AU";
        case Region::BR: return "BR";
        case Region::IN: return "IN";
        case Region::GLOBAL: return "GLOBAL";
        default: return "GLOBAL";
    }
}

} // namespace I18n
} // namespace LiquidVision