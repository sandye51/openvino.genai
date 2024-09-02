// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "types.hpp"

namespace ov {
namespace genai {
namespace utils {

template <>
void read_json_param(const nlohmann::json& data, const std::string& name, BetaSchedule& param) {
    if (data.contains(name) && data[name].is_string()) {
        std::string beta_schedule_str = data[name].get<std::string>();
        if (beta_schedule_str == "linear")
            param = BetaSchedule::LINEAR;
        else if (beta_schedule_str == "scaled_linear")
            param = BetaSchedule::SCALED_LINEAR;
        else if (beta_schedule_str == "squaredcos_cap_v2")
            param = BetaSchedule::SQUAREDCOS_CAP_V2;
        else if (!beta_schedule_str.empty()) {
            OPENVINO_THROW("Unsupported value for 'beta_schedule' ", beta_schedule_str);
        }
    }
}

template <>
void read_json_param(const nlohmann::json& data, const std::string& name, PredictionType& param) {
    if (data.contains(name) && data[name].is_string()) {
        std::string prediction_type_str = data[name].get<std::string>();
        if (prediction_type_str == "epsilon")
            param = PredictionType::EPSILON;
        else if (prediction_type_str == "sample")
            param = PredictionType::SAMPLE;
        else if (prediction_type_str == "v_prediction")
            param = PredictionType::V_PREDICTION;
        else if (!prediction_type_str.empty()) {
            OPENVINO_THROW("Unsupported value for 'prediction_type' ", prediction_type_str);
        }
    }
}

template <>
void read_json_param(const nlohmann::json& data, const std::string& name, SchedulerType& param) {
    if (data.contains(name) && data[name].is_string()) {
        std::string scheduler_type_str = data[name].get<std::string>();
        if (scheduler_type_str == "LCMScheduler")
            param = SchedulerType::LCM;
        else if (scheduler_type_str == "DDIMScheduler")
            // TODO: remove this TMP workaround
            param = SchedulerType::LMS_DISCRETE;
        else if (scheduler_type_str == "LMSDiscreteScheduler")
            param = SchedulerType::LMS_DISCRETE;
        else if (!scheduler_type_str.empty()) {
            OPENVINO_THROW("Unsupported value for 'prediction_type' ", scheduler_type_str);
        }
    }
}

}  // namespace utils
}  // namespace genai
}  // namespace ov

std::ostream& operator<<(std::ostream& os, const SchedulerType& scheduler_type) {
    switch (scheduler_type) {
    case SchedulerType::LCM:
        return os << "LCMScheduler";
    case SchedulerType::LMS_DISCRETE:
        return os << "LMSDiscreteScheduler";
    case SchedulerType::AUTO:
        return os << "AutoScheduler";
    default:
        OPENVINO_THROW("Unsupported scheduler type value");
    }
}
