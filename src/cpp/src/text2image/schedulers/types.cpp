// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "text2image/schedulers/types.hpp"

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
void read_json_param(const nlohmann::json& data, const std::string& name, Text2ImagePipeline::Scheduler::Type& param) {
    if (data.contains(name) && data[name].is_string()) {
        std::string scheduler_type_str = data[name].get<std::string>();
        if (scheduler_type_str == "LCMScheduler")
            param = Text2ImagePipeline::Scheduler::LCM;
        else if (scheduler_type_str == "DDIMScheduler")
            // TODO: remove this TMP workaround
            param = Text2ImagePipeline::Scheduler::LMS_DISCRETE;
        else if (scheduler_type_str == "LMSDiscreteScheduler")
            param = Text2ImagePipeline::Scheduler::LMS_DISCRETE;
        else if (!scheduler_type_str.empty()) {
            OPENVINO_THROW("Unsupported value for 'prediction_type' ", scheduler_type_str);
        }
    }
}

template <>
void read_json_param(const nlohmann::json& data, const std::string& name, TimestepSpacing& param) {
    if (data.contains(name) && data[name].is_string()) {
        std::string timestep_spacing_str = data[name].get<std::string>();
        if (timestep_spacing_str == "linspace")
            param = TimestepSpacing::LINSPACE;
        else if (timestep_spacing_str == "trailing")
            param = TimestepSpacing::TRAILING;
        else if (timestep_spacing_str == "leading")
            param = TimestepSpacing::LEADING;
        else if (!timestep_spacing_str.empty()) {
            OPENVINO_THROW("Unsupported value for 'timestep_spacing' ", timestep_spacing_str);
        }
    }
}

}  // namespace utils
}  // namespace genai
}  // namespace ov

std::ostream& operator<<(std::ostream& os, const ov::genai::Text2ImagePipeline::Scheduler::Type& scheduler_type) {
    switch (scheduler_type) {
    case ov::genai::Text2ImagePipeline::Scheduler::Type::LCM:
        return os << "LCMScheduler";
    case ov::genai::Text2ImagePipeline::Scheduler::Type::LMS_DISCRETE:
        return os << "LMSDiscreteScheduler";
    case ov::genai::Text2ImagePipeline::Scheduler::Type::AUTO:
        return os << "AutoScheduler";
    default:
        OPENVINO_THROW("Unsupported scheduler type value");
    }
}
