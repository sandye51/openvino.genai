// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "diffusers/scheduler.hpp"

#include <fstream>

#include "utils.hpp"

#include "scheduler_lcm.hpp"
#include "scheduler_lms_discrete.hpp"

std::shared_ptr<Scheduler> Scheduler::from_config(const std::string& scheduler_config_path, SchedulerType scheduler_type) {
    std::ifstream file(scheduler_config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", scheduler_config_path);

    if (scheduler_type == SchedulerType::AUTO) {
        nlohmann::json data = nlohmann::json::parse(file);
        auto it = data.find("_class_name");
        OPENVINO_ASSERT(it != data.end(), "Failed to find '_class_name' field in ", scheduler_config_path);

        ov::genai::utils::read_json_param(data, "_class_name", scheduler_type);
        OPENVINO_ASSERT(scheduler_type != SchedulerType::AUTO, "Failed to guess scheduler based on its config ", scheduler_config_path);
    }

    std::shared_ptr<Scheduler> scheduler = nullptr;
    if (scheduler_type == SchedulerType::LCM) {
        // TODO: do we need to pass RNG generator somehow to LCM?
        scheduler = std::make_shared<LCMScheduler>(scheduler_config_path);
    } else if (scheduler_type == SchedulerType::LMS_DISCRETE) {
        scheduler = std::make_shared<LMSDiscreteScheduler>(scheduler_config_path);
    } else {
        OPENVINO_THROW("Unsupported scheduler type '", scheduler_type, ". Please, manually create scheduler via supported one");
    }

    return scheduler;
}
