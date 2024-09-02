// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "openvino/runtime/tensor.hpp"

enum class SchedulerType {
    AUTO = 0,
    LCM = 1,
    LMS_DISCRETE = 2,
};

class Scheduler {
public:
    static std::shared_ptr<Scheduler> from_config(const std::string& scheduler_config_path,
                                                  SchedulerType scheduler_type = SchedulerType::AUTO);

    virtual void set_timesteps(size_t num_inference_steps) = 0;

    virtual std::vector<std::int64_t> get_timesteps() const = 0;

    virtual float get_init_noise_sigma() const = 0;

    virtual void scale_model_input(ov::Tensor sample, size_t inference_step) = 0;

    virtual std::map<std::string, ov::Tensor> step(ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step) = 0;
};
