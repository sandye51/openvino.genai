// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <list>
#include <string>

#include "text2image/schedulers/types.hpp"
#include "text2image/schedulers/ischeduler.hpp"

namespace ov {
namespace genai {

class LMSDiscreteScheduler : public IScheduler {
public:
    struct Config {
        int32_t num_train_timesteps = 1000;
        float beta_start = 0.00085f, beta_end = 0.012f;
        BetaSchedule beta_schedule = BetaSchedule::SCALED_LINEAR;
        PredictionType prediction_type = PredictionType::EPSILON;
        std::vector<float> trained_betas = {};
        TimestepSpacing timestep_spacing = TimestepSpacing::LINSPACE;
        size_t steps_offset = 0;

        Config() = default;
        explicit Config(const std::string& scheduler_config_path);
    };

    explicit LMSDiscreteScheduler(const std::string scheduler_config_path);
    explicit LMSDiscreteScheduler(const Config& scheduler_config);

    void set_timesteps(size_t num_inference_steps) override;

    std::vector<std::int64_t> get_timesteps() const override;

    float get_init_noise_sigma() const override;

    void scale_model_input(ov::Tensor sample, size_t inference_step) override;

    std::map<std::string, ov::Tensor> step(ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step) override;

private:
    Config m_config;

    std::vector<float> m_alphas, m_betas, m_alphas_cumprod;
    std::vector<float> m_sigmas, m_log_sigmas;
    std::vector<int64_t> m_timesteps;
    std::list<std::vector<float>> m_derivative_list;

    int64_t _sigma_to_t(float sigma) const;
};

} // namespace genai
} // namespace ov
