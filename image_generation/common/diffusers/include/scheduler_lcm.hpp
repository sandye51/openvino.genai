// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "scheduler.hpp"

class LCMScheduler : public Scheduler {
public:
    // values from https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lcm.py#L190
    struct Config {
        size_t num_train_timesteps = 1000;
        float beta_start = 0.00085f, beta_end = 0.012f;
        BetaSchedule beta_schedule = BetaSchedule::SCALED_LINEAR;
        std::vector<float> trained_betas = {};
        size_t original_inference_steps = 50;
        bool clip_sample = false;
        float clip_sample_range = 1.0f;
        bool set_alpha_to_one = true;
        size_t steps_offset = 0;
        PredictionType prediction_type = PredictionType::EPSILON;
        bool thresholding = false;
        float dynamic_thresholding_ratio = 0.995f;
        float sample_max_value = 1.0f;
        // std::string timestep_spacing = "leading";
        float timestep_scaling = 10.0f;
        bool rescale_betas_zero_snr = false;

        explicit Config(const std::string scheduler_config_path);
    };

    explicit LCMScheduler(const std::string scheduler_config_path);
    LCMScheduler(const Config& scheduler_config, bool read_torch_noise, uint32_t seed);

    void set_timesteps(size_t num_inference_steps) override;

    std::vector<std::int64_t> get_timesteps() const override;

    float get_init_noise_sigma() const override;

    void scale_model_input(ov::Tensor sample, size_t inference_step) override;

    std::map<std::string, ov::Tensor> step(ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step) override;

private:
    Config m_config;

    std::vector<float> m_alphas_cumprod;
    float m_final_alpha_cumprod;
    size_t m_num_inference_steps;
    float m_sigma_data;

    std::vector<int64_t> m_timesteps;

    bool m_read_torch_noise;

    std::mt19937 m_gen;
    std::normal_distribution<float> m_normal;
    uint32_t m_seed;

    std::vector<float> threshold_sample(const std::vector<float>& flat_sample);
    std::vector<float> randn_function(uint32_t size, uint32_t seed);
};
