// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>

#include "diffusers/scheduler.hpp"

class StableDiffusionPipeline {
public:
    explicit StableDiffusionPipeline(const std::string& root_dir);

    StableDiffusionPipeline(const std::string& root_dir, const std::string& device, const ov::AnyMap& properties = {});

    // ability to override scheduler
    void set_scheduler(std::shared_ptr<Scheduler> scheduler);

    // with static shapes performance is better
    void reshape(const int batch_size, const int height, const int width);

    void compile(const std::string& device, const ov::AnyMap& properties = {});

    void apply_lora(const std::string& lora_path, float alpha);

    ov::Tensor generate(const std::string& positive_prompt,
                        const std::string& negative_prompt /* can be empty */,
                        float guidance_scale = 7.5f,
                        int64_t height = -1,
                        int64_t width = -1,
                        size_t num_inference_steps = 50,
                        size_t num_images_per_prompt = 1);

private:
    class StableDiffusionPipelineImpl;
    std::shared_ptr<StableDiffusionPipelineImpl> m_impl;
};
