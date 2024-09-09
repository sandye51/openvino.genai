// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <random>
#include <ctime>
#include <cstdlib>

#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"

#include "diffusers/scheduler.hpp"

class Generator {
public:
    using Ptr = std::shared_ptr<Generator>;

    virtual float next() = 0;
};

class CppStdGenerator : public Generator {
public:
    // creates 'std::mt19937' with initial 'seed' to generate numbers within a range [0.0f, 1.0f]
    explicit CppStdGenerator(uint32_t seed);

    virtual float next() override;

private:
    std::mt19937 gen;
    std::normal_distribution<float> normal;
};

class StableDiffusionPipeline {
public:
    struct GenerationConfig {
        std::string negative_prompt;
        size_t num_images_per_prompt = 1;

        // random generator to have deterministic results
        Generator::Ptr random_generator = std::make_shared<CppStdGenerator>(42);

        // the following values depend on HF diffusers class used to perform generation
        float guidance_scale = 7.5f;
        int64_t height = -1;
        int64_t width = -1;
        size_t num_inference_steps = 50;

        void update_generation_config(const ov::AnyMap& config_map);

        // checks whether is config is valid
        void validate() const;

        template <typename... Properties>
        ov::util::EnableIfAllStringAny<void, Properties...> update_generation_config(Properties&&... properties) {
            return update_generation_config(ov::AnyMap{std::forward<Properties>(properties)...});
        }
    };

    explicit StableDiffusionPipeline(const std::string& root_dir);

    StableDiffusionPipeline(const std::string& root_dir, const std::string& device, const ov::AnyMap& properties = {});

    // ability to override scheduler
    void set_scheduler(std::shared_ptr<Scheduler> scheduler);

    // with static shapes performance is better
    void reshape(const int num_images_per_prompt, const int height, const int width, const float guidance_scale);

    GenerationConfig get_generation_config() const;
    void set_generation_config(const GenerationConfig& generation_config) const;

    void compile(const std::string& device, const ov::AnyMap& properties = {});

    void apply_lora(const std::string& lora_path, float alpha);

    // Returns a tensor with the following dimensions [num_images_per_prompt, height, width, 3]
    ov::Tensor generate(const std::string& positive_prompt, const ov::AnyMap& generation_config = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<ov::Tensor, Properties...> generate(
            const std::string& positive_prompt,
            Properties&&... properties) {
        return generate(positive_prompt, ov::AnyMap{std::forward<Properties>(properties)...});
    }

private:
    class StableDiffusionPipelineImpl;
    std::shared_ptr<StableDiffusionPipelineImpl> m_impl;
};

static constexpr ov::Property<Generator::Ptr> random_generator{"random_generator"};

static constexpr ov::Property<std::string> negative_prompt{"negative_prompt"};
static constexpr ov::Property<size_t> num_images_per_prompt{"num_images_per_prompt"};

static constexpr ov::Property<float> guidance_scale{"guidance_scale"};
static constexpr ov::Property<int64_t> height{"height"};
static constexpr ov::Property<int64_t> width{"width"};
static constexpr ov::Property<size_t> num_inference_steps{"num_inference_steps"};

// similarly to LLMPipeline's GenerationConfig
std::pair<std::string, ov::Any> generation_config(const StableDiffusionPipeline::GenerationConfig& generation_config);
