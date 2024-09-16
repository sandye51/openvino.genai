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

#include "diffusers/clip_text_model.hpp"
#include "diffusers/unet2d_condition_model.hpp"
#include "diffusers/autoencoder_kl.hpp"

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

class Text2ImagePipeline {
public:
    enum Type {
        STABLE_DIFFUSION,
        LCM,
        STABLE_DIFFUSION_XL,
        STABLE_DIFFUSION_3,
        FLUX,
    };

    struct GenerationConfig {
        // LCM: promp only w/o negative prompt
        // SD XL: prompt2 and negative_prompt2
        // FLUX: prompt2 (prompt if prompt2 is not defined explicitly)
        // SD 3: prompt2, prompt3 (with fallback to prompt) and negative_prompt2, negative_prompt3
        std::string prompt2, prompt3;
        std::string negative_prompt, negative_prompt2, negative_prompt3;

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

    explicit Text2ImagePipeline(const std::string& root_dir);

    Text2ImagePipeline(const std::string& root_dir, const std::string& device, const ov::AnyMap& properties = {});

    // creates either LCM or SD pipeline from building blocks
    Text2ImagePipeline(Type type, const CLIPTextModel& clip_text_encoder, const UNet2DConditionModel& unet, const AutoencoderKL& vae_decoder);

    GenerationConfig get_generation_config() const;
    void set_generation_config(const GenerationConfig& generation_config);

    // ability to override scheduler
    // TODO: do we need it?
    void set_scheduler(std::shared_ptr<Scheduler> scheduler);

    // with static shapes performance is better
    void reshape(const int num_images_per_prompt, const int height, const int width, const float guidance_scale);

    void apply_lora(const std::string& lora_path, float alpha);

    void compile(const std::string& device, const ov::AnyMap& properties = {});

    // Returns a tensor with the following dimensions [num_images_per_prompt, height, width, 3]
    ov::Tensor generate(const std::string& positive_prompt, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<ov::Tensor, Properties...> generate(
            const std::string& positive_prompt,
            Properties&&... properties) {
        return generate(positive_prompt, ov::AnyMap{std::forward<Properties>(properties)...});
    }

private:
    class DiffusionPipeline;
    std::shared_ptr<DiffusionPipeline> m_impl;

    class StableDiffusionPipeline;
};

static constexpr ov::Property<Generator::Ptr> random_generator{"random_generator"};

static constexpr ov::Property<std::string> prompt2{"prompt2"};
static constexpr ov::Property<std::string> prompt3{"prompt3"};

static constexpr ov::Property<std::string> negative_prompt{"negative_prompt"};
static constexpr ov::Property<std::string> negative_prompt2{"negative_prompt2"};
static constexpr ov::Property<std::string> negative_prompt3{"negative_prompt3"};

static constexpr ov::Property<size_t> num_images_per_prompt{"num_images_per_prompt"};

static constexpr ov::Property<float> guidance_scale{"guidance_scale"};
static constexpr ov::Property<int64_t> height{"height"};
static constexpr ov::Property<int64_t> width{"width"};
static constexpr ov::Property<size_t> num_inference_steps{"num_inference_steps"};

// similarly to LLMPipeline's GenerationConfig
std::pair<std::string, ov::Any> generation_config(const Text2ImagePipeline::GenerationConfig& generation_config);
