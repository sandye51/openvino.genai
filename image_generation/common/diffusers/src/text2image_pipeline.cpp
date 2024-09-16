// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "stable_diffusion_pipeline.hpp"

#include "utils.hpp"

CppStdGenerator::CppStdGenerator(uint32_t seed)
    : gen(seed), normal(0.0f, 1.0f) {
}

float CppStdGenerator::next() {
    return normal(gen);
}

//
// GenerationConfig
//

std::pair<std::string, ov::Any> generation_config(const Text2ImagePipeline::GenerationConfig& generation_config) {
    return {"SD_GENERATION_CONFIG", ov::Any::make<Text2ImagePipeline::GenerationConfig>(generation_config)};
}

void Text2ImagePipeline::GenerationConfig::update_generation_config(const ov::AnyMap& properties) {
    // override whole generation config first
    read_anymap_param(properties, "SD_GENERATION_CONFIG", *this);
    // then try per-parameter values
    read_anymap_param(properties, "negative_prompt", negative_prompt);
    read_anymap_param(properties, "num_images_per_prompt", num_images_per_prompt);
    read_anymap_param(properties, "random_generator", random_generator);
    read_anymap_param(properties, "guidance_scale", guidance_scale);
    read_anymap_param(properties, "height", height);
    read_anymap_param(properties, "width", width);
    read_anymap_param(properties, "num_inference_steps", num_inference_steps);
 
    validate();
}

void Text2ImagePipeline::GenerationConfig::validate() const {
    OPENVINO_ASSERT(guidance_scale >= 1.0f || negative_prompt.empty(), "Guidance scale < 1.0 ignores negative prompt");
}

//
// Text2ImagePipeline
//

Text2ImagePipeline::Text2ImagePipeline(const std::string& root_dir) {
    const std::string class_name = get_class_name(root_dir);

    if (class_name == "StableDiffusionPipeline" || 
        class_name == "LatentConsistencyModelPipeline") {
        m_impl = std::make_shared<StableDiffusionPipeline>(root_dir);
    } else {
        OPENVINO_THROW("Unsupported text to image generation pipeline '", class_name, "'");
    }
}

Text2ImagePipeline::Text2ImagePipeline(const std::string& root_dir, const std::string& device, const ov::AnyMap& properties) {
    const std::string class_name = get_class_name(root_dir);

    if (class_name == "StableDiffusionPipeline" || 
        class_name == "LatentConsistencyModelPipeline") {
        m_impl = std::make_shared<StableDiffusionPipeline>(root_dir, device, properties);
    } else {
        OPENVINO_THROW("Unsupported text to image generation pipeline '", class_name, "'");
    }
}

Text2ImagePipeline::Text2ImagePipeline(Type type, const CLIPTextModel& clip_text_encoder, const UNet2DConditionModel& unet, const AutoencoderKL& vae_decoder) {
    switch (type) {
        case Type::LCM:
        case Type::STABLE_DIFFUSION:
            m_impl = std::make_shared<StableDiffusionPipeline>(clip_text_encoder, unet, vae_decoder);
            break;
        default:
            OPENVINO_THROW("Unsupported pipeline type");
    };
}

Text2ImagePipeline::GenerationConfig Text2ImagePipeline::get_generation_config() const {
    return m_impl->get_generation_config();
}

void Text2ImagePipeline::set_generation_config(const GenerationConfig& generation_config) {
    m_impl->set_generation_config(generation_config);
}

void Text2ImagePipeline::set_scheduler(std::shared_ptr<Scheduler> scheduler) {
    m_impl->set_scheduler(scheduler);
}

void Text2ImagePipeline::apply_lora(const std::string& lora_path, float alpha) {
    m_impl->apply_lora(lora_path, alpha);
}

void Text2ImagePipeline::reshape(const int num_images_per_prompt, const int height, const int width, const float guidance_scale) {
    m_impl->reshape(num_images_per_prompt, height, width, guidance_scale);
}

void Text2ImagePipeline::compile(const std::string& device, const ov::AnyMap& properties) {
    m_impl->compile(device, properties);
}

ov::Tensor Text2ImagePipeline::generate(const std::string& positive_prompt, const ov::AnyMap& properties) {
    return m_impl->generate(positive_prompt, properties);
}
