// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "diffusers/stable_diffusion_pipeline.hpp"

#include <ctime>

#include "clip_text_model.hpp"
#include "unet2d_condition_model.hpp"
#include "autoencoder_kl.hpp"

#include "utils.hpp"

namespace {

ov::Tensor get_guidance_scale_embedding(float guidance_scale, uint32_t embedding_dim) {
    float w = guidance_scale * 1000;
    uint32_t half_dim = embedding_dim / 2;
    float emb = log(10000) / (half_dim - 1);

    ov::Shape embedding_shape = {1, embedding_dim};
    ov::Tensor w_embedding(ov::element::f32, embedding_shape);
    float* w_embedding_data = w_embedding.data<float>();

    for (size_t i = 0; i < half_dim; ++i) {
        float temp = std::exp((i * (-emb))) * w;
        w_embedding_data[i] = std::sin(temp);
        w_embedding_data[i + half_dim] = std::cos(temp);
    }

    if (embedding_dim % 2 == 1)
        w_embedding_data[embedding_dim - 1] = 0;

    return w_embedding;
}

// TODO: remove duplication with utils.hpp
template <typename T>
void read_anymap_param(const ov::AnyMap& config_map, const std::string& name, T& param) {
    auto it = config_map.find(name);
    if (it != config_map.end()) {
        param = it->second.as<T>();
    }
}

} // namespace

CppStdGenerator::CppStdGenerator(uint32_t seed)
    : gen(seed), normal(0.0f, 1.0f) {
}

float CppStdGenerator::next() {
    return normal(gen);
}

void StableDiffusionPipeline::GenerationConfig::update_generation_config(const ov::AnyMap& properties) {
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

std::pair<std::string, ov::Any> generation_config(const StableDiffusionPipeline::GenerationConfig& generation_config) {
    return {"SD_GENERATION_CONFIG", ov::Any::make<StableDiffusionPipeline::GenerationConfig>(generation_config)};
}

void StableDiffusionPipeline::GenerationConfig::validate() const {
    OPENVINO_ASSERT(guidance_scale >= 1.0f || negative_prompt.empty(), "Guidance scale < 1.0 ignores negative prompt");
}

class StableDiffusionPipeline::StableDiffusionPipelineImpl {
public:
    explicit StableDiffusionPipelineImpl(const std::string& root_dir) {
        const std::string model_index_path = root_dir + "/model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using ov::genai::utils::read_json_param;

        m_scheduler = Scheduler::from_config(root_dir + "/scheduler/scheduler_config.json");

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "CLIPTextModel") {
            m_clip_text_encoder = std::make_shared<CLIPTextModel>(root_dir);
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string unet = data["unet"][1].get<std::string>();
        if (unet == "UNet2DConditionModel") {
            m_unet = std::make_shared<UNet2DConditionModel>(root_dir);
        } else {
            OPENVINO_THROW("Unsupported '", unet, "' UNet type");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKL") {
            m_vae_decoder = std::make_shared<AutoencoderKL>(root_dir);
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        // initialize generation config
        initialize_generation_config(data["_class_name"].get<std::string>());
    }

    StableDiffusionPipelineImpl(const std::string& root_dir, const std::string& device, const ov::AnyMap& properties) {
        const std::string model_index_path = root_dir + "/model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using ov::genai::utils::read_json_param;

        m_scheduler = Scheduler::from_config(root_dir + "/scheduler/scheduler_config.json");

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "CLIPTextModel") {
            m_clip_text_encoder = std::make_shared<CLIPTextModel>(root_dir, device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string unet = data["unet"][1].get<std::string>();
        if (unet == "UNet2DConditionModel") {
            m_unet = std::make_shared<UNet2DConditionModel>(root_dir, device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", unet, "' UNet type");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKL") {
            m_vae_decoder = std::make_shared<AutoencoderKL>(root_dir, device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        // initialize generation config
        initialize_generation_config(data["_class_name"].get<std::string>());
    }

    void set_scheduler(std::shared_ptr<Scheduler> scheduler) {
        m_scheduler = scheduler;
    }

    void reshape(const int num_images_per_prompt, const int height, const int width, const float guidance_scale) {
        const size_t batch_size_multiplier = do_classifier_free_guidance(guidance_scale) ? 2 : 1;  // Unet accepts 2x batch in case of CFG
        m_clip_text_encoder->reshape(batch_size_multiplier);
        m_unet->reshape(num_images_per_prompt * batch_size_multiplier, height, width, m_clip_text_encoder->get_config().max_position_embeddings);
        m_vae_decoder->reshape(num_images_per_prompt, height, width);
    }

    void compile(const std::string& device, const ov::AnyMap& properties) {
        m_clip_text_encoder->compile(device, properties);
        m_unet->compile(device, properties);
        m_vae_decoder->compile(device, properties);
    }

    void apply_lora(const std::string& lora_path, float alpha) {
        std::map<std::string, InsertLoRA::LoRAMap> lora_weights = read_lora_adapters(lora_path, alpha);

        m_clip_text_encoder->apply_lora(lora_weights["text_encoder"]);
        m_unet->apply_lora(lora_weights["unet"]);
    }

    GenerationConfig get_generation_config() const {
        return m_generation_config;
    }

    void set_generation_config(const GenerationConfig& generation_config) {
        m_generation_config = generation_config;
    }

    ov::Tensor generate(const std::string& positive_prompt,
                        const ov::AnyMap& properties) {
        GenerationConfig generation_config = m_generation_config;
        generation_config.update_generation_config(properties);

        OPENVINO_ASSERT(generation_config.num_images_per_prompt == 1, "Currently only num_images_per_prompt = 1 is supported");

        // Stable Diffusion pipeline
        // see https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline#deconstruct-the-stable-diffusion-pipeline

        const auto& unet_config = m_unet->get_config();
        const size_t batch_size_multiplier = do_classifier_free_guidance(generation_config.guidance_scale) ? 2 : 1;  // Unet accepts 2x batch in case of CFG
        const size_t vae_scale_factor = m_unet->get_vae_scale_factor();

        if (generation_config.height < 0)
            generation_config.height = unet_config.sample_size * vae_scale_factor;
        if (generation_config.width < 0)
            generation_config.width = unet_config.sample_size * vae_scale_factor;

        if (generation_config.random_generator == nullptr) {
            uint32_t seed = time(NULL);
            generation_config.random_generator = std::make_shared<CppStdGenerator>(seed);
        }

        ov::Tensor encoder_hidden_states = m_clip_text_encoder->forward(positive_prompt, generation_config.negative_prompt,
            batch_size_multiplier > 1);
        m_unet->set_hidden_states("encoder_hidden_states", encoder_hidden_states);

        if (unet_config.time_cond_proj_dim >= 0) {
            ov::Tensor guidance_scale_embedding = get_guidance_scale_embedding(generation_config.guidance_scale, unet_config.time_cond_proj_dim);
            m_unet->set_hidden_states("timestep_cond", guidance_scale_embedding);
        }

        m_scheduler->set_timesteps(generation_config.num_inference_steps);
        std::vector<std::int64_t> timesteps = m_scheduler->get_timesteps();

        ov::Tensor denoised;
        for (uint32_t n = 0; n < generation_config.num_images_per_prompt; n++) {
            // latents are multiplied by 'init_noise_sigma'
            ov::Shape latent_shape{generation_config.num_images_per_prompt, unet_config.in_channels,
                                   generation_config.height / vae_scale_factor, generation_config.width / vae_scale_factor};
            ov::Shape latent_shape_cfg = latent_shape;
            latent_shape_cfg[0] *= batch_size_multiplier;

            ov::Tensor latent(ov::element::f32, latent_shape), latent_cfg(ov::element::f32, latent_shape_cfg);
            std::generate_n(latent.data<float>(), latent.get_size(), [&]() -> float {
                return generation_config.random_generator->next() * m_scheduler->get_init_noise_sigma();
            });

            for (size_t inference_step = 0; inference_step < generation_config.num_inference_steps; inference_step++) {
                // concat the same latent twice along a batch dimension in case of CFG
                latent.copy_to(
                    ov::Tensor(latent_cfg, {0, 0, 0, 0}, {1, latent_shape[1], latent_shape[2], latent_shape[3]}));
                if (batch_size_multiplier > 1) {
                    latent.copy_to(
                        ov::Tensor(latent_cfg, {1, 0, 0, 0}, {2, latent_shape[1], latent_shape[2], latent_shape[3]}));
                }

                m_scheduler->scale_model_input(latent_cfg, inference_step);

                ov::Tensor timestep(ov::element::i64, {1}, &timesteps[inference_step]);
                ov::Tensor noise_pred_tensor = m_unet->forward(latent_cfg, timestep);

                ov::Shape noise_pred_shape = noise_pred_tensor.get_shape();
                noise_pred_shape[0] = generation_config.num_images_per_prompt;

                ov::Tensor noisy_residual(noise_pred_tensor.get_element_type(), noise_pred_shape);

                if (batch_size_multiplier > 1) {
                    // perform guidance
                    const float* noise_pred_uncond = noise_pred_tensor.data<const float>();
                    const float* noise_pred_text = noise_pred_uncond + ov::shape_size(noise_pred_shape);

                    for (size_t i = 0; i < ov::shape_size(noise_pred_shape); ++i) {
                        noisy_residual.data<float>()[i] =
                            noise_pred_uncond[i] + generation_config.guidance_scale * (noise_pred_text[i] - noise_pred_uncond[i]);
                    }
                } else {
                    noisy_residual = noise_pred_tensor;
                }

                auto scheduler_step_result = m_scheduler->step(noisy_residual, latent, inference_step);
                latent = scheduler_step_result["latent"];

                // check whether scheduler returns "denoised" image, which should be passed to VAE decoder
                const auto it = scheduler_step_result.find("denoised");
                denoised = it != scheduler_step_result.end() ? it->second : latent;
            }
        }

        return m_vae_decoder->forward(denoised);
    }

private:
    bool do_classifier_free_guidance(float guidance_scale) const {
        return guidance_scale > 1.0 && m_unet->get_config().time_cond_proj_dim < 0;
    }

    void initialize_generation_config(const std::string& class_name) {
        OPENVINO_ASSERT(m_unet, "UNet model must be initialized first");
        const auto& unet_config = m_unet->get_config();
        const size_t vae_scale_factor = m_unet->get_vae_scale_factor();

        m_generation_config.height = unet_config.sample_size * vae_scale_factor;
        m_generation_config.width = unet_config.sample_size * vae_scale_factor;

        if (class_name == "StableDiffusionPipeline") {
            m_generation_config.guidance_scale = 7.5f;
            m_generation_config.num_inference_steps = 50;
        } else if (class_name == "StableDiffusionXLPipeline") {
            m_generation_config.guidance_scale = 5.0f;
            m_generation_config.num_inference_steps = 50;
        } else if (class_name == "LatentConsistencyModelPipeline") {
            m_generation_config.guidance_scale = 7.5f;
            m_generation_config.num_inference_steps = 50;
        } else {
            OPENVINO_THROW("Unsupported class_name '", class_name, "'. Please, contact OpenVINO GenAI developers");
        }
    }

    std::shared_ptr<Scheduler> m_scheduler;
    std::shared_ptr<CLIPTextModel> m_clip_text_encoder;
    std::shared_ptr<UNet2DConditionModel> m_unet;
    std::shared_ptr<AutoencoderKL> m_vae_decoder;
    StableDiffusionPipeline::GenerationConfig m_generation_config;
};

StableDiffusionPipeline::StableDiffusionPipeline(const std::string& root_dir)
    : m_impl(std::make_shared<StableDiffusionPipelineImpl>(root_dir)) {
}

StableDiffusionPipeline::StableDiffusionPipeline(const std::string& root_dir, const std::string& device, const ov::AnyMap& properties)
    : m_impl(std::make_shared<StableDiffusionPipelineImpl>(root_dir, device, properties)) {
}

void StableDiffusionPipeline::set_scheduler(std::shared_ptr<Scheduler> scheduler) {
    m_impl->set_scheduler(scheduler);
}

StableDiffusionPipeline::GenerationConfig StableDiffusionPipeline::get_generation_config() const {
    return m_impl->get_generation_config();
}

void StableDiffusionPipeline::set_generation_config(const GenerationConfig& generation_config) const {
    m_impl->set_generation_config(generation_config);
}

void StableDiffusionPipeline::reshape(const int num_images_per_prompt, const int height, const int width, const float guidance_scale) {
    m_impl->reshape(num_images_per_prompt, height, width, guidance_scale);
}

void StableDiffusionPipeline::compile(const std::string& device, const ov::AnyMap& properties) {
    m_impl->compile(device, properties);
}

void StableDiffusionPipeline::apply_lora(const std::string& lora_path, float alpha) {
    m_impl->apply_lora(lora_path, alpha);
}

ov::Tensor StableDiffusionPipeline::generate(const std::string& positive_prompt, const ov::AnyMap& generation_config) {
    return m_impl->generate(positive_prompt, generation_config);
}
