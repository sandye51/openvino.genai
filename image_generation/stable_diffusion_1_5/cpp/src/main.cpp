// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "cxxopts.hpp"
#include "imwrite.hpp"

#include "diffusers/lora.hpp"
#include "diffusers/scheduler.hpp"

#include "openvino/genai/tokenizer.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"

#include "utils.hpp"

class Timer {
    const decltype(std::chrono::steady_clock::now()) m_start;

public:
    Timer(const std::string& scope) : m_start(std::chrono::steady_clock::now()) {
        (std::cout << scope << ": ").flush();
    }

    ~Timer() {
        auto m_end = std::chrono::steady_clock::now();
        std::cout << std::chrono::duration<double, std::milli>(m_end - m_start).count() << " ms" << std::endl;
    }
};

ov::Tensor randn_tensor(ov::Shape shape, bool use_np_latents, uint32_t seed = 42) {
    ov::Tensor noise(ov::element::f32, shape);
    if (use_np_latents) {
        // read np generated latents with defaut seed 42
        const char* latent_file_name = "../np_latents_512x512.txt";
        std::ifstream latent_copy_file(latent_file_name, std::ios::ate);
        OPENVINO_ASSERT(latent_copy_file.is_open(), "Cannot open ", latent_file_name);

        size_t file_size = latent_copy_file.tellg() / sizeof(float);
        OPENVINO_ASSERT(file_size >= noise.get_size(),
                        "Cannot generate ",
                        noise.get_shape(),
                        " with ",
                        latent_file_name,
                        ". File size is small");

        latent_copy_file.seekg(0, std::ios::beg);
        for (size_t i = 0; i < noise.get_size(); ++i)
            latent_copy_file >> noise.data<float>()[i];
    } else {
        std::mt19937 gen{seed};
        std::normal_distribution<float> normal{0.0f, 1.0f};
        std::generate_n(noise.data<float>(), noise.get_size(), [&]() {
            return normal(gen);
        });
    }
    return noise;
}

class IAdaptedModel {
public:
    explicit IAdaptedModel(const std::string& model_path) {
        m_model = ov::Core().read_model(model_path);
    }

    void apply_lora(InsertLoRA::LoRAMap& lora_map) {
        if (!lora_map.empty()) {
            ov::pass::Manager manager;
            manager.register_pass<InsertLoRA>(lora_map);
            manager.run_passes(m_model);
        }
    }

    void compile(const std::string& device, const ov::AnyMap& properties = {}) {
        OPENVINO_ASSERT(m_model, "Model has been already compiled");
        ov::CompiledModel compiled_model = ov::Core().compile_model(m_model, device, properties);
        m_request = compiled_model.create_infer_request();
        // release the original model
        m_model.reset();
    }

protected:
    std::shared_ptr<ov::Model> m_model;
    ov::InferRequest m_request;
};

class CLIPTextModel : public IAdaptedModel {
public:
    struct Config {
        // TODO: is it better to use tokenizer max length?
        size_t max_position_embeddings = 77;
        size_t projection_dim = 512;

        explicit Config(const std::string& config_path) {
            std::ifstream file(config_path);
            OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

            nlohmann::json data = nlohmann::json::parse(file);
            using ov::genai::utils::read_json_param;

            read_json_param(data, "max_position_embeddings", max_position_embeddings);
            read_json_param(data, "projection_dim", projection_dim);
        }
    };

    explicit CLIPTextModel(const std::string root_dir) :
        IAdaptedModel(root_dir + "/text_encoder/openvino_model.xml"),
        m_clip_tokenizer(root_dir + "/tokenizer"),
        m_config(root_dir + "/text_encoder/config.json") {
    }

    CLIPTextModel(const std::string& root_dir,
                  const std::string& device,
                  const ov::AnyMap& properties) :
        CLIPTextModel(root_dir) {
        compile(device, properties);
    }

    const Config& get_config() const {
        return m_config;
    }

    void reshape(size_t batch_size) {
        ov::PartialShape input_shape = m_model->input(0).get_partial_shape();
        input_shape[0] = batch_size;
        input_shape[1] = m_config.max_position_embeddings;
        std::map<size_t, ov::PartialShape> idx_to_shape{{0, input_shape}};
        m_model->reshape(idx_to_shape);
    }

    ov::Tensor forward(const std::string& pos_prompt, const std::string& neg_prompt, bool do_classifier_free_guidance) {
        const int32_t pad_token_id = m_clip_tokenizer.get_pad_token_id();
        const size_t text_embedding_batch_size = do_classifier_free_guidance ? 2 : 1;

        auto compute_text_embeddings = [&](const std::string& prompt, ov::Tensor encoder_output_tensor) {
            ov::Tensor input_ids(ov::element::i32, {1, m_config.max_position_embeddings});
            std::fill_n(input_ids.data<int32_t>(), input_ids.get_size(), pad_token_id);

            // tokenization
            ov::Tensor input_ids_token = m_clip_tokenizer.encode(prompt).input_ids;
            std::copy_n(input_ids_token.data<std::int64_t>(), input_ids_token.get_size(), input_ids.data<std::int32_t>());

            // text embeddings
            m_request.set_tensor("input_ids", input_ids);
            m_request.set_output_tensor(0, encoder_output_tensor);
            m_request.infer();
        };

        ov::Tensor text_embeddings(ov::element::f32, {text_embedding_batch_size, m_config.max_position_embeddings, m_config.projection_dim});

        size_t current_batch_idx = 0;
        if (do_classifier_free_guidance) {
            compute_text_embeddings(neg_prompt,
                                    ov::Tensor(text_embeddings, {current_batch_idx, 0, 0},
                                                                {current_batch_idx + 1, m_config.max_position_embeddings, m_config.projection_dim}));
            ++current_batch_idx;
        } else {
            // Negative prompt is ignored when --guidanceScale < 1.0
        }

        compute_text_embeddings(pos_prompt,
                                ov::Tensor(text_embeddings, {current_batch_idx, 0, 0},
                                                            {current_batch_idx + 1, m_config.max_position_embeddings, m_config.projection_dim}));

        return text_embeddings;
    }

private:
    ov::genai::Tokenizer m_clip_tokenizer;
    Config m_config;
};

class UNet2DConditionModel : public IAdaptedModel {
public:
    struct Config {
        size_t in_channels = 4;
        size_t sample_size = 0;
        std::vector<size_t> block_out_channels = { 320, 640, 1280, 1280 };
        int time_cond_proj_dim = -1;

        explicit Config(const std::string& config_path) {
            std::ifstream file(config_path);
            OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

            nlohmann::json data = nlohmann::json::parse(file);
            using ov::genai::utils::read_json_param;

            read_json_param(data, "in_channels", in_channels);
            read_json_param(data, "sample_size", sample_size);
            read_json_param(data, "block_out_channels", block_out_channels);
            read_json_param(data, "time_cond_proj_dim", time_cond_proj_dim);
        }
    };

    explicit UNet2DConditionModel(const std::string root_dir) :
        IAdaptedModel(root_dir + "/unet/openvino_model.xml"),
        m_config(root_dir + "/unet/config.json") {
        // compute VAE scale factor
        m_vae_scale_factor = std::pow(2, m_config.block_out_channels.size() - 1);
    }

    UNet2DConditionModel(const std::string& root_dir,
                         const std::string& device,
                         const ov::AnyMap& properties) :
        UNet2DConditionModel(root_dir) {
        compile(device, properties);
    }

    const Config& get_config() const {
        return m_config;
    }

    size_t get_vae_scale_factor() const {
        return m_vae_scale_factor;
    }

    void set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states) {
        OPENVINO_ASSERT(m_request, "UNet model must be compiled first");
        m_request.set_tensor(tensor_name, encoder_hidden_states);
    }

    ov::Tensor forward(ov::Tensor sample, ov::Tensor timestep) {
        OPENVINO_ASSERT(m_request, "UNet model must be compiled first");

        m_request.set_tensor("sample", sample);
        m_request.set_tensor("timestep", timestep);

        m_request.infer();

        return m_request.get_output_tensor();
    }

    void reshape(int64_t batch_size, int64_t height, int64_t width, int64_t tokenizer_model_max_length) {
        OPENVINO_ASSERT(m_model, "Model has been already compiled");

        // TODO: what if it's disabled?
        // The factor of 2 comes from the guidance scale > 1
        batch_size *= m_config.time_cond_proj_dim <= 0 ? 2 : 1;

        height /= m_vae_scale_factor;
        width /= m_vae_scale_factor;

        std::map<std::string, ov::PartialShape> name_to_shape;

        for (auto && input : m_model->inputs()) {
            std::string input_name = input.get_any_name();
            name_to_shape[input_name] = input.get_partial_shape();
            if (input_name == "timestep") {
                name_to_shape[input_name][0] = 1;
            } else if (input_name == "sample") {
                name_to_shape[input_name] = {batch_size, name_to_shape[input_name][1], height, width};
            } else if (input_name == "time_ids") {
                name_to_shape[input_name][0] = batch_size;
            } else if (input_name == "encoder_hidden_states") {
                name_to_shape[input_name][0] = batch_size;
                name_to_shape[input_name][1] = tokenizer_model_max_length;
            }
        }

        m_model->reshape(name_to_shape);
    }

private:
    Config m_config;
    size_t m_vae_scale_factor;
};

class AutoencoderKL {
public:
    struct Config {
        size_t in_channels = 3;
        size_t latent_channels = 4;
        size_t out_channels = 3;
        float scaling_factor = 0.18215f;
        std::vector<size_t> block_out_channels = { 64 };

        explicit Config(const std::string& config_path) {
            std::ifstream file(config_path);
            OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

            nlohmann::json data = nlohmann::json::parse(file);
            using ov::genai::utils::read_json_param;

            read_json_param(data, "in_channels", in_channels);
            read_json_param(data, "latent_channels", latent_channels);
            read_json_param(data, "out_channels", out_channels);
            read_json_param(data, "scaling_factor", scaling_factor);
            read_json_param(data, "block_out_channels", block_out_channels);
        }
    };

    explicit AutoencoderKL(const std::string& root_dir)
        : m_config(root_dir + "/vae_decoder/config.json") {
        m_model = ov::Core().read_model(root_dir + "/vae_decoder/openvino_model.xml");
        // apply VaeImageProcessor postprocessing steps by merging them into the VAE decoder model
        merge_vae_image_processor();
    }

    AutoencoderKL(const std::string& root_dir,
                  const std::string& device,
                  const ov::AnyMap& properties = {})
        : AutoencoderKL(root_dir) {
        compile(device, properties);
    }

    ov::Tensor forward(ov::Tensor latent) {
        OPENVINO_ASSERT(m_request, "VAE decoder model must be compiled first");

        m_request.set_input_tensor(latent);
        m_request.infer();
        return m_request.get_output_tensor();
    }

    void compile(const std::string& device, const ov::AnyMap& properties = {}) {
        OPENVINO_ASSERT(m_model, "Model has been already compiled");
        ov::CompiledModel compiled_model = ov::Core().compile_model(m_model, device, properties);
        m_request = compiled_model.create_infer_request();
        // release the original model
        m_model.reset();
    }

    void reshape(int64_t height, int64_t width) {
        OPENVINO_ASSERT(m_model, "Model has been already compiled");

        const size_t vae_scale_factor = std::pow(2, m_config.block_out_channels.size() - 1);
        const size_t batch_size = 1;

        height /= vae_scale_factor;
        width /= vae_scale_factor;

        ov::PartialShape input_shape = m_model->input(0).get_partial_shape();
        std::map<size_t, ov::PartialShape> idx_to_shape{{0, {batch_size, input_shape[1], height, width}}};
        m_model->reshape(idx_to_shape);
    }

private:
    void merge_vae_image_processor() const {
        ov::preprocess::PrePostProcessor ppp(m_model);

        // scale input before VAE encoder
        ppp.input().preprocess().scale(m_config.scaling_factor);

        // apply VaeImageProcessor normalization steps
        // https://github.com/huggingface/diffusers/blob/v0.30.1/src/diffusers/image_processor.py#L159
        ppp.output().postprocess().custom([](const ov::Output<ov::Node>& port) {
            auto constant_0_5 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 0.5f);
            auto constant_255 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 255.0f);
            auto scaled_0_5 = std::make_shared<ov::op::v1::Multiply>(port, constant_0_5);
            auto added_0_5 = std::make_shared<ov::op::v1::Add>(scaled_0_5, constant_0_5);
            auto clamped = std::make_shared<ov::op::v0::Clamp>(added_0_5, 0.0f, 1.0f);
            return std::make_shared<ov::op::v1::Multiply>(clamped, constant_255);
        });
        ppp.output().postprocess().convert_element_type(ov::element::u8);
        // layout conversion
        // https://github.com/huggingface/diffusers/blob/v0.30.1/src/diffusers/image_processor.py#L144
        ppp.output().model().set_layout("NCHW");
        ppp.output().tensor().set_layout("NHWC");

        ppp.build();
    }

    Config m_config;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
};

class StableDiffusionPipeline {
public:
    StableDiffusionPipeline(const std::string& root_dir) {
        const std::string model_index_path = root_dir + "/model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using ov::genai::utils::read_json_param;

        m_scheduler = Scheduler::from_config(root_dir + "/scheduler/scheduler_config.json");
        m_clip_text_encoder = std::make_shared<CLIPTextModel>(root_dir);
        m_unet = std::make_shared<UNet2DConditionModel>(root_dir);
        m_vae_decoder = std::make_shared<AutoencoderKL>(root_dir);
    }

    void reshape(const size_t batch_size, const size_t height, const size_t width) {
        m_clip_text_encoder->reshape(batch_size);
        m_unet->reshape(batch_size, height, width, m_clip_text_encoder->get_config().max_position_embeddings);
        m_vae_decoder->reshape(height, width);
    }

    void compile(const std::string& device, const ov::AnyMap& properties = {}) {
        m_clip_text_encoder->compile(device, properties);
        m_unet->compile(device, properties);
        m_vae_decoder->compile(device, properties);
    }

    void apply_lora(const std::string& lora_path, float alpha) {
        Timer t("Loading and multiplying LoRA weights");
        std::map<std::string, InsertLoRA::LoRAMap> lora_weights = read_lora_adapters(lora_path, alpha);

        m_clip_text_encoder->apply_lora(lora_weights["text_encoder"]);
        m_unet->apply_lora(lora_weights["unet"]);
    }

    ov::Tensor generate(const std::string& positive_prompt,
                        const std::string& negative_prompt /* can be empty */,
                        float guidance_scale = 7.5f,
                        int64_t height = -1,
                        int64_t width = -1,
                        size_t num_inference_steps = 50,
                        size_t num_images_per_prompt = 1) {
        Timer t("Running Stable Diffusion pipeline");
        OPENVINO_ASSERT(num_images_per_prompt == 1, "Currently only num_images_per_prompt = 1 is supported");

        // Stable Diffusion pipeline
        // see https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline#deconstruct-the-stable-diffusion-pipeline

        const size_t batch_size = num_images_per_prompt;
        const auto& unet_config = m_unet->get_config();
        const bool do_classifier_free_guidance = guidance_scale > 1.0 && unet_config.time_cond_proj_dim < 0;
        const size_t batch_size_multiplier = do_classifier_free_guidance ? 2 : 1;  // Unet accepts 2x batch in case of CFG
        const size_t vae_scale_factor = m_unet->get_vae_scale_factor();

        // TODO: drop these variables
        const bool read_np_latent = false;
        const size_t user_seed = 42;

        if (height < 0)
            height = unet_config.sample_size * vae_scale_factor;
        if (width < 0)
            width = unet_config.sample_size * vae_scale_factor;

        ov::Tensor encoder_hidden_states = m_clip_text_encoder->forward(positive_prompt, negative_prompt, do_classifier_free_guidance);
        m_unet->set_hidden_states("encoder_hidden_states", encoder_hidden_states);

        if (unet_config.time_cond_proj_dim >= 0) {
            ov::Tensor guidance_scale_embedding = get_guidance_scale_embedding(guidance_scale, unet_config.time_cond_proj_dim);
            m_unet->set_hidden_states("timestep_cond", guidance_scale_embedding);
        }

        m_scheduler->set_timesteps(num_inference_steps);
        std::vector<std::int64_t> timesteps = m_scheduler->get_timesteps();

        ov::Tensor denoised;
        for (uint32_t n = 0; n < num_images_per_prompt; n++) {
            std::uint32_t seed = user_seed + n;

            // latents are multiplied by 'init_noise_sigma'
            ov::Shape latent_shape = ov::Shape{batch_size, unet_config.in_channels, height / vae_scale_factor, width / vae_scale_factor};
            ov::Shape latent_shape_cfg = latent_shape;
            ov::Tensor noise = randn_tensor(latent_shape, read_np_latent, seed);
            latent_shape_cfg[0] *= batch_size_multiplier;

            ov::Tensor latent(ov::element::f32, latent_shape), latent_cfg(ov::element::f32, latent_shape_cfg);
            for (size_t i = 0; i < noise.get_size(); ++i) {
                latent.data<float>()[i] = noise.data<float>()[i] * m_scheduler->get_init_noise_sigma();
            }

            for (size_t inference_step = 0; inference_step < num_inference_steps; inference_step++) {
                // concat the same latent twice along a batch dimension in case of CFG
                latent.copy_to(
                    ov::Tensor(latent_cfg, {0, 0, 0, 0}, {1, latent_shape[1], latent_shape[2], latent_shape[3]}));
                if (do_classifier_free_guidance) {
                    latent.copy_to(
                        ov::Tensor(latent_cfg, {1, 0, 0, 0}, {2, latent_shape[1], latent_shape[2], latent_shape[3]}));
                }

                m_scheduler->scale_model_input(latent_cfg, inference_step);

                ov::Tensor timestep(ov::element::i64, {1}, &timesteps[inference_step]);
                ov::Tensor noise_pred_tensor = m_unet->forward(latent_cfg, timestep);

                ov::Shape noise_pred_shape = noise_pred_tensor.get_shape();
                noise_pred_shape[0] = batch_size;

                ov::Tensor noisy_residual(noise_pred_tensor.get_element_type(), noise_pred_shape);

                if (do_classifier_free_guidance) {
                    // perform guidance
                    const float* noise_pred_uncond = noise_pred_tensor.data<const float>();
                    const float* noise_pred_text = noise_pred_uncond + ov::shape_size(noise_pred_shape);

                    for (size_t i = 0; i < ov::shape_size(noise_pred_shape); ++i) {
                        noisy_residual.data<float>()[i] =
                            noise_pred_uncond[i] + guidance_scale * (noise_pred_text[i] - noise_pred_uncond[i]);
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
    static ov::Tensor get_guidance_scale_embedding(float guidance_scale, uint32_t embedding_dim) {
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

    std::shared_ptr<Scheduler> m_scheduler;
    std::shared_ptr<CLIPTextModel> m_clip_text_encoder;
    std::shared_ptr<UNet2DConditionModel> m_unet;
    std::shared_ptr<AutoencoderKL> m_vae_decoder;
};

int32_t main(int32_t argc, char* argv[]) try {
    cxxopts::Options options("stable_diffusion", "Stable Diffusion implementation in C++ using OpenVINO\n");

    options.add_options()
    ("p,posPrompt", "Initial positive prompt for SD", cxxopts::value<std::string>()->default_value("cyberpunk cityscape like Tokyo New York  with tall buildings at dusk golden hour cinematic lighting"))
    ("n,negPrompt", "The prompt to guide the image generation away from. Ignored when not using guidance (`--guidanceScale` is less than `1`)", cxxopts::value<std::string>()->default_value(""))
    ("d,device", "AUTO, CPU, or GPU.\nDoesn't apply to Tokenizer model, OpenVINO Tokenizers can be inferred on a CPU device only", cxxopts::value<std::string>()->default_value("CPU"))
    ("step", "Number of diffusion steps", cxxopts::value<size_t>()->default_value("20"))
    ("s,seed", "Number of random seed to generate latent for one image output", cxxopts::value<size_t>()->default_value("42"))
    ("guidanceScale", "A higher guidance scale value encourages the model to generate images closely linked to the text prompt at the expense of lower image quality", cxxopts::value<float>()->default_value("7.5"))
    ("num", "Number of image output", cxxopts::value<size_t>()->default_value("1"))
    ("height", "Destination image height", cxxopts::value<size_t>()->default_value("512"))
    ("width", "Destination image width", cxxopts::value<size_t>()->default_value("512"))
    ("c,useCache", "Use model caching", cxxopts::value<bool>()->default_value("false"))
    ("r,readNPLatent", "Read numpy generated latents from file", cxxopts::value<bool>()->default_value("false"))
    ("m,modelPath", "Specify path of SD model IRs", cxxopts::value<std::string>()->default_value("./models/dreamlike_anime_1_0_ov"))
    ("dynamic", "Specify the model input shape to use dynamic shape", cxxopts::value<bool>()->default_value("false"))
    ("l,loraPath", "Specify path of LoRA file. (*.safetensors).", cxxopts::value<std::string>()->default_value(""))
    ("a,alpha", "alpha for LoRA", cxxopts::value<float>()->default_value("0.75"))("h,help", "Print usage");
    cxxopts::ParseResult result;

    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& e) {
        std::cout << e.what() << "\n\n";
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    std::string positive_prompt = result["posPrompt"].as<std::string>();
    std::string negative_prompt = result["negPrompt"].as<std::string>();
    const std::string device = result["device"].as<std::string>();
    const uint32_t num_inference_steps = result["step"].as<size_t>();
    const uint32_t user_seed = result["seed"].as<size_t>();
    const float guidance_scale = result["guidanceScale"].as<float>();
    const uint32_t num_images_per_prompt = result["num"].as<size_t>();
    const uint32_t height = result["height"].as<size_t>();
    const uint32_t width = result["width"].as<size_t>();
    const bool use_cache = result["useCache"].as<bool>();
    const bool read_np_latent = result["readNPLatent"].as<bool>();
    const std::string models_path = result["modelPath"].as<std::string>();
    const bool use_dynamic_shapes = result["dynamic"].as<bool>();
    const std::string lora_path = result["loraPath"].as<std::string>();
    const float alpha = result["alpha"].as<float>();

    OPENVINO_ASSERT(
        !read_np_latent || (read_np_latent && (num_images_per_prompt == 1)),
        "\"readNPLatent\" option is only supported for one output image. Number of image output was set to " +
            std::to_string(num_images_per_prompt));

    const std::string folder_name = "images";
    try {
        std::filesystem::create_directory(folder_name);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create dir" << e.what() << std::endl;
    }

    std::cout << "OpenVINO version: " << ov::get_openvino_version() << std::endl;

    ov::AnyMap properties;
    if (use_cache)
        properties.insert(ov::cache_dir("./cache_dir"));

    StableDiffusionPipeline pipe(models_path);
    if (!use_dynamic_shapes)
        pipe.reshape(num_images_per_prompt, height, width);
    pipe.compile(device, properties);

    {
        ov::Tensor generated_images = pipe.generate(positive_prompt, negative_prompt, guidance_scale,
            height, width, num_inference_steps, num_images_per_prompt);

        for (size_t n = 0; n < num_images_per_prompt; ++n) {
            ov::Tensor generated_image(generated_images, { n, 0, 0, 0 }, { n + 1, height, width, 3 });
            std::string result_image_path = "./images/seed_" + std::to_string(n) + ".bmp";
            imwrite(result_image_path, generated_image, true);
        }
    }

    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
