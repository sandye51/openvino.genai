// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <iostream>
#include <string>

#include "cxxopts.hpp"
#include "imwrite.hpp"

#include "openvino/core/version.hpp"
#include "openvino/runtime/properties.hpp"

#include "diffusers/stable_diffusion_pipeline.hpp"

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

    // by default DDIM is used, let's override to LMSDiscreteScheduler
    pipe.set_scheduler(Scheduler::from_config(models_path + "/scheduler/scheduler_config.json", SchedulerType::LMS_DISCRETE));

    ov::Tensor generated_images = pipe.generate(positive_prompt, negative_prompt, guidance_scale,
        height, width, num_inference_steps, num_images_per_prompt);

    for (size_t n = 0; n < num_images_per_prompt; ++n) {
        ov::Tensor generated_image(generated_images, { n, 0, 0, 0 }, { n + 1, height, width, 3 });
        std::string result_image_path = "./images/seed_" + std::to_string(n) + ".bmp";
        imwrite(result_image_path, generated_image, true);
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
