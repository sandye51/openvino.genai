// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <fstream>

#include "utils.hpp"

#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/constant.hpp"

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
        : m_config(root_dir + "/config.json") {
        m_model = ov::Core().read_model(root_dir + "/openvino_model.xml");
        // apply VaeImageProcessor postprocessing steps by merging them into the VAE decoder model
        merge_vae_image_processor();
    }

    AutoencoderKL(const std::string& root_dir,
                  const std::string& device,
                  const ov::AnyMap& properties = {})
        : AutoencoderKL(root_dir) {
        compile(device, properties);
    }

    AutoencoderKL(const AutoencoderKL&) = default;

    void reshape(int batch_size, int height, int width) {
        OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

        const size_t vae_scale_factor = std::pow(2, m_config.block_out_channels.size() - 1);

        height /= vae_scale_factor;
        width /= vae_scale_factor;

        ov::PartialShape input_shape = m_model->input(0).get_partial_shape();
        std::map<size_t, ov::PartialShape> idx_to_shape{{0, {batch_size, input_shape[1], height, width}}};
        m_model->reshape(idx_to_shape);
    }

    void compile(const std::string& device, const ov::AnyMap& properties = {}) {
        OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
        ov::CompiledModel compiled_model = ov::Core().compile_model(m_model, device, properties);
        m_request = compiled_model.create_infer_request();
        // release the original model
        m_model.reset();
    }

    ov::Tensor forward(ov::Tensor latent) {
        OPENVINO_ASSERT(m_request, "VAE decoder model must be compiled first. Cannot infer non-compiled model");

        m_request.set_input_tensor(latent);
        m_request.infer();
        return m_request.get_output_tensor();
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
