// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <fstream>

#include "iadapted_model.hpp"

#include "utils.hpp"

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

    void reshape(int batch_size, int height, int width, int tokenizer_model_max_length) {
        OPENVINO_ASSERT(m_model, "Model has been already compiled");

        height /= m_vae_scale_factor;
        width /= m_vae_scale_factor;

        std::map<std::string, ov::PartialShape> name_to_shape;

        for (auto && input : m_model->inputs()) {
            std::string input_name = input.get_any_name();
            name_to_shape[input_name] = input.get_partial_shape();
            if (input_name == "timestep") {
                name_to_shape[input_name][0] = 1;
            } else if (input_name == "sample") {
                name_to_shape[input_name] = {1, name_to_shape[input_name][1], height, width};
            } else if (input_name == "time_ids") {
                name_to_shape[input_name][0] = 1;
            } else if (input_name == "encoder_hidden_states") {
                name_to_shape[input_name][0] = 1;
                name_to_shape[input_name][1] = tokenizer_model_max_length;
            }
        }

        m_model->reshape(name_to_shape);
    }

private:
    Config m_config;
    size_t m_vae_scale_factor;
};
