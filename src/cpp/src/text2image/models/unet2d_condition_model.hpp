// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <string>
#include <memory>

#include "openvino/genai/visibility.hpp"

#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"

namespace ov {
namespace genai {

class UNet2DConditionModel {
public:
    struct Config {
        size_t in_channels = 4;
        size_t sample_size = 0;
        std::vector<size_t> block_out_channels = { 320, 640, 1280, 1280 };
        int time_cond_proj_dim = -1;

        explicit Config(const std::string& config_path);
    };

    explicit UNet2DConditionModel(const std::string root_dir);

    UNet2DConditionModel(const std::string& root_dir,
                         const std::string& device,
                         const ov::AnyMap& properties = {});

    UNet2DConditionModel(const UNet2DConditionModel&);

    const Config& get_config() const;

    size_t get_vae_scale_factor() const;

    void reshape(int batch_size, int height, int width, int tokenizer_model_max_length);

    void compile(const std::string& device, const ov::AnyMap& properties = {});

    void set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states);

    ov::Tensor forward(ov::Tensor sample, ov::Tensor timestep);

private:
    Config m_config;
    std::shared_ptr<ov::Model> m_model;
    ov::InferRequest m_request;
    size_t m_vae_scale_factor;
};

} // namespace genai
} // namespace ov
