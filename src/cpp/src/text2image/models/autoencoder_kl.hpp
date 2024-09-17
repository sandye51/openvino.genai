// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"

#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {

class AutoencoderKL {
public:
    struct Config {
        size_t in_channels = 3;
        size_t latent_channels = 4;
        size_t out_channels = 3;
        float scaling_factor = 0.18215f;
        std::vector<size_t> block_out_channels = { 64 };

        explicit Config(const std::string& config_path);
    };

    explicit AutoencoderKL(const std::string& root_dir);

    AutoencoderKL(const std::string& root_dir,
                  const std::string& device,
                  const ov::AnyMap& properties = {});

    AutoencoderKL(const AutoencoderKL&);

    void reshape(int batch_size, int height, int width);

    void compile(const std::string& device, const ov::AnyMap& properties = {});

    ov::Tensor forward(ov::Tensor latent);

private:
    void merge_vae_image_processor() const;

    Config m_config;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
};

} // namespace genai
} // namespace ov
