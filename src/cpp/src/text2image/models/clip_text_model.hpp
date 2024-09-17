// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "openvino/genai/visibility.hpp"
#include "openvino/genai/tokenizer.hpp"

#include "openvino/core/any.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"

namespace ov {
namespace genai {

class CLIPTextModel {
public:
    struct Config {
        size_t max_position_embeddings = 77;
        size_t hidden_size = 512;

        explicit Config(const std::string& config_path);
    };

    explicit CLIPTextModel(const std::string root_dir);

    CLIPTextModel(const std::string& root_dir,
                  const std::string& device,
                  const ov::AnyMap& properties = {});

    CLIPTextModel(const CLIPTextModel&);

    const Config& get_config() const;

    void reshape(int batch_size);

    void compile(const std::string& device, const ov::AnyMap& properties = {});

    ov::Tensor forward(const std::string& pos_prompt, const std::string& neg_prompt, bool do_classifier_free_guidance);

private:
    Config m_config;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;

    Tokenizer m_clip_tokenizer;
};

} // namespace genai
} // namespace ov
