// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fstream>

#include "iadapted_model.hpp"

#include "openvino/genai/tokenizer.hpp"

#include "utils.hpp"

class CLIPTextModel : public IAdaptedModel {
public:
    struct Config {
        // TODO: is it better to use tokenizer max length?
        size_t max_position_embeddings = 77;
        size_t hidden_size = 512;

        explicit Config(const std::string& config_path) {
            std::ifstream file(config_path);
            OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

            nlohmann::json data = nlohmann::json::parse(file);
            using ov::genai::utils::read_json_param;

            read_json_param(data, "max_position_embeddings", max_position_embeddings);
            read_json_param(data, "hidden_size", hidden_size);
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

    void reshape(int batch_size) {
        ov::PartialShape input_shape = m_model->input(0).get_partial_shape();
        input_shape[0] = batch_size;
        input_shape[1] = m_config.max_position_embeddings;
        std::map<size_t, ov::PartialShape> idx_to_shape{{0, input_shape}};
        m_model->reshape(idx_to_shape);
    }

    ov::Tensor forward(const std::string& pos_prompt, const std::string& neg_prompt, bool do_classifier_free_guidance) {
        const int32_t pad_token_id = m_clip_tokenizer.get_pad_token_id();
        const size_t text_embedding_batch_size = do_classifier_free_guidance ? 2 : 1;

        auto perform_tokenization = [&](const std::string& prompt, ov::Tensor input_ids) {
            std::fill_n(input_ids.data<int32_t>(), input_ids.get_size(), pad_token_id);

            ov::Tensor input_ids_token = m_clip_tokenizer.encode(prompt).input_ids;
            std::copy_n(input_ids_token.data<std::int64_t>(), input_ids_token.get_size(), input_ids.data<std::int32_t>());
        };

        ov::Tensor input_ids(ov::element::i32, {text_embedding_batch_size, m_config.max_position_embeddings});
        size_t current_batch_idx = 0;

        if (do_classifier_free_guidance) {
            perform_tokenization(neg_prompt,
                                 ov::Tensor(input_ids, {current_batch_idx    , 0},
                                                       {current_batch_idx + 1, m_config.max_position_embeddings}));
            ++current_batch_idx;
        } else {
            // Negative prompt is ignored when --guidanceScale < 1.0
        }

        perform_tokenization(pos_prompt,
                             ov::Tensor(input_ids, {current_batch_idx    , 0},
                                                   {current_batch_idx + 1, m_config.max_position_embeddings}));

        // text embeddings
        m_request.set_tensor("input_ids", input_ids);
        m_request.infer();

        return m_request.get_output_tensor(0);
    }

private:
    ov::genai::Tokenizer m_clip_tokenizer;
    Config m_config;
};
