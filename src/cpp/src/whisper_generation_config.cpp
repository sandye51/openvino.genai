// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/whisper_generation_config.hpp"

#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <openvino/runtime/core.hpp>

#include "utils.hpp"

namespace ov {
namespace genai {

WhisperGenerationConfig::WhisperGenerationConfig(const std::string& json_path) {
    using ov::genai::utils::read_json_param;

    std::ifstream f(json_path);
    OPENVINO_ASSERT(f.is_open(), "Failed to open '" + json_path + "' with generation config");

    nlohmann::json data = nlohmann::json::parse(f);

    read_json_param(data, "max_new_tokens", max_new_tokens);
    read_json_param(data, "max_length", max_length);
    read_json_param(data, "begin_suppress_tokens", begin_suppress_tokens);
    read_json_param(data, "suppress_tokens", suppress_tokens);
    read_json_param(data, "decoder_start_token_id", decoder_start_token_id);
    read_json_param(data, "eos_token_id", eos_token_id);
    read_json_param(data, "pad_token_id", pad_token_id);
    read_json_param(data, "no_timestamps_token_id", no_timestamps_token_id);
    read_json_param(data, "begin_timestamps_token_id", begin_timestamps_token_id);

    read_json_param(data, "is_multilingual", is_multilingual);
    if (is_multilingual) {
        read_json_param(data, "task_to_id.transcribe", transcribe_token_id);
        read_json_param(data, "task_to_id.translate", translate_token_id);
    }
}

void WhisperGenerationConfig::set_eos_token_id(int64_t tokenizer_eos_token_id) {
    if (eos_token_id < 0) {
        eos_token_id = tokenizer_eos_token_id;
    } else {
        OPENVINO_ASSERT(eos_token_id == tokenizer_eos_token_id,
                        "EOS token ID is different in generation config (",
                        eos_token_id,
                        ") and tokenizer (",
                        tokenizer_eos_token_id,
                        ")");
    }
}

void WhisperGenerationConfig::update_generation_config(const ov::AnyMap& config_map) {
    using ov::genai::utils::read_anymap_param;

    read_anymap_param(config_map, "max_new_tokens", max_new_tokens);
    read_anymap_param(config_map, "max_length", max_length);
    read_anymap_param(config_map, "begin_suppress_tokens", begin_suppress_tokens);
    read_anymap_param(config_map, "suppress_tokens", suppress_tokens);
    read_anymap_param(config_map, "decoder_start_token_id", decoder_start_token_id);
    read_anymap_param(config_map, "eos_token_id", eos_token_id);
    read_anymap_param(config_map, "pad_token_id", pad_token_id);
    read_anymap_param(config_map, "transcribe_token_id", transcribe_token_id);
    read_anymap_param(config_map, "translate_token_id", translate_token_id);
    read_anymap_param(config_map, "no_timestamps_token_id", no_timestamps_token_id);
    read_anymap_param(config_map, "begin_timestamps_token_id", begin_timestamps_token_id);
    read_anymap_param(config_map, "is_multilingual", is_multilingual);
}

size_t WhisperGenerationConfig::get_max_new_tokens(size_t prompt_length) const {
    // max_new_tokens has priority over max_length, only if max_new_tokens was not specified use max_length
    if (max_new_tokens != SIZE_MAX) {
        return max_new_tokens;
    } else {
        return max_length - prompt_length;
    }
}

void WhisperGenerationConfig::validate() const {
    OPENVINO_ASSERT(max_new_tokens > 0, "'max_new_tokens' must be greater than 0");

    // max_new_tokens has priority over max_length
    // if max_new_tokens is defined no need to check max_length
    OPENVINO_ASSERT(max_new_tokens != SIZE_MAX || max_length > 0,
                    "'max_length' must be greater than 0 or 'max_new_tokens' should be defined");

    OPENVINO_ASSERT(eos_token_id != -1 || max_new_tokens != SIZE_MAX || max_length != SIZE_MAX,
                    "Either 'eos_token_id', or 'max_new_tokens', or 'max_length' should be defined.");
}
}  // namespace genai
}  // namespace ov
