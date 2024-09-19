// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/tokenizer.hpp"
#include "openvino/runtime/compiled_model.hpp"

namespace ov {
namespace genai {

/**
 * @brief Structure to keep whisper generation config parameters.
 */
class OPENVINO_GENAI_EXPORTS WhisperGenerationConfig {
public:
    WhisperGenerationConfig() = default;
    explicit WhisperGenerationConfig(const std::string& json_path);

    // Generic

    // the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
    // `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
    size_t max_new_tokens = SIZE_MAX;
    // the maximum numbers of tokens to generate, excluding the number of tokens in the prompt.
    // max_new_tokens has priority over max_length.
    size_t max_length = SIZE_MAX;

    // Whisper specific

    // Corresponds to the ”<|startoftranscript|>” token.
    int64_t decoder_start_token_id = 50258;

    // The language token id of the transcription text. It is appended to the start of the sequence for
    // multilingual speech recognition, e.g. for Spanish the token id corresponding to the "<|es|>" is appended to the
    // start of sequence.
    int64_t language_token_id = 50259;

    // End of stream token id.
    int64_t eos_token_id = 50257;

    // Padding token id.
    int64_t pad_token_id = 50257;

    // Translate token id.
    int64_t translate_token_id = 50358;

    // Transcribe token id.
    int64_t transcribe_token_id = 50359;

    // No timestamps token id.
    int64_t no_timestamps_token_id = 50363;

    // Begin timestamps token id.
    int64_t begin_timestamps_token_id = 50364;

    bool is_multilingual = true;

    // A list containing tokens that will be supressed at the beginning of the sampling process.
    std::vector<int64_t> begin_suppress_tokens;

    // A list containing the non-speech tokens that will be supressed during generation.
    std::vector<int64_t> suppress_tokens;

    /** @brief sets eos_token_id to tokenizer_eos_token_id if eos_token_id is less than 0.
     * Otherwise verifies eos_token_id == tokenizer_eos_token_id.
     */
    void set_eos_token_id(int64_t tokenizer_eos_token_id);
    size_t get_max_new_tokens(size_t prompt_length = 0) const;

    void update_generation_config(const ov::AnyMap& config_map = {});

    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> update_generation_config(Properties&&... properties) {
        return update_generation_config(AnyMap{std::forward<Properties>(properties)...});
    }

    /// @brief checks that are no conflicting parameters.
    /// @throws Exception if config is invalid.
    void validate() const;
};

/*
 * utils that allow to use generate and operator() in the following way:
 * pipe.generate(input_ids, ov::genai::max_new_tokens(200),...)
 * pipe(text, ov::genai::max_new_tokens(200),...)
 */

static constexpr ov::Property<std::vector<int64_t>> begin_suppress_tokens{"begin_suppress_tokens"};
static constexpr ov::Property<std::vector<int64_t>> suppress_tokens{"suppress_tokens"};
static constexpr ov::Property<int64_t> decoder_start_token_id{"decoder_start_token_id"};

static constexpr ov::Property<int64_t> pad_token_id{"pad_token_id"};
static constexpr ov::Property<int64_t> transcribe_token_id{"transcribe_token_id"};
static constexpr ov::Property<int64_t> translate_token_id{"translate_token_id"};
static constexpr ov::Property<int64_t> no_timestamps_token_id{"no_timestamps_token_id"};
static constexpr ov::Property<int64_t> begin_timestamps_token_id{"begin_timestamps_token_id"};

}  // namespace genai
}  // namespace ov
