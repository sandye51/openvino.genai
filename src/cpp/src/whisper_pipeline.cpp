// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/whisper_pipeline.hpp"

#include <algorithm>
#include <filesystem>
#include <openvino/openvino.hpp>
#include <variant>

#include "text_callback_streamer.hpp"
#include "utils.hpp"
#include "whisper/whisper_feature_extractor.hpp"
#include "whisper/whisper_models.hpp"

namespace {
ov::genai::WhisperGenerationConfig from_config_json_if_exists(const std::filesystem::path& model_path) {
    auto config_file_path = model_path / "generation_config.json";
    if (std::filesystem::exists(config_file_path)) {
        return ov::genai::WhisperGenerationConfig((config_file_path).string());
    } else {
        return ov::genai::WhisperGenerationConfig{};
    }
}

ov::genai::OptionalWhisperGenerationConfig get_config_from_map(const ov::AnyMap& config_map) {
    if (config_map.count("generation_config")) {
        return config_map.at("generation_config").as<ov::genai::WhisperGenerationConfig>();
    } else {
        return std::nullopt;
    }
}
}  // namespace

namespace ov {
namespace genai {

std::vector<int64_t> whisper_generate(const ov::genai::WhisperGenerationConfig& config,
                                      const RawSpeechInput& raw_speech,
                                      ov::genai::WhisperInitializedModels& models,
                                      ov::genai::WhisperFeatureExtractor& feature_extractor,
                                      const std::shared_ptr<StreamerBase> streamer);

class WhisperPipeline::Impl {
public:
    ov::genai::WhisperGenerationConfig m_generation_config;
    ov::genai::WhisperInitializedModels m_models;
    ov::genai::WhisperFeatureExtractor m_feature_extractor;
    Tokenizer m_tokenizer;
    float m_load_time_ms = 0;

    Impl(const std::filesystem::path& model_path,
         const ov::genai::Tokenizer& tokenizer,
         const std::string& device,
         const ov::AnyMap& plugin_config)
        : m_generation_config{from_config_json_if_exists(model_path)},
          m_tokenizer{tokenizer},
          m_feature_extractor{(model_path / "preprocessor_config.json").string()} {
        ov::Core core;
        core.set_property(device, plugin_config);

        m_models.encoder = core.compile_model(model_path / "openvino_encoder_model.xml", device).create_infer_request();
        m_models.decoder = core.compile_model(model_path / "openvino_decoder_model.xml", device).create_infer_request();
        m_models.decoder_with_past =
            core.compile_model(model_path / "openvino_decoder_with_past_model.xml", device).create_infer_request();

        // If eos_token_id was not provided, take value
        if (m_generation_config.eos_token_id == -1) {
            m_generation_config.set_eos_token_id(m_tokenizer.get_eos_token_id());
        }
    }

    Impl(const std::filesystem::path& model_path, const std::string& device, const ov::AnyMap& plugin_config)
        : Impl{model_path, Tokenizer(model_path.string()), device, plugin_config} {}

    DecodedResults generate(const RawSpeechInput& raw_speech_input,
                            OptionalWhisperGenerationConfig generation_config,
                            StreamerVariant streamer) {
        auto start_time = std::chrono::steady_clock::now();
        WhisperGenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;

        std::shared_ptr<StreamerBase> streamer_ptr;
        if (auto streamer_obj = std::get_if<std::monostate>(&streamer)) {
            streamer_ptr = nullptr;
        } else if (auto streamer_obj = std::get_if<std::shared_ptr<StreamerBase>>(&streamer)) {
            streamer_ptr = *streamer_obj;
        } else if (auto callback = std::get_if<std::function<bool(std::string)>>(&streamer)) {
            streamer_ptr = std::make_shared<TextCallbackStreamer>(m_tokenizer, *callback);
        }

        auto tokens =
            ov::genai::whisper_generate(config, raw_speech_input, m_models, m_feature_extractor, streamer_ptr);

        DecodedResults decoded_results{std::vector{m_tokenizer.decode(tokens)}, std::vector{1.f}};
        return decoded_results;
    }
};

}  // namespace genai
}  // namespace ov

ov::genai::WhisperPipeline::WhisperPipeline(const std::string& model_path,
                                            const ov::genai::Tokenizer& tokenizer,
                                            const std::string& device,
                                            const ov::AnyMap& plugin_config) {
    auto start_time = std::chrono::steady_clock::now();
    m_impl = std::make_unique<WhisperPipeline::Impl>(model_path, tokenizer, device, plugin_config);
    auto stop_time = std::chrono::steady_clock::now();
    m_impl->m_load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
}

ov::genai::WhisperPipeline::WhisperPipeline(const std::string& model_path,
                                            const std::string& device,
                                            const ov::AnyMap& plugin_config) {
    auto start_time = std::chrono::steady_clock::now();
    m_impl = std::make_unique<WhisperPipeline::Impl>(model_path, device, plugin_config);
    auto stop_time = std::chrono::steady_clock::now();
    m_impl->m_load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
}

ov::genai::DecodedResults ov::genai::WhisperPipeline::generate(const RawSpeechInput& raw_speech_input,
                                                               OptionalWhisperGenerationConfig generation_config,
                                                               StreamerVariant streamer) {
    return m_impl->generate(raw_speech_input, generation_config, streamer);
}

ov::genai::DecodedResults ov::genai::WhisperPipeline::generate(const RawSpeechInput& raw_speech_input,
                                                               const ov::AnyMap& config_map) {
    auto config_arg = get_config_from_map(config_map);
    WhisperGenerationConfig config = (config_arg.has_value()) ? *config_arg : get_generation_config();
    config.update_generation_config(config_map);

    return m_impl->generate(raw_speech_input, config, utils::get_streamer_from_map(config_map));
}

ov::genai::WhisperGenerationConfig ov::genai::WhisperPipeline::get_generation_config() const {
    return m_impl->m_generation_config;
}

ov::genai::Tokenizer ov::genai::WhisperPipeline::get_tokenizer() {
    return m_impl->m_tokenizer;
}

void ov::genai::WhisperPipeline::set_generation_config(const WhisperGenerationConfig& config) {
    int64_t default_eos_token_id = m_impl->m_generation_config.eos_token_id;
    m_impl->m_generation_config = config;
    // if eos_token_id was not provided in config forward from default config
    if (config.eos_token_id == -1)
        m_impl->m_generation_config.eos_token_id = default_eos_token_id;

    m_impl->m_generation_config.validate();
}

ov::genai::WhisperPipeline::~WhisperPipeline() = default;
