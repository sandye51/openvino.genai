// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "../cpp/src/tokenizers_path.hpp"
#include "./utils.hpp"
#include "openvino/genai/whisper_generation_config.hpp"
#include "openvino/genai/whisper_pipeline.hpp"

namespace py = pybind11;
using ov::genai::DecodedResults;
using ov::genai::OptionalWhisperGenerationConfig;
using ov::genai::RawSpeechInput;
using ov::genai::StreamerBase;
using ov::genai::StreamerVariant;
using ov::genai::Tokenizer;
using ov::genai::WhisperGenerationConfig;
using ov::genai::WhisperPipeline;

namespace utils = ov::genai::pybind::utils;

namespace {

auto whisper_generate_docstring = R"(
    High level generate that receives raw speech as a vector of floats and returns decoded output.
    
    :param raw_speech_input: inputs in the form of list of floats. Required to be normalized to near [-1, 1] range and have 16k Hz sampling rate.
    :type raw_speech_input: List[float]
    
    :param generation_config: generation_config
    :type generation_config: WhisperGenerationConfig or a Dict

    :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped
    :type : Callable[[str], bool], ov.genai.StreamerBase

    :param kwargs: arbitrary keyword arguments with keys corresponding to WhisperGenerationConfig fields.
    :type : Dict

    :return: return results in encoded, or decoded form depending on inputs type
    :rtype: DecodedResults
)";

auto whisper_generation_config_docstring = R"(
    WhisperGenerationConfig parameters
    max_length: the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
                `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
    type: int

    max_new_tokens: the maximum numbers of tokens to generate, excluding the number of tokens in the prompt. max_new_tokens has priority over max_length.
    type: int

    eos_token_id: End of stream token id.
    type: int

    Whisper specific parameters:

    decoder_start_token_id: Corresponds to the ”<|startoftranscript|>” token.
    type: int

    language_token_id: The language token id of the transcription text. It is appended to the start of the sequence for
                       multilingual speech recognition, e.g. for Spanish the token id corresponding to the "<|es|>" is appended to the
                       start of sequence.
    type: int
    
    pad_token_id: Padding token id.
    type: int
    
    translate_token_id: Translate token id.
    type: int
    
    transcribe_token_id: Transcribe token id.
    type: int
    
    no_timestamps_token_id: No timestamps token id.
    type: int
    
    begin_timestamps_token_id: Begin timestamps token id.
    type: int
    
    is_multilingual:
    type: bool
    
    begin_suppress_tokens: A list containing tokens that will be supressed at the beginning of the sampling process.
    type: list[int]

    suppress_tokens: A list containing the non-speech tokens that will be supressed during generation.
    type: list[int]
)";

OptionalWhisperGenerationConfig update_whisper_config_from_kwargs(const OptionalWhisperGenerationConfig& config,
                                                                  const py::kwargs& kwargs) {
    if (!config.has_value() && kwargs.empty())
        return std::nullopt;

    WhisperGenerationConfig res_config;
    if (config.has_value())
        res_config = *config;

    for (const auto& item : kwargs) {
        std::string key = py::cast<std::string>(item.first);
        py::object value = py::cast<py::object>(item.second);

        if (item.second.is_none()) {
            // Even if argument key name does not fit GenerationConfig name
            // it's not an eror if it's not defined.
            // Some HF configs can have parameters for methods currenly unsupported in ov_genai
            // but if their values are not set / None, then this should not block
            // us from reading such configs, e.g. {"typical_p": None, 'top_p': 1.0,...}
            return res_config;
        }

        if (key == "max_new_tokens") {
            res_config.max_new_tokens = py::cast<int>(item.second);
        } else if (key == "max_length") {
            res_config.max_length = py::cast<int>(item.second);
        } else if (key == "decoder_start_token_id") {
            res_config.decoder_start_token_id = py::cast<int>(item.second);
        } else if (key == "language_token_id") {
            res_config.language_token_id = py::cast<int>(item.second);
        } else if (key == "pad_token_id") {
            res_config.pad_token_id = py::cast<int>(item.second);
        } else if (key == "translate_token_id") {
            res_config.translate_token_id = py::cast<int>(item.second);
        } else if (key == "transcribe_token_id") {
            res_config.transcribe_token_id = py::cast<int>(item.second);
        } else if (key == "no_timestamps_token_id") {
            res_config.no_timestamps_token_id = py::cast<int>(item.second);
        } else if (key == "begin_timestamps_token_id") {
            res_config.begin_timestamps_token_id = py::cast<int>(item.second);
        } else if (key == "begin_suppress_tokens") {
            res_config.begin_suppress_tokens = py::cast<std::vector<int64_t>>(item.second);
        } else if (key == "suppress_tokens") {
            res_config.suppress_tokens = py::cast<std::vector<int64_t>>(item.second);
        } else if (key == "is_multilingual") {
            res_config.is_multilingual = py::cast<bool>(item.second);
        } else if (key == "eos_token_id") {
            res_config.set_eos_token_id(py::cast<int>(item.second));
        } else {
            throw(std::invalid_argument(
                "'" + key +
                "' is incorrect WhisperGenerationConfig parameter name. "
                "Use help(openvino_genai.WhisperGenerationConfig) to get list of acceptable parameters."));
        }
    }

    return res_config;
}

py::object call_whisper_common_generate(WhisperPipeline& pipe,
                                        const RawSpeechInput& raw_speech_input,
                                        const OptionalWhisperGenerationConfig& config,
                                        const utils::PyBindStreamerVariant& py_streamer,
                                        const py::kwargs& kwargs) {
    // whisper config should initialized from generation_config.json in case of only kwargs provided
    // otherwise it would be initialized with default values which is unexpected for kwargs use case
    // if full config was provided then rely on it as a base config
    OptionalWhisperGenerationConfig base_config = config.has_value() ? config : pipe.get_generation_config();

    auto updated_config = update_whisper_config_from_kwargs(base_config, kwargs);

    StreamerVariant streamer = std::monostate();

    std::visit(utils::overloaded{[&streamer](const std::function<bool(py::str)>& py_callback) {
                                     // Wrap python streamer with manual utf-8 decoding. Do not rely
                                     // on pybind automatic decoding since it raises exceptions on incomplete strings.
                                     auto callback_wrapped = [&py_callback](std::string subword) -> bool {
                                         auto py_str =
                                             PyUnicode_DecodeUTF8(subword.data(), subword.length(), "replace");
                                         return py_callback(py::reinterpret_borrow<py::str>(py_str));
                                     };
                                     streamer = callback_wrapped;
                                 },
                                 [&streamer](std::shared_ptr<StreamerBase> streamer_cls) {
                                     streamer = streamer_cls;
                                 },
                                 [](std::monostate none) { /*streamer is already a monostate */ }},
               py_streamer);

    return py::cast(pipe.generate(raw_speech_input, updated_config, streamer));
}
}  // namespace

void init_whisper_pipeline(py::module_& m) {
    m.doc() = "Pybind11 binding for Whisper Pipeline";

    // Binding for WhisperGenerationConfig
    py::class_<WhisperGenerationConfig>(m, "WhisperGenerationConfig", whisper_generation_config_docstring)
        .def(py::init<std::string>(), py::arg("json_path"), "path where generation_config.json is stored")
        .def(py::init([](py::kwargs kwargs) {
            return *update_whisper_config_from_kwargs(WhisperGenerationConfig(), kwargs);
        }))
        .def_readwrite("max_new_tokens", &WhisperGenerationConfig::max_new_tokens)
        .def_readwrite("max_length", &WhisperGenerationConfig::max_length)
        .def_readwrite("begin_suppress_tokens", &WhisperGenerationConfig::begin_suppress_tokens)
        .def_readwrite("suppress_tokens", &WhisperGenerationConfig::suppress_tokens)
        .def_readwrite("decoder_start_token_id", &WhisperGenerationConfig::decoder_start_token_id)
        .def_readwrite("language_token_id", &WhisperGenerationConfig::language_token_id)
        .def_readwrite("eos_token_id", &WhisperGenerationConfig::eos_token_id)
        .def_readwrite("pad_token_id", &WhisperGenerationConfig::pad_token_id)
        .def_readwrite("translate_token_id", &WhisperGenerationConfig::translate_token_id)
        .def_readwrite("transcribe_token_id", &WhisperGenerationConfig::transcribe_token_id)
        .def_readwrite("begin_timestamps_token_id", &WhisperGenerationConfig::begin_timestamps_token_id)
        .def_readwrite("no_timestamps_token_id", &WhisperGenerationConfig::no_timestamps_token_id)
        .def_readwrite("is_multilingual", &WhisperGenerationConfig::is_multilingual)
        .def("set_eos_token_id", &WhisperGenerationConfig::set_eos_token_id);

    py::class_<WhisperPipeline>(m, "WhisperPipeline")
        .def(py::init([](const std::string& model_path,
                         const std::string& device,
                         const std::map<std::string, py::object>& config) {
                 ScopedVar env_manager(utils::ov_tokenizers_module_path());
                 return std::make_unique<WhisperPipeline>(model_path, device, utils::properties_to_any_map(config));
             }),
             py::arg("model_path"),
             "folder with openvino_model.xml and openvino_tokenizer[detokenizer].xml files",
             py::arg("device") = "CPU",
             "device on which inference will be done",
             py::arg("config") = ov::AnyMap({}),
             "openvino.properties map",
             R"(
            WhisperPipeline class constructor.
            model_path (str): Path to the model file.
            device (str): Device to run the model on (e.g., CPU, GPU). Default is 'CPU'.
        )")

        .def(py::init([](const std::string& model_path,
                         const Tokenizer& tokenizer,
                         const std::string& device,
                         const std::map<std::string, py::object>& config) {
                 return std::make_unique<WhisperPipeline>(model_path,
                                                          tokenizer,
                                                          device,
                                                          utils::properties_to_any_map(config));
             }),
             py::arg("model_path"),
             py::arg("tokenizer"),
             py::arg("device") = "CPU",
             py::arg("config") = ov::AnyMap({}),
             "openvino.properties map",
             R"(
            WhisperPipeline class constructor for manualy created openvino_genai.Tokenizer.
            model_path (str): Path to the model file.
            tokenizer (openvino_genai.Tokenizer): tokenizer object.
            device (str): Device to run the model on (e.g., CPU, GPU). Default is 'CPU'.
        )")

        .def(
            "generate",
            [](WhisperPipeline& pipe,
               const RawSpeechInput& raw_speech_input,
               const OptionalWhisperGenerationConfig& generation_config,
               const utils::PyBindStreamerVariant& streamer,
               const py::kwargs& kwargs) {
                return call_whisper_common_generate(pipe, raw_speech_input, generation_config, streamer, kwargs);
            },
            py::arg("raw_speech_input"),
            "List of floats representing raw speech audio. "
            "Required to be normalized to near [-1, 1] range and have 16k Hz sampling rate.",
            py::arg("generation_config") = std::nullopt,
            "generation_config",
            py::arg("streamer") = std::monostate(),
            "streamer",
            (whisper_generate_docstring + std::string(" \n ") + whisper_generation_config_docstring).c_str())

        .def("get_tokenizer", &WhisperPipeline::get_tokenizer)
        .def("get_generation_config", &WhisperPipeline::get_generation_config, py::return_value_policy::copy)
        .def("set_generation_config", &WhisperPipeline::set_generation_config);
}
