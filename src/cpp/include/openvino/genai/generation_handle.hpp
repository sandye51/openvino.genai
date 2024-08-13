// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <unordered_map>

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov::genai {
enum class GenerationStatus {
    RUNNING = 0, // Default status for ongoing generation
    FINISHED = 1, // Status set when generation has been finished
    IGNORED = 2, // Status set when generation run into out-of-memory condition and could not be continued
    DROPPED_BY_PIPELINE = 3, // Currently not used, TODO: implement abort functionality
    DROPPED_BY_HANDLE = 4 // Status set when generation handle is dropped
};

struct EncodedGenerationResult {
    // request ID - obsolete when handle API is approved as handle will connect results with prompts.
    uint64_t m_request_id;

    // in a generic case we have multiple generation results per initial prompt
    // depending on sampling parameters (e.g. beam search or parallel sampling)
    std::vector<std::vector<int64_t>> m_generation_ids;
    // scores
    std::vector<float> m_scores;

    // Status of generation
    GenerationStatus m_status = GenerationStatus::RUNNING;
};

enum class GenerationFinishReason {
    NONE = 0, // Default value, when generation is not yet finished
    STOP = 1, // Generation finished naturally, by reaching end of sequence token
    LENGTH = 2 // Generation finished by reaching max_new_tokens limit
};

// Output of generate() method, which represents full information about request with a given request_id
struct GenerationResult {
    // request ID - obsolete when handle API is approved as handle will connect results with prompts.
    uint64_t m_request_id;

    // in a generic case we have multiple generation results per initial prompt
    // depending on sampling parameters (e.g. beam search or parallel sampling)
    std::vector<std::string> m_generation_ids;
    // scores
    std::vector<float> m_scores;

    // Status of generation
    GenerationStatus m_status = GenerationStatus::RUNNING;
};

// Represents already generated tokens of running generate() method.
// E.g. typically generate() method consists of multiple step() which generate
// token by token. This structure represents a vector of already generated tokens so far
// for a given prompt.
struct GenerationOutput {
    // Currently generated list of tokens
    std::vector<int64_t> generated_token_ids;
    // Score
    // For beam search case: beam score
    // For other sampling types: cumulative log probabilitity of output tokens
    float score;
    // Finish reason if generation has finished, NONE otherwise
    GenerationFinishReason finish_reason;
};

// Current outputs of step() method for all scheduled requests
using GenerationOutputs = std::unordered_map<uint64_t, GenerationOutput>;

class GenerationStream;

class OPENVINO_GENAI_EXPORTS GenerationHandle {
    std::shared_ptr<GenerationStream> m_generation_stream;
    ov::genai::GenerationConfig m_sampling_params;

    // whether client ha dropped session with pipeline
    bool is_dropped() const;
 
public:
    using Ptr = std::shared_ptr<GenerationHandle>;

    GenerationHandle(std::shared_ptr<GenerationStream> generation_stream, const ov::genai::GenerationConfig& sampling_params) :
        m_generation_stream(std::move(generation_stream)),
        m_sampling_params(sampling_params) {
    }

    ~GenerationHandle();

    // There can be only one handle for a request
    GenerationHandle(const GenerationHandle&) = delete;
    GenerationHandle& operator=(const GenerationHandle&) = delete;

    GenerationStatus get_status() const;

    // whether new tokens are available
    bool can_read() const;

    // client drops generation session on server
    void drop();

    GenerationOutputs back();
    // Reads result of a generation for single iteration
    GenerationOutputs read();
    // Reads all generated tokens for all sequences
    std::vector<GenerationOutput> read_all();
};

}
