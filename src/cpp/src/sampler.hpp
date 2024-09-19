
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <list>
#include <cassert>
#include <cstdlib>
#include <limits>
#include <map>
#include <algorithm>
#include <cmath>
#include <random>
#include <set>

#include "openvino/runtime/tensor.hpp"

#include "logit_processor.hpp"
#include "scheduler.hpp"
#include "sequence_group.hpp"

namespace ov::genai {
// Modifyed Knuth–Morris–Pratt algorithm which returns tokens following after every needle occurance in haystack
std::vector<int64_t> kmp_search(const std::vector<int64_t>& haystack, const std::vector<int64_t>& needle) {
    if (needle.empty()) {  // no_repeat_ngram_size == 1, ban every token
        return {haystack.begin(), haystack.end()};
    }
    std::vector<int> partial_match_table(needle.size() + 1, -1);
    int cnd = 0;
    for (size_t pos = 1; pos < needle.size(); ++pos) {
        if (needle.at(pos) == needle.at(size_t(cnd))) {
            partial_match_table.at(pos) = partial_match_table.at(size_t(cnd));
        } else {
            partial_match_table.at(pos) = cnd;
            while (cnd >= 0 && needle.at(pos) != needle.at(size_t(cnd))) {
                cnd = partial_match_table.at(size_t(cnd));
            }
        }
        ++cnd;
    }
    partial_match_table.back() = cnd;
    std::vector<int64_t> res;
    size_t haystack_id = 0;
    int needle_id = 0;
    while (haystack_id < haystack.size() - 1) {
        if (needle.at(size_t(needle_id)) == haystack.at(haystack_id)) {
            ++haystack_id;
            ++needle_id;
            if (needle_id == int(needle.size())) {
                res.push_back(haystack.at(haystack_id));
                needle_id = partial_match_table.at(size_t(needle_id));
            }
        } else {
            needle_id = partial_match_table.at(size_t(needle_id));
            if (needle_id < 0) {
                ++haystack_id;
                ++needle_id;
            }
        }
    }
    return res;
}

std::vector<Token> log_softmax(const ov::Tensor& logits, size_t batch_idx) {
    ov::Shape shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3);
    size_t batch = shape[0], seq_len = shape[1], vocab_size = shape[2];
    OPENVINO_ASSERT(batch_idx < batch, "Logits batch size doesn't match the number of beams");

    size_t batch_offset = batch_idx * seq_len * vocab_size, sequence_offset = (seq_len - 1) * vocab_size;
    const float* beam_logits = logits.data<const float>() + batch_offset + sequence_offset;
    float max_logit = *std::max_element(beam_logits, beam_logits + vocab_size);
    float log_sum = std::log(std::accumulate(
        beam_logits, beam_logits + vocab_size, 0.0f, [max_logit](float accumulated, float to_add) {
            return accumulated + std::exp(to_add - max_logit);
    }));

    std::vector<Token> tokens;
    tokens.reserve(vocab_size);
    for (size_t idx = 0; idx < vocab_size; ++idx)
        tokens.push_back({beam_logits[idx] - max_logit - log_sum, int64_t(idx)});

    return tokens;
}

std::vector<int64_t> wrap_tokens(const std::vector<int64_t>& tokens, const std::vector<int64_t>& prefix_tokens, const std::vector<int64_t>& suffix_tokens) {
    std::vector<int64_t> all_tokens = prefix_tokens;
    all_tokens.insert(all_tokens.end(), tokens.begin(), tokens.end());
    all_tokens.insert(all_tokens.end(), suffix_tokens.begin(), suffix_tokens.end());
    return all_tokens;
}

std::string clean_wrapped_text(const std::string& wrapped_text, const std::string& prefix, const std::string& suffix) {
    auto prefix_pos = wrapped_text.find(prefix);
    OPENVINO_ASSERT(prefix_pos != std::string::npos);
    auto suffix_pos = wrapped_text.rfind(suffix);
    OPENVINO_ASSERT(suffix_pos != std::string::npos);
    auto clean_text_start = prefix_pos + prefix.size();
    auto clean_text_length = suffix_pos - clean_text_start;
    std::string clean_text = wrapped_text.substr(clean_text_start, clean_text_length);
    return clean_text;
}

// Return number of last tokens that match one of the stop_strings. If there's no match 0 is returned.
int match_stop_string(Tokenizer & tokenizer, const TokenIds & generated_tokens, const std::set<std::string> & stop_strings) {
    /*
    For catching stop_string hit we run comparisons character-wise to catch cases where stop string 
    overlaps with part of another token on both sides or is just a part of a single token. 
    For every stop_string we iterate over generated tokens starting from the last one and going backwards. 
    Every token is wrapped with prefix tokens to ensure tokenizer doesn't remove prefix whitespace of the actual token.
    After that all tokens are decoded and prefix is removed from the decoded text, so we end up with decoded token.
    Its characters are compared to the stop_string character at a current_position 
    (position of a character in the stop_string counting from the last one) - at the begining position is 0.
    When characters match we increase current_position and check if we have a full match already, if not we continue.
    If we have already matched some characters (current_position > 0) and next character is not matching 
    before we reach the full match, then we reset current_position to 0. 
    */ 
    std::string prefix = "a";
    auto prefix_ov = tokenizer.encode(prefix).input_ids;
    std::vector<int64_t> prefix_tokens(prefix_ov.data<int64_t>(), prefix_ov.data<int64_t>() + prefix_ov.get_size());
    std::string suffix = "b";
    auto suffix_ov = tokenizer.encode(suffix).input_ids;
    std::vector<int64_t> suffix_tokens(suffix_ov.data<int64_t>(), suffix_ov.data<int64_t>() + suffix_ov.get_size());

    // Since whitespace can be added at the beginning of the suffix we also try to capture that behavior here
    // and get suffix string that will actually be part of the decoded string so we can remove it correctly
    auto wrapped_suffix_tokens = suffix_tokens;
    wrapped_suffix_tokens.insert(wrapped_suffix_tokens.begin(), prefix_tokens.begin(), prefix_tokens.end());
    std::string wrapped_suffix = tokenizer.decode(wrapped_suffix_tokens);
    auto wrapper_pos = wrapped_suffix.find(prefix);
    suffix = wrapped_suffix.substr(wrapper_pos + prefix.size());
    
    for (auto stop_string: stop_strings) {
        int current_position = 0;
        int num_matched_tokens = 0; 
        // Getting reverse iterator to check tokens starting from the last one generated and going backwards
        auto generated_tokens_rit = generated_tokens.rbegin();
        std::vector<int64_t> tokens_buffer;
        while (generated_tokens_rit != generated_tokens.rend()) {
            num_matched_tokens++;
            tokens_buffer.insert(tokens_buffer.begin(), *generated_tokens_rit);

            std::vector<int64_t> wrapped_tokens = wrap_tokens(tokens_buffer, prefix_tokens, suffix_tokens);
            std::string wrapped_text = tokenizer.decode(wrapped_tokens);
            std::string clean_text = clean_wrapped_text(wrapped_text, prefix, suffix);

            if (clean_text == "" || (clean_text.size() >= 3 && (clean_text.compare(clean_text.size() - 3, 3, "�") == 0))) { 
                generated_tokens_rit++;
                continue;
            } else {
                tokens_buffer.clear();
            }
            // Checking clean_text characters starting from the last one
            for (auto clean_text_rit = clean_text.rbegin(); clean_text_rit != clean_text.rend(); clean_text_rit++) {
                // On character match increment current_position for the next comparisons
                if (*clean_text_rit == *(stop_string.rbegin() + current_position)) {
                    current_position++;
                    // If this is the last character from the stop_string we have a match
                    if ((stop_string.rbegin() + current_position) == stop_string.rend()) {
                        return num_matched_tokens;
                    } 
                } else if (current_position) {
                    // Already found matching characters, but the last one didn't match, so we reset current_position
                    current_position = 0;
                    // Looking for the match will start over from this character so we decrement iterator
                    clean_text_rit--;
                }
            }
            generated_tokens_rit++;
        }
    }
    return 0;
}

// Return number of last tokens that match one of the stop_strings. If there's no match 0 is returned.
// Number of tokens might not be exact as if there's no direct token match, we decode generated tokens incrementally expanding decoding scope
// with 4 next tokens with each iteration until we check all tokens.
int match_stop_string2(Tokenizer & tokenizer, const TokenIds & generated_tokens, const std::set<std::string> & stop_strings) {
    for (auto stop_string: stop_strings) {
        auto stop_tokens_ov = tokenizer.encode(stop_string).input_ids;
        size_t num_tokens = stop_tokens_ov.get_size();
        if(num_tokens > generated_tokens.size())
            continue;

        // Check direct token match
        std::vector<int64_t> stop_tokens(stop_tokens_ov.data<int64_t>(), stop_tokens_ov.data<int64_t>() + num_tokens);
        std::vector<int64_t> last_generated_tokens(generated_tokens.end()-num_tokens, generated_tokens.end());
        if (stop_tokens == last_generated_tokens)
            return num_tokens;
        
        // Continue checking chunks of 4 tokens
        num_tokens += 4;
        while (num_tokens <= generated_tokens.size()) {
            std::vector<int64_t> last_generated_tokens(generated_tokens.end()-num_tokens, generated_tokens.end());
            std::string decoded_last_tokens = tokenizer.decode(last_generated_tokens);
            if (decoded_last_tokens.find(stop_string) != std::string::npos) {
                return num_tokens;
            }
            num_tokens += 4;
        }
    }

    return 0;
}

// Handle stop_token_ids
bool is_stop_token_id_hit(int64_t generated_token, const std::set<int64_t> & stop_token_ids) {
    for (auto & stop_token_id : stop_token_ids) {
        if (generated_token == stop_token_id)
            return true;
    }
    return false;
}

struct Beam {
    Sequence::Ptr m_sequence;
    size_t m_global_beam_idx = 0;

    // beam is made on top of sequence
    float m_log_prob = 0.0f;
    int64_t m_token_id = -1;

    // cumulative log probabilities
    float m_score = -std::numeric_limits<float>::infinity();

    Beam(Sequence::Ptr sequence)
        : m_sequence(std::move(sequence)) { }

    size_t get_generated_len() const {
        return m_sequence->get_generated_len();
    }
};

bool greater(const Beam& left, const Beam& right) {
    return left.m_score > right.m_score;
}

struct Group {
    std::vector<Beam> ongoing;  // Best beams in front
    std::vector<Beam> min_heap;  // The worst of the best completed beams is the first
    bool done = false;

    int64_t finish(Beam beam, const ov::genai::GenerationConfig& sampling_params) {
        int64_t preeempted_sequence_id = -1;
        float generated_len = beam.get_generated_len() + (is_stop_token_id_hit(beam.m_token_id, sampling_params.stop_token_ids) ? 1 : 0); // HF counts EOS token in generation length
        beam.m_score /= std::pow(generated_len, sampling_params.length_penalty);

        min_heap.push_back(beam);
        std::push_heap(min_heap.begin(), min_heap.end(), greater);
        assert(sampling_params.num_beams % sampling_params.num_beam_groups == 0 &&
            "number of beams should be divisible by number of groups");
        size_t group_size = sampling_params.num_beams / sampling_params.num_beam_groups;
        if (min_heap.size() > group_size) {
            std::pop_heap(min_heap.begin(), min_heap.end(), greater);
            preeempted_sequence_id = min_heap.back().m_sequence->get_id();
            min_heap.pop_back();
        }

        return preeempted_sequence_id;
    }

    void is_done(const ov::genai::GenerationConfig& sampling_params) {
        assert(sampling_params.num_beams % sampling_params.num_beam_groups == 0 &&
            "number of beams should be divisible by number of groups");
        size_t group_size = sampling_params.num_beams / sampling_params.num_beam_groups;
        if (min_heap.size() < group_size)
            return;

        const Beam& best_running_sequence = ongoing.front(), & worst_finished_sequence = min_heap.front();
        size_t cur_len = best_running_sequence.m_sequence->get_generated_len();
        float best_sum_logprobs = best_running_sequence.m_score;
        float worst_score = worst_finished_sequence.m_score;
        switch (sampling_params.stop_criteria) {
        case ov::genai::StopCriteria::EARLY:
            done = true;
            return;
        case ov::genai::StopCriteria::HEURISTIC: {
            float highest_attainable_score = best_sum_logprobs / std::pow(float(cur_len), sampling_params.length_penalty);
            done = worst_score >= highest_attainable_score;
            return;
        }
        case ov::genai::StopCriteria::NEVER: {
            size_t length = sampling_params.length_penalty > 0.0 ? sampling_params.max_new_tokens : cur_len;
            float highest_attainable_score = best_sum_logprobs / std::pow(float(length), sampling_params.length_penalty);
            done = worst_score >= highest_attainable_score;
            return;
        }
        default:
            OPENVINO_THROW("Beam search internal error: unkown mode");
        }
    }
};

struct SamplerOutput {
    // IDs of sequences that need to be dropped
    std::vector<uint64_t> m_dropped_sequences;
    // IDs of sequences that need to be forked (note, the same sequence can be forked multiple times)
    // it will later be used by scheduler to fork block_tables for child sequences
    std::unordered_map<uint64_t, std::list<uint64_t>> m_forked_sequences;
};

class GroupBeamSearcher {
    SequenceGroup::Ptr m_sequence_group;
    ov::genai::GenerationConfig m_parameters;
    std::vector<Group> m_groups;
    Tokenizer m_tokenizer;
public:
    explicit GroupBeamSearcher(SequenceGroup::Ptr sequence_group, Tokenizer tokenizer);

    void select_next_tokens(const ov::Tensor& logits, SamplerOutput& sampler_output);

    void finalize(SamplerOutput& sampler_output) {
        for (Group& group : m_groups) {
            if (!group.done) {
                for (Beam& beam : group.ongoing) {
                    uint64_t sequence_id = beam.m_sequence->get_id();

                    int64_t preempted_id = group.finish(beam, m_parameters);
                    if (preempted_id >= 0) {
                        // remove preempted one
                        m_sequence_group->remove_sequence(preempted_id);
                    }

                    // mark current sequence as finished
                    beam.m_sequence->set_status(SequenceStatus::FINISHED);
                    // Setting length since this function is used when sequence generated tokens number reaches max_new_tokens 
                    beam.m_sequence->set_finish_reason(GenerationFinishReason::LENGTH);
                    // we also need to drop add ongoing / forked sequences from scheduler
                    sampler_output.m_dropped_sequences.push_back(sequence_id);
                }
            }
        }
    }
};

class Sampler {

    Logits _get_logit_vector(ov::Tensor logits, size_t batch_idx = 1) {
        ov::Shape logits_shape = logits.get_shape();
        size_t batch_size = logits_shape[0], seq_len = logits_shape[1], vocab_size = logits_shape[2];
        OPENVINO_ASSERT(batch_idx <= batch_size);
        size_t batch_offset = batch_idx * seq_len * vocab_size;
        size_t sequence_offset = (seq_len - 1) * vocab_size;
        float* logits_data = logits.data<float>() + batch_offset + sequence_offset;

        return Logits{logits_data, vocab_size};
    }

    Token _greedy_sample(const Logits& logits) const {
        // For greedy sampling we do not expect sorting or shrinking considered tokens
        // so we can operate directly on the data buffer
        float max_value = -std::numeric_limits<float>::infinity();
        size_t max_index = 0;
        for (size_t i = 0; i < logits.m_size; ++i) {
            if (logits.m_data[i] > max_value) {
                max_value = logits.m_data[i];
                max_index = i;
            }
        }
        return Token(logits.m_data[max_index], max_index);
    }

    std::vector<Token> _multinomial_sample(const Logits& logits, size_t num_tokens_per_sequence) {
        // If top_p or top_k was applied we use sorted vector, if not we go with original buffer.
        std::vector<float> multinomial_weights;
        multinomial_weights.reserve(logits.m_size);
        if (logits.is_vector_initialized())
            for (auto& logit: logits.m_vector) multinomial_weights.emplace_back(logit.m_log_prob);
        else
            multinomial_weights.assign(logits.m_data, logits.m_data + logits.m_size);

        auto dist = std::discrete_distribution<size_t>(multinomial_weights.begin(), multinomial_weights.end()); // equivalent to multinomial with number of trials == 1
        
        std::vector<Token> out_tokens;
        for (size_t token_idx = 0; token_idx < num_tokens_per_sequence; ++token_idx) {
            size_t element_to_pick = dist(rng_engine);
            if (logits.is_vector_initialized())
                out_tokens.push_back(logits.m_vector[element_to_pick]);
            else
                out_tokens.emplace_back(logits.m_data[element_to_pick], element_to_pick);
        }
        return out_tokens;
    }

    std::vector<int64_t> _try_finish_generation(SequenceGroup::Ptr & sequence_group) {
        auto sampling_params = sequence_group->get_sampling_parameters();
        std::vector<int64_t> dropped_seq_ids;
        for (auto& running_sequence : sequence_group->get_running_sequences()) {
            const auto generated_len = running_sequence->get_generated_len();
            if (sampling_params.max_new_tokens == generated_len || 
                is_stop_token_id_hit(running_sequence->get_generated_ids().back(), sampling_params.stop_token_ids) && !sampling_params.ignore_eos) {
                // stop sequence by max_new_tokens or stop token (eos included)
                running_sequence->set_status(SequenceStatus::FINISHED);

                if (is_stop_token_id_hit(running_sequence->get_generated_ids().back(), sampling_params.stop_token_ids) && !sampling_params.ignore_eos) {
                    running_sequence->set_finish_reason(GenerationFinishReason::STOP);
                } else if (sampling_params.max_new_tokens == generated_len) {
                    running_sequence->set_finish_reason(GenerationFinishReason::LENGTH);
                }
                
                dropped_seq_ids.push_back(running_sequence->get_id());
                continue;
            }

            if (!sampling_params.stop_strings.empty()) {
                int num_matched_last_tokens = match_stop_string(m_tokenizer, running_sequence->get_generated_ids(), sampling_params.stop_strings);
                if (num_matched_last_tokens) {
                    if (!sampling_params.include_stop_str_in_output)
                        running_sequence->remove_last_tokens(num_matched_last_tokens);
                    running_sequence->set_status(SequenceStatus::FINISHED);
                    running_sequence->set_finish_reason(GenerationFinishReason::STOP);
                    dropped_seq_ids.push_back(running_sequence->get_id());
                }
            }
        }
        return dropped_seq_ids;
    }

    // request ID => beam search tracking information
    std::map<uint64_t, GroupBeamSearcher> m_beam_search_info;

    std::mt19937 rng_engine;
    // { request_id, logit_processor }
    std::map<uint64_t, LogitProcessor> m_logit_processors;

    Tokenizer m_tokenizer;

public:

    Sampler(Tokenizer & tokenizer) : m_tokenizer(tokenizer) {};

    SamplerOutput sample(std::vector<SequenceGroup::Ptr> & sequence_groups, ov::Tensor logits);

    void set_seed(size_t seed) { rng_engine.seed(seed); }

    void clear_beam_search_info(uint64_t request_id);
};

SamplerOutput Sampler::sample(std::vector<SequenceGroup::Ptr> & sequence_groups, ov::Tensor logits) {
    const float * logits_data = logits.data<float>();
    ov::Shape logits_shape = logits.get_shape();
    OPENVINO_ASSERT(logits_shape.size() == 3);
    size_t batch_seq_len = logits_shape[1], vocab_size = logits_shape[2];

    SamplerOutput sampler_output;

    for (size_t sequence_group_id = 0, currently_processed_tokens = 0; sequence_group_id < sequence_groups.size(); ++sequence_group_id) {
        SequenceGroup::Ptr sequence_group = sequence_groups[sequence_group_id];
        if (!sequence_group->is_scheduled())
            continue;

        size_t num_running_sequences = sequence_group->num_running_seqs();
        size_t actual_seq_len = sequence_group->get_num_scheduled_tokens(); // points to a token which needs to be sampled
        size_t padded_amount_of_processed_tokens = std::max(actual_seq_len, batch_seq_len);
        const ov::genai::GenerationConfig& sampling_params = sequence_group->get_sampling_parameters();

        const auto request_id = sequence_group->get_request_id();
        if (!m_logit_processors.count(request_id)) {
            m_logit_processors.insert({request_id, LogitProcessor(sampling_params, sequence_group->get_prompt_ids())});
        }
        auto& logit_processor = m_logit_processors.at(request_id);

        const void * sequence_group_logits_data = logits_data + vocab_size * currently_processed_tokens;
        ov::Tensor sequence_group_logits(ov::element::f32, ov::Shape{num_running_sequences, actual_seq_len, vocab_size}, (void *)sequence_group_logits_data);

        if (sequence_group->requires_sampling()) {
            if (sampling_params.is_greedy_decoding() || sampling_params.is_multinomial()) {
                std::vector<Sequence::Ptr> running_sequences = sequence_group->get_running_sequences();
                if (sampling_params.is_greedy_decoding()) {
                    OPENVINO_ASSERT(num_running_sequences == 1);
                }
                auto register_new_token = [&](const Token& sampled_token_id, Sequence::Ptr running_sequence) {
                    logit_processor.register_new_generated_token(sampled_token_id.m_index);
                    running_sequence->append_token(sampled_token_id.m_index, sampled_token_id.m_log_prob);
                };
                for (size_t running_sequence_id = 0; running_sequence_id < num_running_sequences; ++running_sequence_id) {
                    auto logit_vector = _get_logit_vector(sequence_group_logits, running_sequence_id);
                    logit_processor.apply(logit_vector);
                    Token sampled_token_id;
                    if (sampling_params.is_greedy_decoding()) {
                        sampled_token_id = _greedy_sample(logit_vector);
                    } else {
                        // is_multinomial()
                        const bool is_generate_n_tokens = sequence_group->num_total_seqs() == 1;
                        const size_t num_tokens_per_sequence = is_generate_n_tokens ? sampling_params.num_return_sequences : 1;
                        auto sampled_token_ids = _multinomial_sample(logit_vector, num_tokens_per_sequence);
                        sampled_token_id = sampled_token_ids[0];

                        if (is_generate_n_tokens) {
                            auto sequence_to_fork = running_sequences[0];
                            std::list<uint64_t> forked_seq_ids;
                            for (size_t i = num_running_sequences; i < num_tokens_per_sequence; ++i) {
                                const auto forked_sequence = sequence_group->fork_sequence(sequence_to_fork);
                                forked_seq_ids.push_back(forked_sequence->get_id());
                                register_new_token(sampled_token_ids[i], forked_sequence);
                            }
                            sampler_output.m_forked_sequences.insert({running_sequences[0]->get_id(), forked_seq_ids});
                        }
                    }
                    
                    register_new_token(sampled_token_id, running_sequences[running_sequence_id]);
                }
                logit_processor.increment_gen_tokens();
                for (const auto& dropped_seq_id : _try_finish_generation(sequence_group)) {
                    sampler_output.m_dropped_sequences.push_back(dropped_seq_id);
                }
            } else if (sampling_params.is_beam_search()) {
                uint64_t request_id = sequence_group->get_request_id();

                // create beam search info if we are on the first generate
                if (m_beam_search_info.find(request_id) == m_beam_search_info.end()) {
                    m_beam_search_info.emplace(request_id, GroupBeamSearcher(sequence_group, m_tokenizer));
                }

                // current algorithm already adds new tokens to running sequences and
                m_beam_search_info.at(request_id).select_next_tokens(sequence_group_logits, sampler_output);

                // check max length stop criteria
                std::vector<Sequence::Ptr> running_sequences = sequence_group->get_running_sequences();
                if (!sequence_group->has_finished() &&
                    running_sequences[0]->get_generated_len() == sampling_params.max_new_tokens) {
                    // stop sequence by max_new_tokens
                    m_beam_search_info.at(request_id).finalize(sampler_output);
                }
            }
            // Notify handle after sampling is done. 
            // For non-streaming this is effective only when the generation is finished.
            sequence_group->notify_handle();
        } else {
            // we are in prompt processing phase when prompt is split into chunks and processed step by step
        }

        // NOTE: it should be before 'get_num_scheduled_tokens' is used
        // update internal state of sequence group to reset scheduler tokens and update currently processed ones
        sequence_group->finish_iteration();

        // accumulate a number of processed tokens
        currently_processed_tokens += padded_amount_of_processed_tokens * num_running_sequences;
    }

    return sampler_output;
}

GroupBeamSearcher::GroupBeamSearcher(SequenceGroup::Ptr sequence_group, Tokenizer tokenizer)
    : m_sequence_group(sequence_group),
        m_parameters{m_sequence_group->get_sampling_parameters()},
        m_groups{m_parameters.num_beam_groups},
        m_tokenizer(tokenizer) {
    OPENVINO_ASSERT(m_sequence_group->num_running_seqs() == 1);
    assert(m_parameters.num_beams % m_parameters.num_beam_groups == 0 &&
        "number of beams should be divisible by number of groups");
    size_t group_size = m_parameters.num_beams / m_parameters.num_beam_groups;

    for (Group& group : m_groups) {
        group.ongoing.reserve(group_size);
        // initially we just add our "base" sequence to beams inside each group
        for (size_t i = 0; i < group_size; ++i)
            group.ongoing.push_back(Beam((*sequence_group)[0]));
        // to avoid selecting the same tokens for beams within group, let's just initialize score
        // for the front one
        group.ongoing.front().m_score = 0.0f;
    }
}

void GroupBeamSearcher::select_next_tokens(const ov::Tensor& logits, SamplerOutput& sampler_output) {
    assert(m_parameters.num_beams % m_parameters.num_beam_groups == 0 &&
        "number of beams should be divisible by number of groups");
    size_t group_size = m_parameters.num_beams / m_parameters.num_beam_groups;
    std::vector<int64_t> next_tokens;
    std::vector<int32_t> next_beams;
    next_tokens.reserve(m_parameters.num_beams);
    next_beams.reserve(m_parameters.num_beams);

    // parent sequence ID -> number of child sequences
    std::map<uint64_t, uint64_t> parent_2_num_childs_map;

    for (Group& group : m_groups) {
        if (!group.done) {
            for (Beam& beam : group.ongoing) {
                uint64_t parent_seq_id = beam.m_sequence->get_id();

                // here we need to map index of sequence in beam search group(s) and sequence group
                beam.m_global_beam_idx = [this] (uint64_t seq_id) -> size_t {
                    std::vector<Sequence::Ptr> running_seqs = m_sequence_group->get_running_sequences();
                    for (size_t seq_global_index = 0; seq_global_index < running_seqs.size(); ++seq_global_index) {
                        if (seq_id == running_seqs[seq_global_index]->get_id())
                            return seq_global_index;
                    }
                    OPENVINO_THROW("Internal error in beam search: should not be here");
                } (parent_seq_id);

                // zero out all parent forks counts
                parent_2_num_childs_map[parent_seq_id] = 0;
            }
        }
    }

    auto try_to_finish_candidate = [&] (Group& group, Beam& candidate, bool include_candidate_token = true) -> void {
        uint64_t seq_id = candidate.m_sequence->get_id();
        // try to finish candidate
        int64_t preempted_seq_id = group.finish(candidate, m_parameters);

        // if candidate has lower score than others finished
        if (preempted_seq_id == seq_id) {
            // do nothing and just ignore current finished candidate
        } else {
            if (preempted_seq_id >= 0) {
                m_sequence_group->remove_sequence(preempted_seq_id);
            }

            // need to insert candidate to a sequence group
            Sequence::Ptr forked_sequence = m_sequence_group->fork_sequence(candidate.m_sequence);
            // and finish immidiately
            forked_sequence->set_status(SequenceStatus::FINISHED);
            // Setting stop since this function is used when sequence generated eos token
            forked_sequence->set_finish_reason(GenerationFinishReason::STOP);

            // TODO: make it more simplier
            // currently, we finish sequence and then fork it in current code
            {
                for (size_t i = 0; i < group.min_heap.size(); ++i) {
                    if (group.min_heap[i].m_sequence->get_id() == seq_id) {
                        group.min_heap[i].m_sequence = forked_sequence;
                        break;
                    }
                }
            }

            // append token from candidate to actual sequence
            if (include_candidate_token)
                forked_sequence->append_token(candidate.m_token_id, candidate.m_log_prob);
        }
    };

    // group ID => child beams
    std::map<int, std::vector<Beam>> child_beams_per_group;

    for (size_t group_id = 0; group_id < m_groups.size(); ++group_id) {
        Group & group = m_groups[group_id];
        if (group.done)
            continue;

        std::vector<Beam> candidates;
        candidates.reserve(group_size * 2 * group_size);
        for (const Beam& beam : group.ongoing) {
            std::vector<Token> tokens = log_softmax(logits, beam.m_global_beam_idx);

            // apply diversity penalty
            for (auto prev_group_id = 0; prev_group_id < group_id; ++prev_group_id) {
                for (const Beam& prev_beam : child_beams_per_group[prev_group_id]) {
                    tokens[prev_beam.m_token_id].m_log_prob -= m_parameters.diversity_penalty;
                }
            }

            // apply n_gramm
            std::vector<int64_t> full_text{m_sequence_group->get_prompt_ids()};
            full_text.insert(full_text.end(), beam.m_sequence->get_generated_ids().begin(), beam.m_sequence->get_generated_ids().end());
            if (full_text.size() > 1 && full_text.size() >= m_parameters.no_repeat_ngram_size) {
                auto tail_start = full_text.end() - ptrdiff_t(m_parameters.no_repeat_ngram_size) + 1;
                for (int64_t banned_token : kmp_search(full_text, {tail_start, full_text.end()})) {
                    tokens[banned_token].m_log_prob = -std::numeric_limits<float>::infinity();
                }
            }

            // sort tokens
            std::sort(tokens.begin(), tokens.end(), [](Token left, Token right) {
                return left.m_log_prob > right.m_log_prob;  // Most probable tokens in front
            });

            size_t add_count = 0;
            for (Token token : tokens) {
                Beam new_candidate = beam;
                new_candidate.m_score += new_candidate.m_log_prob = token.m_log_prob;
                new_candidate.m_token_id = token.m_index;

                // TODO: fix it
                // and ensure cumulative_log prob is used
                if (/* m_parameters.early_finish(new_candidate) */ false) {
                    try_to_finish_candidate(group, new_candidate);
                } else {
                    candidates.push_back(new_candidate);
                    if (++add_count == 2 * group_size) {
                        break;
                    }
                }
            }
        }

        // Sample 2 * group_size highest score tokens to get at least 1 non EOS token per beam
        OPENVINO_ASSERT(candidates.size() >= 2 * group_size, "No beams left to search");

        auto to_sort = candidates.begin() + ptrdiff_t(2 * group_size);
        std::partial_sort(candidates.begin(), to_sort, candidates.end(), greater);

        for (size_t cand_idx = 0; cand_idx < candidates.size(); ++cand_idx) {
            Beam & candidate = candidates[cand_idx];
            if (is_stop_token_id_hit(candidate.m_token_id, m_sequence_group->get_sampling_parameters().stop_token_ids)) {
                // If beam_token does not belong to top num_beams tokens, it should not be added
                if (cand_idx >= group_size)
                    continue;

                // try to finish candidate
                try_to_finish_candidate(group, candidate);
                continue;
            }

            if (!m_parameters.stop_strings.empty()) {
                // We need to include candidate token to already generated tokens to check if stop string has been generated
                // There's probably a better way to do that, than copying whole vector...
                std::vector<int64_t> token_ids = candidate.m_sequence->get_generated_ids();
                token_ids.push_back(candidate.m_token_id);
                int num_last_matched_tokens = match_stop_string(m_tokenizer, token_ids, m_sequence_group->get_sampling_parameters().stop_strings);
                if (num_last_matched_tokens) {
                    // If beam_token does not belong to top num_beams tokens, it should not be added
                    if (cand_idx >= group_size)
                        continue;

                    if(!m_parameters.include_stop_str_in_output) {
                        // remove tokens that match stop_string from output (last token is not included in candidate.m_sequence at this point)
                        candidate.m_sequence->remove_last_tokens(num_last_matched_tokens - 1);
                    }

                    // try to finish candidate
                    try_to_finish_candidate(group, candidate, m_parameters.include_stop_str_in_output);
                    continue;
                }
            }

            parent_2_num_childs_map[candidate.m_sequence->get_id()] += 1;
            child_beams_per_group[group_id].push_back(candidate);

            // if num childs are enough
            if (child_beams_per_group[group_id].size() == group_size) {
                break;
            }
        }

        // check whether group has finished
        group.is_done(m_parameters);

        // group cannot continue if there are no valid child beams
        if (child_beams_per_group[group_id].size() == 0) {
            group.done = true;
        }

        if (group.done) {
            // group has finished, group all running sequences
            for (const Beam& beam : group.ongoing) {
                uint64_t seq_id = beam.m_sequence->get_id();
                m_sequence_group->remove_sequence(seq_id);
                sampler_output.m_dropped_sequences.push_back(seq_id);
            }
            group.ongoing.clear();
        }
    }

    // fork child sequences for non-finished groups

    for (size_t group_id = 0; group_id < m_groups.size(); ++group_id) {
        Group & group = m_groups[group_id];

        if (!group.done) {
            for (Beam& child_beam : child_beams_per_group[group_id]) {
                uint64_t parent_sequence_id = child_beam.m_sequence->get_id();
                uint64_t& num_childs = parent_2_num_childs_map[parent_sequence_id];

                // if current beam is forked multiple times
                if (num_childs > 1) {
                    child_beam.m_sequence = m_sequence_group->fork_sequence(child_beam.m_sequence);
                    child_beam.m_sequence->append_token(child_beam.m_token_id, child_beam.m_log_prob);

                    // reduce forks count, since fork already happened and next loop iteration
                    // will go by the second branch (num_childs == 1)
                    --num_childs;

                    // fill out sampler output
                    sampler_output.m_forked_sequences[parent_sequence_id].push_back(child_beam.m_sequence->get_id());
                } else if (num_childs == 1) {
                    // keep current sequence going and add a new token
                    child_beam.m_sequence->append_token(child_beam.m_token_id, child_beam.m_log_prob);
                }
            }

            // drop beams which are not forked by current group
            for (const Beam& beam : group.ongoing) {
                size_t num_childs = parent_2_num_childs_map[beam.m_sequence->get_id()];
                if (num_childs == 0) {
                    // drop sequence as not forked
                    sampler_output.m_dropped_sequences.push_back(beam.m_sequence->get_id());
                    m_sequence_group->remove_sequence(beam.m_sequence->get_id());
                }
            }

            // child become parents
            group.ongoing = child_beams_per_group[group_id];
        }
    }
}

void Sampler::clear_beam_search_info(uint64_t request_id) { 
    m_beam_search_info.erase(request_id);
}
}
