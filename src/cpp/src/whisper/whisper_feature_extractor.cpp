// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper_feature_extractor.hpp"

#include <string>
#include <vector>

#include "openvino/genai/visibility.hpp"

#ifdef _WIN32
#    define _USE_MATH_DEFINES
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <openvino/core/except.hpp>
#include <thread>
#include <vector>

#include "../utils.hpp"

namespace {

struct whisper_mel {
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

struct whisper_filters {
    const size_t n_mel;
    const size_t n_fft;  // 1 + (N_FFT / 2)
    std::vector<float> data;
};

static bool hann_window(const size_t length, const bool periodic, std::vector<float>& output) {
    if (output.size() < length) {
        output.resize(length);
    }
    int offset = -1;
    if (periodic) {
        offset = 0;
    }
    for (int i = 0; i < length; i++) {
        output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
    }

    return true;
}

// naive Discrete Fourier Transform
// input is real-valued
// output is complex-valued
static void dft(const std::vector<float>& in,
                std::vector<float>& out,
                const std::vector<float>& sin_vals,
                const std::vector<float>& cos_vals,
                const size_t n_fft) {
    int N = in.size();

    out.resize(N * 2);
    const int sin_cos_step = n_fft / N;

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++) {
            int idx = (k * n * sin_cos_step) % (n_fft);  // t = 2*M_PI*k*n/N
            re += in[n] * cos_vals[idx];                 // cos(t)
            im -= in[n] * sin_vals[idx];                 // sin(t)
        }

        out[k * 2 + 0] = re;
        out[k * 2 + 1] = im;
    }
}

// Cooley-Tukey FFT
// input is real-valued
// output is complex-valued
static void fft(const std::vector<float>& in,
                std::vector<float>& out,
                const std::vector<float>& sin_vals,
                const std::vector<float>& cos_vals,
                const size_t n_fft) {
    out.resize(in.size() * 2);

    int N = in.size();

    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }

    if (N % 2 == 1) {
        dft(in, out, sin_vals, cos_vals, n_fft);
        return;
    }

    std::vector<float> even;
    std::vector<float> odd;

    even.reserve(N / 2);
    odd.reserve(N / 2);

    for (int i = 0; i < N; i++) {
        if (i % 2 == 0) {
            even.push_back(in[i]);
        } else {
            odd.push_back(in[i]);
        }
    }

    std::vector<float> even_fft;
    std::vector<float> odd_fft;

    fft(even, even_fft, sin_vals, cos_vals, n_fft);
    fft(odd, odd_fft, sin_vals, cos_vals, n_fft);

    const int sin_cos_step = n_fft / N;
    for (int k = 0; k < N / 2; k++) {
        int idx = k * sin_cos_step;  // t = 2*M_PI*k/N
        float re = cos_vals[idx];    // cos(t)
        float im = -sin_vals[idx];   // sin(t)

        float re_odd = odd_fft[2 * k + 0];
        float im_odd = odd_fft[2 * k + 1];

        out[2 * k + 0] = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

        out[2 * (k + N / 2) + 0] = even_fft[2 * k + 0] - re * re_odd + im * im_odd;
        out[2 * (k + N / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
}

static void log_mel_spectrogram_worker_thread(int ith,
                                              const std::vector<float>& hann,
                                              const std::vector<float>& samples,
                                              int n_samples,
                                              int frame_size,
                                              int frame_step,
                                              int n_threads,
                                              const whisper_filters& filters,
                                              whisper_mel& mel,
                                              const std::vector<float>& sin_vals,
                                              const std::vector<float>& cos_vals) {
    std::vector<float> fft_in(frame_size, 0.0);
    std::vector<float> fft_out(2 * frame_size);
    int n_fft = filters.n_fft;
    int i = ith;

    // make sure n_fft == 1 + (WHISPER_N_FFT / 2), bin_0 to bin_nyquist
    assert(n_fft == 1 + (frame_size / 2));

    // calculate FFT only when fft_in are not all zero
    for (; i < std::min(n_samples / frame_step + 1, mel.n_len); i += n_threads) {
        const int offset = i * frame_step;

        // apply Hanning window (~10% faster)
        for (int j = 0; j < std::min(frame_size, n_samples - offset); j++) {
            fft_in[j] = hann[j] * samples[offset + j];
        }
        // fill the rest with zeros
        if (n_samples - offset < frame_size) {
            std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0);
        }

        // FFT
        fft(fft_in, fft_out, sin_vals, cos_vals, frame_size);

        // Calculate modulus^2 of complex numbers
        // Use pow(fft_out[2 * j + 0], 2) + pow(fft_out[2 * j + 1], 2) causes inference quality problem? Interesting.
        for (int j = 0; j < n_fft; j++) {
            fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
        }

        // mel spectrogram
        for (int j = 0; j < mel.n_mel; j++) {
            double sum = 0.0;

            // unroll loop (suggested by GH user @lunixbochs)
            int k = 0;
            for (k = 0; k < n_fft - 3; k += 4) {
                sum += fft_out[k + 0] * filters.data[j * n_fft + k + 0] +
                       fft_out[k + 1] * filters.data[j * n_fft + k + 1] +
                       fft_out[k + 2] * filters.data[j * n_fft + k + 2] +
                       fft_out[k + 3] * filters.data[j * n_fft + k + 3];
            }

            // handle n_fft remainder
            for (; k < n_fft; k++) {
                sum += fft_out[k] * filters.data[j * n_fft + k];
            }

            sum = log10(std::max(sum, 1e-10));

            mel.data[j * mel.n_len + i] = sum;
        }
    }

    // Otherwise fft_out are all zero
    double sum = log10(1e-10);
    for (; i < mel.n_len; i += n_threads) {
        for (int j = 0; j < mel.n_mel; j++) {
            mel.data[j * mel.n_len + i] = sum;
        }
    }
}

// python implementation: https://github.com/huggingface/transformers/blob/check_gemma/src/transformers/audio_utils.py

float hertz_to_mel(const float freq) {
    constexpr float min_log_hertz = 1000.0;
    constexpr float min_log_mel = 15.0;
    const float logstep = 27.0 / log(6.4);
    float mel = 3.0 * freq / 200.0;

    if (freq >= min_log_hertz) {
        mel = min_log_mel + log(freq / min_log_hertz) * logstep;
    }
    return mel;
}

float mel_to_hertz(const float mel) {
    constexpr float min_log_hertz = 1000.0;
    constexpr float min_log_mel = 15.0;
    const float logstep = log(6.4) / 27.0;
    float freq = 200.0 * mel / 3.0;

    if (mel >= min_log_mel) {
        freq = min_log_hertz * exp(logstep * (mel - min_log_mel));
    }

    return freq;
}

std::vector<std::vector<float>> create_triangular_filter_bank(const std::vector<float>& fft_freqs,
                                                              const std::vector<float>& filter_freqs) {
    std::vector<float> filter_diff(filter_freqs.size() - 1);
    for (size_t i = 0; i < filter_diff.size(); i++) {
        filter_diff[i] = filter_freqs[i + 1] - filter_freqs[i];
    }

    std::vector<std::vector<float>> slopes(fft_freqs.size(), std::vector<float>(filter_freqs.size()));
    for (size_t row = 0; row < slopes.size(); row++) {
        for (size_t col = 0; col < slopes[0].size(); col++) {
            slopes[row][col] = filter_freqs[col] - fft_freqs[row];
        }
    }

    std::vector<std::vector<float>> down_slopes(fft_freqs.size(), std::vector<float>(filter_freqs.size() - 2));
    for (size_t row = 0; row < down_slopes.size(); row++) {
        for (size_t col = 0; col < down_slopes[0].size(); col++) {
            down_slopes[row][col] = -slopes[row][col] / filter_diff[col];
        }
    }

    std::vector<std::vector<float>> up_slopes(fft_freqs.size(), std::vector<float>(filter_freqs.size() - 2));
    for (size_t row = 0; row < up_slopes.size(); row++) {
        for (size_t col = 0; col < up_slopes[0].size(); col++) {
            up_slopes[row][col] = slopes[row][col + 2] / filter_diff[col + 1];
        }
    }

    std::vector<std::vector<float>> result(fft_freqs.size(), std::vector<float>(filter_freqs.size() - 2));
    for (size_t row = 0; row < result.size(); row++) {
        for (size_t col = 0; col < result[0].size(); col++) {
            result[row][col] = std::max(float(0), std::min(down_slopes[row][col], up_slopes[row][col]));
        }
    }

    return result;
}

std::vector<std::vector<float>> mel_filter_bank(const int64_t num_frequency_bins,
                                                const int64_t num_mel_filters,
                                                const int64_t sampling_rate,
                                                const float min_frequency = 0.0f,
                                                const float max_frequency = 8000.0f) {
    OPENVINO_ASSERT(max_frequency <= (sampling_rate / 2), "max_frequency should be less or equal sampling_rate / 2");

    const float mel_min = hertz_to_mel(min_frequency);
    const float mel_max = hertz_to_mel(max_frequency);

    const float mel_freqs_step = (mel_max - mel_min) / float(num_mel_filters + 1);
    std::vector<float> filter_freqs(num_mel_filters + 2);
    for (size_t i = 0; i < filter_freqs.size(); i++) {
        filter_freqs[i] = mel_to_hertz(mel_min + i * mel_freqs_step);
    }

    std::vector<float> fft_freqs(num_frequency_bins);
    const float fft_freq_step = float(sampling_rate / 2) / float(num_frequency_bins - 1);
    for (size_t i = 0; i < num_frequency_bins; i++) {
        fft_freqs[i] = i * fft_freq_step;
    }

    auto mel_filters = create_triangular_filter_bank(fft_freqs, filter_freqs);

    std::vector<float> enorm(num_mel_filters);
    for (size_t i = 0; i < enorm.size(); i++) {
        enorm[i] = 2.0f / (filter_freqs[i + 2] - filter_freqs[i]);
    }

    for (size_t row = 0; row < mel_filters.size(); row++) {
        for (size_t col = 0; col < mel_filters[0].size(); col++) {
            mel_filters[row][col] *= enorm[col];
        }
    }

    return mel_filters;
}

// In FFT, we frequently use sine and cosine operations with the same values.
// We can use precalculated values to speed up the process.
void fill_sin_cos_table(std::vector<float>& sin_vals, std::vector<float>& cos_vals, const size_t n_fft) {
    sin_vals.resize(n_fft);
    cos_vals.resize(n_fft);

    for (size_t i = 0; i < n_fft; i++) {
        double theta = (2 * M_PI * i) / n_fft;
        sin_vals[i] = sinf(theta);
        cos_vals[i] = cosf(theta);
    }
}

std::vector<float> mel_spectrogram_convert_audio(const std::vector<float>& raw_speech,
                                                 const size_t sampling_rate,
                                                 const size_t feature_size,
                                                 const size_t n_fft,
                                                 const size_t hop_length,
                                                 const size_t n_threads,
                                                 const std::vector<float>& mel_filter,
                                                 const std::vector<float>& sin_vals,
                                                 const std::vector<float>& cos_vals) {
    const float* samples = raw_speech.data();
    const int n_samples = raw_speech.size();
    whisper_filters filters{feature_size, 1 + n_fft / 2, mel_filter};
    whisper_mel mel;

    // Hanning window (Use cosf to eliminate difference)
    // ref: https://pytorch.org/docs/stable/generated/torch.hann_window.html
    // ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L147
    std::vector<float> hann;
    hann_window(n_fft, true, hann);

    // Calculate the length of padding
    int64_t stage_1_pad = sampling_rate * 30;
    int64_t stage_2_pad = n_fft / 2;

    // Initialize a vector and copy data from C array to it.
    std::vector<float> samples_padded;
    samples_padded.resize(n_samples + stage_1_pad + stage_2_pad * 2);
    std::copy(samples, samples + n_samples, samples_padded.begin() + stage_2_pad);

    // pad 30 seconds of zeros at the end of audio (480,000 samples) + reflective pad 200 samples at the end of audio
    std::fill(samples_padded.begin() + n_samples + stage_2_pad,
              samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad,
              0);

    // reflective pad 200 samples at the beginning of audio
    std::reverse_copy(samples + 1, samples + 1 + stage_2_pad, samples_padded.begin());
    // eric tmp
    samples_padded.resize(stage_1_pad + stage_2_pad * 2);

    mel.n_mel = filters.n_mel;
    // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SpectralOps.cpp#L936
    // Calculate number of frames + remove the last frame
    mel.n_len = (samples_padded.size() - n_fft) / hop_length;
    // Calculate semi-padded sample length to ensure compatibility
    mel.n_len_org = 1 + (n_samples + stage_2_pad - n_fft) / hop_length;
    mel.data.resize(mel.n_mel * mel.n_len);

    {
        std::vector<std::thread> workers(n_threads - 1);
        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw] = std::thread(log_mel_spectrogram_worker_thread,
                                      iw + 1,
                                      std::cref(hann),
                                      samples_padded,
                                      n_samples + stage_2_pad,
                                      n_fft,
                                      hop_length,
                                      n_threads,
                                      std::cref(filters),
                                      std::ref(mel),
                                      std::cref(sin_vals),
                                      std::cref(cos_vals));
        }

        // main thread
        log_mel_spectrogram_worker_thread(0,
                                          hann,
                                          samples_padded,
                                          n_samples + stage_2_pad,
                                          n_fft,
                                          hop_length,
                                          n_threads,
                                          filters,
                                          mel,
                                          sin_vals,
                                          cos_vals);

        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw].join();
        }
    }

    // clamping and normalization
    double mmax = -1e20;
    for (int i = 0; i < mel.n_mel * mel.n_len; i++) {
        if (mel.data[i] > mmax) {
            mmax = mel.data[i];
        }
    }

    mmax -= 8.0;

    for (int i = 0; i < mel.n_mel * mel.n_len; i++) {
        if (mel.data[i] < mmax) {
            mel.data[i] = mmax;
        }

        mel.data[i] = (mel.data[i] + 4.0) / 4.0;
    }

    return mel.data;
}

}  // namespace

namespace ov {
namespace genai {

WhisperFeatureExtractor::WhisperFeatureExtractor(const std::string& preprocessor_json_path) {
    init_parameters(preprocessor_json_path);
    fill_sin_cos_table(sin_vals, cos_vals, n_fft);
    init_mel_filter();
}

void WhisperFeatureExtractor::init_parameters(const std::string& preprocessor_json_path) {
    // preprocessor_config.json not found. Skip parameters initialization from file, use defaults.
    if (!std::filesystem::exists(preprocessor_json_path)) {
        return;
    }

    using ov::genai::utils::read_json_param;

    std::ifstream f(preprocessor_json_path);
    OPENVINO_ASSERT(f.is_open(), "Failed to open '" + preprocessor_json_path + "' with preprocessor config");

    nlohmann::json data = nlohmann::json::parse(f);

    read_json_param(data, "feature_size", feature_size);
    read_json_param(data, "sampling_rate", sampling_rate);
    read_json_param(data, "hop_length", hop_length);
    read_json_param(data, "n_fft", n_fft);
    read_json_param(data, "chunk_length", chunk_length);
    read_json_param(data, "n_samples", n_samples);
    read_json_param(data, "nb_max_frames", nb_max_frames);
};

void WhisperFeatureExtractor::init_mel_filter() {
    auto mel_data = mel_filter_bank(1 + n_fft / 2, feature_size, sampling_rate);
    mel_filter.resize(mel_data.size() * mel_data[0].size());

    for (size_t col = 0; col < mel_data[0].size(); col++) {
        for (size_t row = 0; row < mel_data.size(); row++) {
            mel_filter[col * mel_data.size() + row] = mel_data[row][col];
        }
    }
}

std::vector<float> WhisperFeatureExtractor::extract(const std::vector<float>& raw_speech) {
    size_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    return mel_spectrogram_convert_audio(raw_speech,
                                         sampling_rate,
                                         feature_size,
                                         n_fft,
                                         hop_length,
                                         n_threads,
                                         mel_filter,
                                         sin_vals,
                                         cos_vals);
}

}  // namespace genai
}  // namespace ov
