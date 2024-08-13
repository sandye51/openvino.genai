// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/runtime/core.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

#include "openvino/genai/scheduler_config.hpp"

namespace ov::genai {
class DeviceConfig {
    ov::element::Type m_kv_cache_type;
    ov::Shape m_key_cache_shape, m_value_cache_shape;
    ov::Shape::value_type m_num_kv_heads, m_head_size, m_num_decoder_layers;
    size_t m_num_kv_blocks = 0;
    size_t m_block_size = 0;
    size_t m_cache_size = 0;
    std::string m_device;
    bool m_initialized = false;

    void _check_is_initialized() const {
        OPENVINO_ASSERT(m_initialized, "Internal error: some CB parameters are not yet computed. Please, call set_model_params first");
    }

public:
    DeviceConfig(ov::Core& core, const SchedulerConfig& scheduling_config, const std::string& device, const ov::AnyMap& plugin_config = {}) {
        m_device = device;

        // keep information about block size
        m_block_size = scheduling_config.block_size;

        if (m_device == "CPU") {
            auto inference_precision = core.get_property(device, ov::hint::inference_precision);
            m_kv_cache_type = inference_precision == ov::element::bf16 ? ov::element::bf16 : ov::element::f16;

            // if user sets precision hint, kv cache type should be changed
            const auto inference_precision_it = plugin_config.find(ov::hint::inference_precision.name());
            if (inference_precision_it != plugin_config.end()) {
                const auto inference_precision = inference_precision_it->second.as<ov::element::Type>();
                if (inference_precision == ov::element::f32) {
                    m_kv_cache_type = ov::element::f32;
                } else if (inference_precision == ov::element::f16) {
                    m_kv_cache_type = ov::element::f16;
                } else if (inference_precision == ov::element::bf16) {
                    m_kv_cache_type = ov::element::bf16;
                } else {
                    // use default f32
                    m_kv_cache_type = ov::element::f32;
                }
            }

            // if user sets ov::kv_cache_precision hint
            const auto kv_cache_precision_it = plugin_config.find(ov::hint::kv_cache_precision.name());
            if (kv_cache_precision_it != plugin_config.end()) {
                const auto kv_cache_precision = kv_cache_precision_it->second.as<ov::element::Type>();
                m_kv_cache_type = kv_cache_precision;
            }
        } else if (m_device == "GPU") {
            OPENVINO_ASSERT("GPU is not currently supported. Please, remove this assert and fill configuration");
        } else {
            OPENVINO_THROW(m_device, " is not supported by OpenVINO Continuous Batching");
        }

        OPENVINO_ASSERT(scheduling_config.num_kv_blocks > 0 || scheduling_config.cache_size > 0, "num_kv_blocks or cache_size should be more than zero.");
        if (scheduling_config.num_kv_blocks > 0) {
            m_num_kv_blocks = scheduling_config.num_kv_blocks;
        }
        else {
            m_cache_size = scheduling_config.cache_size;
        }
    }

    void set_model_params(size_t num_kv_heads, size_t head_size, size_t num_decoder_layers) {
        m_num_kv_heads = num_kv_heads;
        m_head_size = head_size;
        m_num_decoder_layers = num_decoder_layers;

        if (m_num_kv_blocks == 0) {
            OPENVINO_ASSERT(m_cache_size > 0, "num_kv_blocks or cache_size should be more than zero.");
            size_t size_in_bytes = m_cache_size * 1024 * 1024 * 1024;
            m_num_kv_blocks = size_in_bytes / (m_num_decoder_layers * 2 * m_num_kv_heads * m_block_size * m_head_size * m_kv_cache_type.size());
        }

        m_key_cache_shape = m_value_cache_shape = ov::Shape{m_num_kv_blocks,
                                                            m_num_kv_heads,
                                                            m_block_size,
                                                            m_head_size};

        m_initialized = true;
    }

    std::string get_device() const {
        return m_device;
    }

    ov::element::Type get_cache_precision() const {
        return m_kv_cache_type;
    }

    size_t get_num_layers() const {
        _check_is_initialized();
        return m_num_decoder_layers;
    }

    ov::Shape get_key_cache_shape() const {
        _check_is_initialized();
        return m_key_cache_shape;
    }

    ov::Shape get_value_cache_shape() const {
        _check_is_initialized();
        return m_value_cache_shape;
    }

    size_t get_num_kv_blocks() const {
        OPENVINO_ASSERT(m_num_kv_blocks > 0, "Num KV blocks is not yet computed. Please, call set_model_params first");
        return m_num_kv_blocks;
    }
};
}
