// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <memory>

#include "openvino/runtime/core.hpp"
#include "openvino/pass/manager.hpp"

#include "diffusers/lora.hpp"

class IAdaptedModel {
public:
    explicit IAdaptedModel(const std::string& model_path) {
        m_model = ov::Core().read_model(model_path);
    }

    void apply_lora(InsertLoRA::LoRAMap& lora_map) {
        if (!lora_map.empty()) {
            ov::pass::Manager manager;
            manager.register_pass<InsertLoRA>(lora_map);
            manager.run_passes(m_model);
        }
    }

    void compile(const std::string& device, const ov::AnyMap& properties = {}) {
        OPENVINO_ASSERT(m_model, "Model has been already compiled");
        ov::CompiledModel compiled_model = ov::Core().compile_model(m_model, device, properties);
        m_request = compiled_model.create_infer_request();
        // release the original model
        m_model.reset();
    }

protected:
    std::shared_ptr<ov::Model> m_model;
    ov::InferRequest m_request;
};

