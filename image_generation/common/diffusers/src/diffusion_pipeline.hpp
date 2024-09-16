// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>

#include "schedulers/ischeduler.hpp"

#include "utils.hpp"

namespace {

void batch_copy(ov::Tensor src, ov::Tensor dst, size_t src_batch, size_t dst_batch, size_t batch_size = 1) {
    const ov::Shape src_shape = src.get_shape(), dst_shape = dst.get_shape();
    ov::Coordinate src_start(src_shape.size(), 0), src_end = src_shape;
    ov::Coordinate dst_start(dst_shape.size(), 0), dst_end = dst_shape;

    src_start[0] = src_batch;
    src_end[0] = src_batch + batch_size;

    dst_start[0] = dst_batch;
    dst_end[0] = dst_batch + batch_size;

    ov::Tensor(src, src_start, src_end).copy_to(ov::Tensor(dst, dst_start, dst_end));
}

// TODO: remove duplication with utils.hpp
template <typename T>
void read_anymap_param(const ov::AnyMap& config_map, const std::string& name, T& param) {
    auto it = config_map.find(name);
    if (it != config_map.end()) {
        param = it->second.as<T>();
    }
}

const std::string get_class_name(const std::string& root_dir) {
    const std::string model_index_path = root_dir + "/model_index.json";
    std::ifstream file(model_index_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using ov::genai::utils::read_json_param;

    return data["_class_name"].get<std::string>();
}

} // namespace

class Text2ImagePipeline::DiffusionPipeline {
public:
    GenerationConfig get_generation_config() const {
        return m_generation_config;
    }

    void set_generation_config(const GenerationConfig& generation_config) {
        m_generation_config = generation_config;
        m_generation_config.validate();
    }

    void set_scheduler(std::shared_ptr<Scheduler> scheduler) {
        auto casted = std::dynamic_pointer_cast<IScheduler>(scheduler);
        OPENVINO_ASSERT(casted != nullptr, "Passed incorrect scheduler type");
        m_scheduler = casted;
    }

    virtual void reshape(const int num_images_per_prompt, const int height, const int width, const float guidance_scale) = 0;

    virtual void apply_lora(const std::string& lora_path, float alpha) = 0;

    virtual void compile(const std::string& device, const ov::AnyMap& properties) = 0;

    virtual ov::Tensor generate(const std::string& positive_prompt, const ov::AnyMap& properties) = 0;

    virtual ~DiffusionPipeline() = default;

protected:
    virtual void initialize_generation_config(const std::string& class_name) = 0;

    virtual void check_inputs(const int height, const int width) const = 0;

    std::shared_ptr<IScheduler> m_scheduler;
    Text2ImagePipeline::GenerationConfig m_generation_config;
};
