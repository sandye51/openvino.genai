// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "./utils.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <openvino/runtime/auto/properties.hpp>

#include "../cpp/src/tokenizers_path.hpp"
#include "openvino/genai/llm_pipeline.hpp"

namespace py = pybind11;
namespace ov::genai::pybind::utils {

bool py_object_is_any_map(const py::object& py_obj) {
    if (!py::isinstance<py::dict>(py_obj)) {
        return false;
    }
    auto dict = py::cast<py::dict>(py_obj);
    return std::all_of(dict.begin(), dict.end(), [&](const std::pair<py::object::handle, py::object::handle>& elem) {
        return py::isinstance<py::str>(elem.first);
    });
}

ov::Any py_object_to_any(const py::object& py_obj);

ov::AnyMap py_object_to_any_map(const py::object& py_obj) {
    OPENVINO_ASSERT(py_object_is_any_map(py_obj), "Unsupported attribute type.");
    ov::AnyMap return_value = {};
    for (auto& item : py::cast<py::dict>(py_obj)) {
        std::string key = py::cast<std::string>(item.first);
        py::object value = py::cast<py::object>(item.second);
        if (py_object_is_any_map(value)) {
            return_value[key] = py_object_to_any_map(value);
        } else {
            return_value[key] = py_object_to_any(value);
        }
    }
    return return_value;
}

ov::Any py_object_to_any(const py::object& py_obj) {
    // Python types
    py::object float_32_type = py::module_::import("numpy").attr("float32");
    
    if (py::isinstance<py::str>(py_obj)) {
        return py_obj.cast<std::string>();
    } else if (py::isinstance<py::bool_>(py_obj)) {
        return py_obj.cast<bool>();
    } else if (py::isinstance<py::bytes>(py_obj)) {
        return py_obj.cast<std::string>();
    } else if (py::isinstance<py::float_>(py_obj)) {
        return py_obj.cast<double>();
    } else if (py::isinstance(py_obj, float_32_type)) {
        return py_obj.cast<float>();
    } else if (py::isinstance<py::int_>(py_obj)) {
        return py_obj.cast<int64_t>();
    } else if (py::isinstance<py::none>(py_obj)) {
        return {};
    } else if (py::isinstance<py::list>(py_obj)) {
        auto _list = py_obj.cast<py::list>();
        enum class PY_TYPE : int { UNKNOWN = 0, STR, INT, FLOAT, BOOL, PARTIAL_SHAPE };
        PY_TYPE detected_type = PY_TYPE::UNKNOWN;
        for (const auto& it : _list) {
            auto check_type = [&](PY_TYPE type) {
                if (detected_type == PY_TYPE::UNKNOWN || detected_type == type) {
                    detected_type = type;
                    return;
                }
                OPENVINO_THROW("Incorrect attribute. Mixed types in the list are not allowed.");
            };
            if (py::isinstance<py::str>(it)) {
                check_type(PY_TYPE::STR);
            } else if (py::isinstance<py::int_>(it)) {
                check_type(PY_TYPE::INT);
            } else if (py::isinstance<py::float_>(it)) {
                check_type(PY_TYPE::FLOAT);
            } else if (py::isinstance<py::bool_>(it)) {
                check_type(PY_TYPE::BOOL);
            } else if (py::isinstance<ov::PartialShape>(it)) {
                check_type(PY_TYPE::PARTIAL_SHAPE);
            }
        }

        if (_list.empty())
            return ov::Any();

        switch (detected_type) {
        case PY_TYPE::STR:
            return _list.cast<std::vector<std::string>>();
        case PY_TYPE::FLOAT:
            return _list.cast<std::vector<double>>();
        case PY_TYPE::INT:
            return _list.cast<std::vector<int64_t>>();
        case PY_TYPE::BOOL:
            return _list.cast<std::vector<bool>>();
        case PY_TYPE::PARTIAL_SHAPE:
            return _list.cast<std::vector<ov::PartialShape>>();
        default:
            OPENVINO_ASSERT(false, "Unsupported attribute type.");
        }
    
    // OV types
    } else if (py_object_is_any_map(py_obj)) {
        return py_object_to_any_map(py_obj);
    } else if (py::isinstance<ov::Any>(py_obj)) {
        return py::cast<ov::Any>(py_obj);
    } else if (py::isinstance<ov::element::Type>(py_obj)) {
        return py::cast<ov::element::Type>(py_obj);
    } else if (py::isinstance<ov::PartialShape>(py_obj)) {
        return py::cast<ov::PartialShape>(py_obj);
    } else if (py::isinstance<ov::hint::Priority>(py_obj)) {
        return py::cast<ov::hint::Priority>(py_obj);
    } else if (py::isinstance<ov::hint::PerformanceMode>(py_obj)) {
        return py::cast<ov::hint::PerformanceMode>(py_obj);
    } else if (py::isinstance<ov::intel_auto::SchedulePolicy>(py_obj)) {
        return py::cast<ov::intel_auto::SchedulePolicy>(py_obj);
    } else if (py::isinstance<ov::hint::SchedulingCoreType>(py_obj)) {
        return py::cast<ov::hint::SchedulingCoreType>(py_obj);
    } else if (py::isinstance<std::set<ov::hint::ModelDistributionPolicy>>(py_obj)) {
        return py::cast<std::set<ov::hint::ModelDistributionPolicy>>(py_obj);
    } else if (py::isinstance<ov::hint::ExecutionMode>(py_obj)) {
        return py::cast<ov::hint::ExecutionMode>(py_obj);
    } else if (py::isinstance<ov::log::Level>(py_obj)) {
        return py::cast<ov::log::Level>(py_obj);
    } else if (py::isinstance<ov::device::Type>(py_obj)) {
        return py::cast<ov::device::Type>(py_obj);
    } else if (py::isinstance<ov::streams::Num>(py_obj)) {
        return py::cast<ov::streams::Num>(py_obj);
    } else if (py::isinstance<ov::Affinity>(py_obj)) {
        return py::cast<ov::Affinity>(py_obj);
    } else if (py::isinstance<ov::Tensor>(py_obj)) {
        return py::cast<ov::Tensor>(py_obj);
    } else if (py::isinstance<ov::Output<ov::Node>>(py_obj)) {
        return py::cast<ov::Output<ov::Node>>(py_obj);
    } else if (py::isinstance<ov::genai::SchedulerConfig>(py_obj)) {
        return py::cast<ov::genai::SchedulerConfig>(py_obj);
    } else if (py::isinstance<py::object>(py_obj)) {
        return py_obj;
    }
    OPENVINO_ASSERT(false, "Unsupported attribute type.");
}

std::map<std::string, ov::Any> properties_to_any_map(const std::map<std::string, py::object>& properties) {
    std::map<std::string, ov::Any> properties_to_cpp;
    for (const auto& property : properties) {
        properties_to_cpp[property.first] = py_object_to_any(property.second);
    }
    return properties_to_cpp;
}

std::string ov_tokenizers_module_path() {
    // Try a path relative to build artifacts folder first.
    std::filesystem::path from_relative = tokenizers_relative_to_genai();
    if (std::filesystem::exists(from_relative)) {
        return from_relative.string();
    }
    return py::str(py::module_::import("openvino_tokenizers").attr("_ext_path"));
}

}  // namespace ov::genai::pybind::utils
