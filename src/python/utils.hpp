// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "openvino/genai/streamer_base.hpp"

namespace py = pybind11;
using ov::genai::StreamerBase;

namespace ov::genai::pybind::utils {

// When StreamerVariant is used utf-8 decoding is done by pybind and can lead to exception on incomplete texts.
// Therefore strings decoding should be handled with PyUnicode_DecodeUTF8(..., "replace") to not throw errors.
using PyBindStreamerVariant = std::variant<std::function<bool(py::str)>, std::shared_ptr<StreamerBase>, std::monostate>;

template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

ov::Any py_object_to_any(const py::object& py_obj);

bool py_object_is_any_map(const py::object& py_obj);

ov::AnyMap py_object_to_any_map(const py::object& py_obj);

std::map<std::string, ov::Any> properties_to_any_map(const std::map<std::string, py::object>& properties);

std::string ov_tokenizers_module_path();

}  // namespace ov::genai::pybind::utils
