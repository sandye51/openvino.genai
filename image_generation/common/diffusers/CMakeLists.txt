# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

file(GLOB_RECURSE "diffusers_SRC" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(diffusers STATIC ${diffusers_SRC})
add_library(diffusers::diffusers ALIAS diffusers)

target_include_directories(diffusers
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(diffusers PUBLIC -march=native -Wall)
endif()

# dependencies

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(OpenVINOGenAI REQUIRED)

target_link_libraries(diffusers
    PUBLIC openvino::runtime
    PRIVATE openvino::genai)

include(FetchContent)

if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

if(NOT TARGET nlohmann_json)
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
        URL_HASH SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406)
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# TODO: remove after merge with openvino_genai
target_include_directories(diffusers PRIVATE
    "../../../src/cpp/src" "../../../src/cpp/include")

target_link_libraries(diffusers PRIVATE nlohmann_json::nlohmann_json)

FetchContent_Declare(eigen3)
FetchContent_GetProperties(eigen3)
if(NOT eigen3_POPULATED)
    FetchContent_Populate(eigen3 QUIET
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
        URL_HASH SHA256=1ccaabbfe870f60af3d6a519c53e09f3dcf630207321dffa553564a8e75c4fc8)
endif()

target_include_directories(diffusers SYSTEM PRIVATE "${eigen3_SOURCE_DIR}")

FetchContent_Declare(safetensors.h
    URL https://github.com/hsnyder/safetensors.h/archive/974a85d7dfd6e010558353226638bb26d6b9d756.tar.gz
    URL_HASH SHA256=9aaf5961609601cf9aaa96582a207bce7c6e5fbf57ed2cc669bb7bde6a937d4b)
FetchContent_MakeAvailable(safetensors.h)

target_include_directories(diffusers SYSTEM PRIVATE "${safetensors.h_SOURCE_DIR}")
