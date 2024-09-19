# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import pytest
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from openvino_genai import ContinuousBatchingPipeline, GenerationConfig, Tokenizer
from typing import List, TypedDict

from common import run_test_pipeline, get_models_list, get_model_and_tokenizer, save_ov_model_from_optimum, \
    generate_and_compare_with_reference_text, get_greedy, get_beam_search, get_multinomial_temperature, \
    get_greedy_with_penalties, get_multinomial_temperature, \
    get_multinomial_temperature_and_top_k, get_multinomial_temperature_and_top_p, \
    get_multinomial_temperature_top_p_and_top_k, DEFAULT_SCHEDULER_CONFIG, get_greedy_with_repetition_penalty, \
    get_multinomial_all_parameters, get_multinomial_temperature_and_num_return_sequence, \
    generate_and_compare_with_reference_text, get_greedy, get_greedy_with_min_and_max_tokens, \
    get_greedy_with_single_stop_string, get_greedy_with_multiple_stop_strings, get_greedy_with_multiple_stop_strings_no_match, \
    get_beam_search, get_beam_search_min_and_max_tokens, get_beam_search_with_single_stop_string, \
    get_beam_search_with_multiple_stop_strings, get_beam_search_with_multiple_stop_strings_no_match, get_multinomial_max_and_min_token, \
    get_multinomial_temperature_and_frequence_penalty, get_multinomial_temperature_and_presence_penalty, \
    generate_and_compare_with_hf, get_multinomial_temperature_and_repetition_penalty, get_scheduler_config


@pytest.mark.precommit
@pytest.mark.parametrize("model_id", get_models_list(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "precommit")))
@pytest.mark.xfail(
    raises=RuntimeError,
    reason="Test fails with error: CPU: head size must be multiple of 16, current: X. CVS-145986.",
    strict=True,
)
def test_sampling_precommit(tmp_path, model_id):
    run_test_pipeline(tmp_path, model_id)


@pytest.mark.nightly
@pytest.mark.parametrize("model_id", get_models_list(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "nightly")))
def test_sampling_nightly(tmp_path, model_id):
    run_test_pipeline(tmp_path, model_id)

@pytest.mark.real_models
@pytest.mark.parametrize("model_id", get_models_list(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "real_models")))
def test_real_models(tmp_path, model_id):
    run_test_pipeline(tmp_path, model_id)


@pytest.mark.precommit
def test_eos_beam_search(tmp_path):
    '''
    Current test checks that in case of beam search, some generation results
    explicitly have EOS token at the end, which is aligned with HF

    Example of current output:
    { -1.23264,  that I don't know about.
    I don't know what you're talking about, but I'm pretty sure it's a Canadian thing.</s> }
    '''
    model_id = "facebook/opt-125m"
    prompts = ["Tell me something about Canada"]
    generation_configs = [get_beam_search()]
    scheduler_config = get_scheduler_config()
    generate_and_compare_with_hf(model_id, prompts, generation_configs, scheduler_config, tmp_path)


@pytest.mark.precommit
def test_eos_greedy(tmp_path):
    '''
    Current test checks that in case of gready, some generation results
    explicitly have EOS token at the end, which is aligned with HF:

    Example of current output:
    {  a software program</s> }
    '''
    model_id = "bigscience/bloomz-560m"
    prompts = ["What is OpenVINO?"]
    generation_configs = [get_greedy()]
    scheduler_config = get_scheduler_config()
    generate_and_compare_with_hf(model_id, prompts, generation_configs, scheduler_config, tmp_path)

@pytest.mark.precommit
@pytest.mark.parametrize("generation_config", [get_greedy(), get_greedy_with_min_and_max_tokens(), get_greedy_with_repetition_penalty(), get_greedy_with_single_stop_string(),
                                               get_greedy_with_multiple_stop_strings(), get_greedy_with_multiple_stop_strings_no_match(), 
                                               get_beam_search(), get_beam_search_min_and_max_tokens(), get_beam_search_with_multiple_stop_strings_no_match(), ],
        ids=[
            "greedy",
            "greedy_with_min_and_max_tokens",
            "greedy_with_repetition_penalty",
            "greedy_with_single_stop_string",
            "greedy_with_multiple_stop_strings",
            "greedy_with_multiple_stop_strings_no_match",
            "beam",
            "beam_search_min_and_max_tokens",
            "beam_search_with_multiple_stop_strings_no_match",
            ])
def test_individual_generation_configs_deterministic(tmp_path, generation_config):
    prompts = [
            "What is OpenVINO?",
            ]
    generation_configs = [generation_config]
    model_id : str = "facebook/opt-125m"
    generate_and_compare_with_hf(model_id, prompts, generation_configs, DEFAULT_SCHEDULER_CONFIG, tmp_path)

@pytest.mark.precommit
@pytest.mark.xfail(
    raises=AssertionError,
    reason="Stop strings do not seem to work as expected with beam search in HF, so comparison will fail. If it changes, these cases shall be merged to the test above.",
    strict=True,
)
@pytest.mark.parametrize("generation_config", [get_beam_search_with_single_stop_string(), get_beam_search_with_multiple_stop_strings(),],
        ids=[
            "beam_search_with_single_stop_string",
            "beam_search_with_multiple_stop_strings",
            ])
def test_beam_search_with_stop_string(tmp_path, generation_config):
    prompts = [
            "What is OpenVINO?",
            ]
    generation_configs = [generation_config]
    model_id : str = "facebook/opt-125m"
    generate_and_compare_with_hf(model_id, prompts, generation_configs, DEFAULT_SCHEDULER_CONFIG, tmp_path)


class PlatformsRefTexts(TypedDict, total=False):
    linux: List[List[str]]
    win32: List[List[str]]
    darwin: List[List[str]]


def get_current_plarform_ref_texts(ref_texts: PlatformsRefTexts) -> List[List[str]]:
    # mac and win often have identical results
    # to avoid duplication, use win32 ref_text if no mac ref_texts were found
    if sys.platform == "darwin":
        result = ref_texts.get("darwin") or ref_texts.get("win32")
    else:
        result = ref_texts.get(sys.platform)
    if not result:
        raise RuntimeError("No ref_texts were provided")
    return result


@dataclass
class RandomSamplingTestStruct:
    generation_config: GenerationConfig
    prompts: List[str]
    ref_texts: List[List[str]]


RANDOM_SAMPLING_TEST_CASES = [
    RandomSamplingTestStruct(
        generation_config=get_multinomial_temperature(),
        prompts=["What is OpenVINO?"],
        ref_texts=[
            [
                "\n\nOpenVINO is a software development platform developed by OpenVINO, a set of technology companies and startups that enables developers to use the most"
            ]
        ],
    ),
    RandomSamplingTestStruct(
        generation_config=get_multinomial_temperature_and_top_p(),
        prompts=["What is OpenVINO?"],
        ref_texts=get_current_plarform_ref_texts({
            "linux": [
                [
                    "\nOpenVINO is an online application that allows users to create, test, and analyze their own software using a collection of software packages. The application"
                ]
            ],
            "win32": [
                [
                    "\n\nOpenVINO is a software development platform designed to allow developers to develop and commercialize the most important software products on the web. OpenV"
                ]
            ],
        })
    ),
    RandomSamplingTestStruct(
        generation_config=get_multinomial_temperature_and_top_k(),
        prompts=["What is OpenVINO?"],
        ref_texts=[
            [
                "\n\nOpenVINO is a software that allows users to create a virtual machine with the ability to create a virtual machine in a virtual environment. Open"
            ]
        ],
    ),
    RandomSamplingTestStruct(
        generation_config=get_multinomial_temperature_top_p_and_top_k(),
        prompts=["What is OpenVINO?"],
        ref_texts=get_current_plarform_ref_texts({
            "linux": [
                [
                    "\nOpenVINO is an open source software that allows developers to create, manage, and distribute software. It is an open source project that allows developers"
                ]
            ],
            "win32": [
                [
                    "\n\nOpenVINO is a software that allows users to create a virtual machine with the ability to create a virtual machine in a virtual environment. Open"
                ]
            ],
        }),
    ),
    RandomSamplingTestStruct(
        generation_config=get_multinomial_temperature_and_repetition_penalty(),
        prompts=["What is OpenVINO?"],
        ref_texts=[
            [
                "\nOpen Vino's are a new and improved way to find cheap, fast-investment frozen vegetables that have no waste or calories. They're"
            ]
        ],
    ),
    RandomSamplingTestStruct(
        generation_config=get_multinomial_temperature_and_num_return_sequence(),
        prompts=["What is location of"],
        ref_texts=[
            [
                " the exact same image?\nI've tried multiple times to find it, but I'm still not sure. I am sure it's the exact same",
                " your new house?\nAnywhere that has a GPS. It will be up to you.",
                " your cat?  He is more likely to be on the floor with him.\nTalduck"
            ]
        ],
    ),
    RandomSamplingTestStruct(
        generation_config=get_multinomial_all_parameters(),
        prompts=["Tell me something about UAE"],
        ref_texts=get_current_plarform_ref_texts({
            "linux": [
                [
                    " and how it's not like we're all in the same boat right now lol (or even close) 😂😁! Just curious :) If",
                    "?  You are my country... so what does our military do here?? What am i missing out on?? And why don't u tell us?",
                    "?\nThe U.S government has been doing quite well with foreign-made aircraft for many years under US administration....and they have very good reasons",
                    "? I think that is a bit of an anomaly, but you might want to ask yourself this question: Where can some young people from Dubai or Bahrain",
                ]
            ],
            "win32": [
                [
                    "? I think that is a bit of an anomaly, especially since there aren't many Americans living here (like us). What makes you say they've",
                    "?  You are my country... so what does our future have to do with your problems?? \U0001f609\U0001f608\U0001f495 \U0001f5a4\ufffd",
                    "?\nThe U.S government has been doing quite well for decades now when compared strictly directly or indirectly as regards security issues.. They even made some",
                    " and how it's not like we're all in the same boat either! We had such fun meeting each other at different times this past summer :) It",
                ]
            ],
        }),
    ),
    RandomSamplingTestStruct(
        generation_config=get_multinomial_temperature_and_presence_penalty(),
        prompts=["What is OpenVINO?"],
        ref_texts=[
            [
                "\n\nOpenVINO is a software development platform developed by OpenVINO, Inc., which uses a RESTful API for server-side web applications"
            ]
        ],
    ),
    RandomSamplingTestStruct(
        generation_config=get_multinomial_temperature_and_frequence_penalty(),
        prompts=["What is OpenVINO?"],
        ref_texts=[
            [
                "\n\nOpenVINO is a software development platform developed by OpenVINO, Inc., which offers the Linux-based platform. OpenVINO's"
            ]
        ],
    ),
    RandomSamplingTestStruct(
        generation_config=get_greedy_with_penalties(),
        prompts=["What is OpenVINO?"],
        ref_texts=[
            [
                "\nOpenVINO is a software that allows users to create and manage their own virtual machines. It's designed for use with Windows, Mac OS X"
            ]
        ],
    ),
    RandomSamplingTestStruct(
        generation_config=get_multinomial_max_and_min_token(),
        prompts=["What is OpenVINO?"],
        ref_texts=get_current_plarform_ref_texts({
            "linux": [
                [
                    "\nOpenVINO is a Linux distro. It's not as simple as using the Linux distro itself. OpenVINO is essentially a dist",
                    "\nOpenVINO is an open-source open-source software that allows anyone to work with a virtual machine, from a smartphone to an iPhone,",
                    "\n\nOpenVINO is a social networking tool. OpenVINO is a free virtualization service that works at scale. The tool provides the ability",
                ]
            ],
            "win32": [
                [
                    "\nOpenVINO is the latest addition to the OpenVINO series of platforms. OpenVINO is an open source software development framework for all platforms",
                    "\nOpenVINO is a browser-based virtual assistant that enables developers and developers to quickly communicate with their own virtual machines. Using this virtual assistant,",
                    "\n\nOpenVINO is a program designed to help you find the best open source open source software. The program, which is a lightweight package and",
                ]
            ],
        }),
    ),
]


@pytest.mark.precommit
@pytest.mark.skip(reason="Random sampling results are non deterministic due to: discrete_distribution impl depends on platform, model inference results may depend on CPU. Test passes on CI but fails locally.")
@pytest.mark.parametrize("test_struct", RANDOM_SAMPLING_TEST_CASES,
        ids=["multinomial_temperature",
             "multinomial_temperature_and_top_p",
             "multinomial_temperature_and_top_k",
             "multinomial_temperature_top_p_and_top_k",
             "multinomial_temperature_and_repetition_penalty",
             "multinomial_temperature_and_num_return_sequence",
             "multinomial_all_parameters",
             "multinomial_temperature_and_presence_penalty",
             "multinomial_temperature_and_frequence_penalty",
             "greedy_with_penalties",
             "multinomial_max_and_min_token"])
def test_individual_generation_configs_random(tmp_path, test_struct: RandomSamplingTestStruct):
    generation_config = test_struct.generation_config

    prompts = test_struct.prompts
    generation_config.rng_seed = 0
    generation_configs = [generation_config]
    model_id : str = "facebook/opt-125m"
    model, hf_tokenizer = get_model_and_tokenizer(model_id, use_optimum=True)

    model_path : Path = tmp_path / model_id
    save_ov_model_from_optimum(model, hf_tokenizer, model_path)

    generate_and_compare_with_reference_text(model_path, prompts, test_struct.ref_texts, generation_configs, DEFAULT_SCHEDULER_CONFIG)



@pytest.mark.precommit
@pytest.mark.parametrize("sampling_config", [get_greedy(), get_beam_search(), get_multinomial_all_parameters()])
def test_post_oom_health(tmp_path, sampling_config):
    generation_config = sampling_config
    generation_config.ignore_eos = True
    generation_config.max_new_tokens = 1000000

    scheduler_config = get_scheduler_config()
    # Low cache size to trigger OOM quickly
    scheduler_config.num_kv_blocks = 10
    generation_configs = [generation_config]
    model_id : str = "facebook/opt-125m"
    model, hf_tokenizer = get_model_and_tokenizer(model_id, use_optimum=True)

    model_path : Path = tmp_path / model_id
    save_ov_model_from_optimum(model, hf_tokenizer, model_path)

    pipe = ContinuousBatchingPipeline(model_path.absolute().as_posix(), Tokenizer(model_path.absolute().as_posix(), {}), scheduler_config, "CPU", {})
    # First run should return incomplete response
    output = pipe.generate(["What is OpenVINO?"], generation_configs)
    assert (len(output))
    assert(len(output[0].m_generation_ids))
    # Same for the second run, here we want to make sure the cleanup works and we have free blocks after recent OOM
    output = pipe.generate(["What is OpenVINO?"], generation_configs)
    assert (len(output))
    assert(len(output[0].m_generation_ids))
    del pipe
    shutil.rmtree(model_path)