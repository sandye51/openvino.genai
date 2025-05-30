#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai
import librosa


def read_wav(filepath):
    raw_speech, samplerate = librosa.load(filepath, sr=16000)
    return raw_speech.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to the model directory")
    parser.add_argument("wav_file_path", help="Path to the WAV file")
    parser.add_argument("device", nargs="?", default="CPU", help="Device to run the model on (default: CPU)")
    args = parser.parse_args()

    pipe = openvino_genai.WhisperPipeline(args.model_dir, args.device)

    config = pipe.get_generation_config()
    config.max_new_tokens = 100  # increase this based on your speech length
    # 'task' and 'language' parameters are supported for multilingual models only
    config.language = "<|en|>"  # can switch to <|zh|> for Chinese language
    config.task = "transcribe"
    config.return_timestamps = True

    # Pipeline expects normalized audio with Sample Rate of 16kHz
    raw_speech = read_wav(args.wav_file_path)
    result = pipe.generate(raw_speech, config)

    print(result)

    if result.chunks:
        for chunk in result.chunks:
            print(f"timestamps: [{chunk.start_ts:.2f}, {chunk.end_ts:.2f}] text: {chunk.text}")


if "__main__" == __name__:
    main()
