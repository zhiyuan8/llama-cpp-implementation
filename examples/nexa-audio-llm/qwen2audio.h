#pragma once

#include "whisper.h"
#include "llama.h"
#include "grammar-parser.h"
#include "common.h"
#include "common-nexa.h"

#include <string>
#include <thread>

#include "audio-projector.h"

//
// Constants
//

static const char *AUDIO_TOKEN = "<|AUDIO|>";

//
// Whisper
//

struct whisper_params
{
    // Thread and processor settings
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    int32_t n_processors = 1;
    // Time and context settings
    int32_t offset_t_ms = 0;
    int32_t offset_n = 0;
    int32_t duration_ms = 0;
    int32_t progress_step = 5;
    int32_t max_context = -1;
    int32_t max_len = 0;
    // Sampling parameters
    int32_t best_of = whisper_full_default_params(WHISPER_SAMPLING_GREEDY).greedy.best_of;
    int32_t beam_size = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH).beam_search.beam_size;
    int32_t audio_ctx = 0;
    // Threshold and penalty values
    float word_thold = 0.01f;
    float entropy_thold = 2.40f;
    float logprob_thold = -1.00f;
    float grammar_penalty = 100.0f;
    float temperature = 0.0f;
    float temperature_inc = 0.2f;
    // Boolean flags for various options
    bool debug_mode = false;
    bool translate = false;
    bool detect_language = false;
    bool diarize = false;
    bool tinydiarize = false;
    bool split_on_word = false;
    bool no_fallback = false;
    bool no_prints = false;
    bool print_special = false;
    bool print_colors = false;
    bool print_progress = false;
    bool no_timestamps = false;
    bool log_score = false;
    bool use_gpu = true;
    bool flash_attn = false;

    std::string language = "en";
    std::string prompt;
    std::string font_path = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf";
    std::string model = "models/ggml-base.en.bin";
    std::string grammar;
    std::string grammar_rule;

    // [TDRZ] speaker turn string
    std::string tdrz_speaker_turn = " [SPEAKER_TURN]"; // TODO: set from command line

    // A regular expression that matches tokens to suppress
    std::string suppress_regex;

    std::string openvino_encode_device = "CPU";

    std::string dtw = "";

    std::vector<std::string> fname_inp = {}; // Input filenames
    std::vector<std::string> fname_out = {}; // Output filenames

    grammar_parser::parse_state grammar_parsed; // Parsed grammar state
};

// Function to convert whisper_params to whisper_full_params
struct whisper_full_params get_whisper_inference_params_from_whisper_params(whisper_params &params);

// === NEXA AI ===
// Structure to hold both GPT and Whisper parameters
struct omni_params
{
    gpt_params gpt; // GPT parameters
    whisper_params whisper; // Whisper parameters
};

// Structure to hold contexts for Whisper, audio projection, and LLaMA
struct omni_context
{
    struct whisper_context *ctx_whisper; // Whisper context
    struct audio_projector *projector; // Audio projector
    struct llama_context *ctx_llama;
    struct llama_model *model; // Gemma model
};

// Function to find the position of the AUDIO_TOKEN in the prompt
size_t find_audio_token(const std::string &prompt);

// Functions for audio projection inference
struct ggml_tensor *audio_projector_inference(audio_projector &model, struct ggml_tensor *audio_feature_tensor);
struct ggml_tensor *audio_projector_inference(audio_projector &model, std::vector<float> &audio_feature_data);

// Function to parse command-line arguments into omni_params
bool omni_params_parse(int argc, char **argv, omni_params &params);

// Function to initialize the omni_context
struct omni_context *omni_init_context(omni_params &params);

// Function to free the omni_context
void omni_free(struct omni_context *ctx_omni);

// Function to process audio input
ggml_tensor *omni_process_audio(struct omni_context *ctx_omni, omni_params &params);

// Function to process the prompt with audio embedding
void omni_process_prompt(struct omni_context *ctx_omni, ggml_tensor *audio_embed, omni_params &params, const std::string &prompt);

// Function to perform the full Omni processing pipeline
void omni_process_full(struct omni_context *ctx_omni, omni_params &params);