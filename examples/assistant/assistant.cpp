// Voice assistant with wake word detection and ollama integration
#include <cstdio>
#include <string>
#include <chrono>
#include <thread>
#include <vector>
#include <fstream>
#include <sstream>
#include <curl/curl.h>

#include "common-sdl.h"
#include "common.h"
#include "whisper.h"

#include <cassert>

// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    bool translate     = false;
    bool no_fallback   = false;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = false;
    bool tinydiarize   = false;
    bool save_audio    = false;
    bool use_gpu       = true;
    bool flash_attn    = false;

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
    std::string fname_out;
};

// 用于存储ollama响应的结构体
struct OllamaResponse {
    std::string data;
};

// CURL写回调
size_t WriteCallback(void* contents, size_t size, size_t nmemb, OllamaResponse* response) {
    size_t total_size = size * nmemb;
    response->data.append((char*)contents, total_size);
    return total_size;
}

// 调用ollama API
std::string call_ollama(const std::string& prompt) {
    CURL* curl = curl_easy_init();
    OllamaResponse response;
    std::string result;
    
    if(curl) {
        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        
        // 准备JSON请求体
        std::string json_body = "{\"model\": \"qwen2.5\", \"prompt\": \"" + prompt + "\"}";
        
        curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:11434/api/generate");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        
        CURLcode res = curl_easy_perform(curl);
        
        if(res == CURLE_OK) {
            // 解析响应
            std::istringstream response_stream(response.data);
            std::string line;
            while (std::getline(response_stream, line)) {
                // 每行都是一个JSON对象，我们只需要提取response字段
                if (line.find("\"response\":") != std::string::npos) {
                    size_t start = line.find("\"response\":\"") + 11;
                    size_t end = line.find("\"", start);
                    if (start != std::string::npos && end != std::string::npos) {
                        result += line.substr(start, end - start);
                    }
                }
            }
        } else {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        }
        
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
    
    return result;
}

// 语音助手参数
struct assistant_params {
    whisper_params whisper; // 继承whisper参数
    
    std::string wake_word = "test";
    int silence_threshold_ms = 1000; // 静默阈值，用于分段
    bool is_active = false; // 是否被唤醒
    
    std::string current_segment; // 当前语音段落
    int64_t last_speech_time = 0; // 上次检测到语音的时间
};

// 检查是否包含唤醒词
bool contains_wake_word(const std::string& text, const std::string& wake_word) {
    return text.find(wake_word) != std::string::npos;
}

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help          [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "            --step N        [%-7d] audio step size in milliseconds\n",                params.step_ms);
    fprintf(stderr, "            --length N      [%-7d] audio length in milliseconds\n",                   params.length_ms);
    fprintf(stderr, "            --keep N        [%-7d] audio to keep from previous step in ms\n",         params.keep_ms);
    fprintf(stderr, "  -c ID,    --capture ID    [%-7d] capture device ID\n",                              params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N  [%-7d] maximum number of tokens per audio chunk\n",       params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N   [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] voice activity detection threshold\n",           params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n",                   params.freq_thold);
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n",                                params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "\n");
}

bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                  arg == "--step")          { params.step_ms       = std::stoi(argv[++i]); }
        else if (                  arg == "--length")        { params.length_ms     = std::stoi(argv[++i]); }
        else if (                  arg == "--keep")          { params.keep_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"    || arg == "--capture")      { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"   || arg == "--max-tokens")   { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"   || arg == "--audio-ctx")    { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-vth"  || arg == "--vad-thold")    { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth"  || arg == "--freq-thold")   { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-l"    || arg == "--language")     { params.language      = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")        { params.model         = argv[++i]; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

int main(int argc, char ** argv) {
    assistant_params params;
    
    // 初始化whisper参数
    if (whisper_params_parse(argc, argv, params.whisper) == false) {
        return 1;
    }

    // 初始化音频
    audio_async audio(params.whisper.length_ms);
    if (!audio.init(params.whisper.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }
    
    audio.resume();

    // 初始化whisper
    struct whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context * ctx = whisper_init_from_file_with_params(
        params.whisper.model.c_str(), cparams);

    std::vector<float> pcmf32(WHISPER_SAMPLE_RATE * 30, 0.0f);
    
    printf("[System started - waiting for wake word '%s']\n", params.wake_word.c_str());
    fflush(stdout);

    bool is_running = true;
    while (is_running) {
        // 获取音频
        audio.get(2000, pcmf32);
        
        // VAD检测
        if (::vad_simple(pcmf32, WHISPER_SAMPLE_RATE, 1000, 
            params.whisper.vad_thold, params.whisper.freq_thold, false)) {
            
            params.last_speech_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            // 运行whisper识别
            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
            wparams.print_progress = false;
            wparams.print_special = false;
            wparams.print_realtime = false;
            wparams.print_timestamps = false;
            wparams.translate = false;
            wparams.language = "en";
            wparams.n_threads = params.whisper.n_threads;

            if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
                fprintf(stderr, "Failed to process audio\n");
                continue;
            }

            // 获取识别结果
            const int n_segments = whisper_full_n_segments(ctx);
            for (int i = 0; i < n_segments; ++i) {
                const char * text = whisper_full_get_segment_text(ctx, i);
                
                if (!params.is_active) {
                    // 检查唤醒词
                    if (contains_wake_word(text, params.wake_word)) {
                        params.is_active = true;
                        printf("\n[Assistant activated]\n");
                        continue;
                    }
                } else {
                    // 累积当前语音段落
                    params.current_segment += text;
                    printf("%s", text);
                    fflush(stdout);
                }
            }
        } else {
            // 检查是否需要处理当前段落
            int64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
                
            if (params.is_active && 
                (current_time - params.last_speech_time) > params.silence_threshold_ms && 
                !params.current_segment.empty()) {
                
                printf("\n[Processing: %s]\n", params.current_segment.c_str());
                
                // 调用ollama并打印响应
                std::string response = call_ollama(params.current_segment);
                printf("\n[Assistant]: %s\n\n", response.c_str());
                fflush(stdout);
                
                // 重置状态
                params.current_segment.clear();
                params.is_active = false;
                
                printf("[Waiting for wake word '%s']\n", params.wake_word.c_str());
                fflush(stdout);
            }
        }

        // 处理Ctrl+C
        is_running = sdl_poll_events();
    }

    whisper_free(ctx);
    return 0;
} 