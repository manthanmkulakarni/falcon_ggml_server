#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "crow.h"
#include <iostream>
#include <string>
#include "falcon_common.h"
#include "libfalcon.h"
#include "build-info.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <signal.h>
#include <shellapi.h>
#endif

static console_state con_st;
static falcon_context ** g_ctx;

std::unordered_map<std::string, int> special_tokens_ids = {
     {">>TITLE<<", 0},
     {">>ABSTRACT<<", 1},
     {">>INTRODUCTION<<", 2},
     {">>SUMMARY<<", 3},
     {">>COMMENT<<", 4},
     {">>ANSWER<<", 5},
     {">>QUESTION<<", 6},
     {">>DOMAIN<<", 7},
     {">>PREFIX<<", 8},
     {">>SUFFIX<<", 9},
     {">>MIDDLE<<", 10},
     {"<|endoftext|>", 11},
};




static bool is_interacting = false;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!is_interacting) {
            is_interacting=true;
        } else {
            console_cleanup(con_st);
            printf("\n");
            falcon_print_timings(*g_ctx);
            _exit(130);
        }
    }
}
#endif

struct FalconLLM{
    gpt_params params;
    falcon_context * ctx;
    std::vector<falcon_token> embd_inp;
};

void initBackend(){
    falcon_init_backend();
}

falcon_context * loadModel(FalconLLM llm){
    return falcon_init_from_gpt_params(llm.params);
}
std::vector<falcon_token> tokenizeTheInput(falcon_context * ctx, std::string prompt){
    std::vector<falcon_token> embd_inp = ::falcon_tokenize(ctx, prompt, false);
    return embd_inp;
}

std::string generateOutput(std::vector<falcon_token> embd_inp, gpt_params params, falcon_context * ctx){
    const int n_ctx = falcon_n_ctx(ctx);
    std::string prediction_result = "";

    if ((int) embd_inp.size() > n_ctx - 4) {
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return "Prompt too long failed to generate output";
    }
    falcon_prepare_buffers(ctx, params.n_batch, embd_inp.size()+1);

    

    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token token to recalculate the cached logits


    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size() || params.instruct) {
        params.n_keep = (int)embd_inp.size();
    }

    // prefix & suffix for instruct mode and finetune modes
    std::vector<falcon_token> inp_system = {}; // system prompt
    std::vector<falcon_token> inp_pfx = {}; // prefix to user prompt
    std::vector<falcon_token> inp_sfx = {}; // suffix to user prompt
    std::vector<std::vector<falcon_token>> stopwords = {};

    if (params.stopwords.size())
    {
        std::string sw_token_str;
        std::vector<std::string> inp_system;
        std::stringstream stopwordStream(params.stopwords);
        std::vector<std::string> sw_token_list;
        while(std::getline(stopwordStream, sw_token_str, ',')) {
            sw_token_list.push_back(sw_token_str);
        }

        for (auto& sw_token : sw_token_list) {
            auto stopword_seq = ::falcon_tokenize(ctx, sw_token, false);
            stopwords.push_back(stopword_seq);
        }
    }


    std::vector<falcon_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);



    bool input_echo           = true;

    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;


    std::vector<falcon_token> embd;

    

    // do one empty run to warm up the model
    {
        const std::vector<falcon_token> tmp = { falcon_token_bos(), };
        falcon_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads,0);
        llama_reset_timings(ctx);
    }

    printf("Entering the generation loop\n");
    while (n_remain != 0) 
    {

      fflush(stdout);
        // predict
        if (embd.size() > 0) 
        {
            // Note: n_ctx - 4 here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            auto max_embd_size = n_ctx - 4;
            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int)embd.size() > max_embd_size) {
                auto skipped_tokens = embd.size() - max_embd_size;
                printf("<<input too long: skipped %zu token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                fflush(stdout);
                embd.resize(max_embd_size);
            }

            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            if (n_past + (int) embd.size() > n_ctx) {
                const int n_left = n_past - params.n_keep;

                n_past = params.n_keep;

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());

            }


            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always
            for (int i = 0; i < (int) embd.size(); i += params.n_batch) 
            {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }
                int debug_timings = params.debug_timings;
                if (n_remain == 1 && debug_timings == 2) debug_timings = 3; // we have no access to the last information in eval()
                if (falcon_eval(ctx, &embd[i], n_eval, n_past, params.n_threads,debug_timings)) {
                    fprintf(stderr, "%s : failed to eval\n", __func__);
                    return "Failed to generate output";
                }
                n_past += n_eval;
            }

        } // if (embd.size() > 0)

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed)  // sample for next generation
        {
            // out of user input, sample next token
            const float   temp            = params.temp;
            const int32_t top_k           = params.top_k <= 0 ? falcon_n_vocab(ctx) : params.top_k;
            const float   top_p           = params.top_p;
            const float   tfs_z           = params.tfs_z;
            const float   typical_p       = params.typical_p;
            const int32_t repeat_last_n   = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
            const float   repeat_penalty  = params.repeat_penalty;
            const float   alpha_presence  = params.presence_penalty;
            const float   alpha_frequency = params.frequency_penalty;
            const int     mirostat        = params.mirostat;
            const float   mirostat_tau    = params.mirostat_tau;
            const float   mirostat_eta    = params.mirostat_eta;
            const bool    penalize_nl     = params.penalize_nl;


            falcon_token id = 0;

            {
                auto logits  = falcon_get_logits(ctx);
                auto n_vocab = falcon_n_vocab(ctx);

                // Apply params.logit_bias map
                for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
                    logits[it->first] += it->second;
                }

                std::vector<falcon_token_data> candidates;
                candidates.reserve(n_vocab);
                for (falcon_token token_id = 0; token_id < n_vocab; token_id++) {
                    candidates.emplace_back(falcon_token_data{token_id, logits[token_id], 0.0f});
                }

                falcon_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

                // Apply penalties
                float nl_logit = logits[falcon_token_nl()];
                auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
                llama_sample_repetition_penalty(ctx, &candidates_p,
                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                    last_n_repeat, repeat_penalty);
                llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                    last_n_repeat, alpha_frequency, alpha_presence);
                if (!penalize_nl) {
                    logits[falcon_token_nl()] = nl_logit;
                }

                if (temp <= 0) {
                    // Greedy sampling
                    id = llama_sample_token_greedy(ctx, &candidates_p);
                } else {
                    if (mirostat == 1) {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        const int mirostat_m = 100;
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                    } else if (mirostat == 2) {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                    } else {
                        // Temperature sampling
                        llama_sample_top_k(ctx, &candidates_p, top_k, 1);
                        llama_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
                        llama_sample_typical(ctx, &candidates_p, typical_p, 1);
                        llama_sample_top_p(ctx, &candidates_p, top_p, 1);
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token(ctx, &candidates_p);
                    }
                }
                // printf("`%d`", candidates_p.size);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            }

            // add it to the context
            embd.push_back(id);

            // echo this to console
            input_echo = true;

            // decrement remaining sampling budget
            --n_remain;
        } else 
        {
            // some user input remains from prompt or interaction, forward it to processing
            if (n_past == 0)
            {
                if (params.enclose_finetune && (inp_pfx.size() || inp_sfx.size()) )
                {
                    // enclose finetune - non interactive mode
                    if (inp_pfx.size())
                    {
                        embd_inp.insert(embd_inp.begin(), inp_pfx.begin(), inp_pfx.end());
                    }
                    if (inp_system.size())
                    {
                        embd_inp.insert(embd_inp.begin(), inp_system.begin(), inp_system.end());
                    }
                    if (inp_sfx.size())
                    {
                        embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
                    }
                }
            }
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        bool stopword_fulfilled = false;
        // stopwords
        if (!embd.empty()) 
        {
            
            for (const auto& stopword : stopwords) {
                if (embd.size() < stopword.size()) {
                    continue; // if embd is smaller than stopword, skip this iteration
                }
                stopword_fulfilled = true; // initially assume stopword is fulfilled
                for (size_t i = 0; i < stopword.size(); ++i) {
                    if (embd[embd.size() - i - 1] != stopword[stopword.size() - i - 1]) {
                        stopword_fulfilled = false;
                        break;
                    }
                }
                if (stopword_fulfilled) {
                    break;
                }
            }
            if (stopword_fulfilled) 
            {
                if (params.verbose_prompt) 
                    fprintf(stderr, " [stopword]\n");
                break;
            }
        }
        // display text
        if (input_echo) 
        {
            std::string placeholder; 
            for (auto id : embd) {
                if (params.instruct && id == falcon_token_eos()) {
                    id = falcon_token_nl();
                }
                placeholder = falcon_token_to_str(ctx, id);
                fprintf(stdout, "%s", placeholder);
                fflush(stdout);
                prediction_result += placeholder;

            }
            fflush(stdout);
        }

        
    }
    return prediction_result;
}
    crow::SimpleApp app;
    FalconLLM llm;

int main(int argc, char ** argv) {
    


    std::string prompt = "who are you";
    if (gpt_params_parse(argc, argv, llm.params) == false) {
        return 1;
    }


    

    initBackend();
    g_ctx = &llm.ctx;

    // load the model and apply lora adapter, if any
    llm.ctx = loadModel(llm);

    if (llm.ctx == NULL) {
        printf("%s: error: unable to load model\n", __func__);
        return 1;
    }


    #if defined(GGML_USE_CUBLAS)
    // wait for cublas and show device information
    {
        ggml_cuda_print_gpu_status(ggml_cuda_get_system_gpu_status(),true);
    }
    #endif

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s\n",
                 falcon_print_system_info(llm.params.n_threads, std::thread::hardware_concurrency()));
    }


    CROW_ROUTE(app, "/")
    ([](){
        return "Hello, World!";
    });


    CROW_ROUTE(app, "/query")
    .methods("POST"_method)
    ([](const crow::request& req){
        auto x = crow::json::load(req.body);


        llm.params.prompt = x["prompt"].s();
        llm.params.n_predict = x["n"].i();

        llm.embd_inp = tokenizeTheInput(llm.ctx, llm.params.prompt);
        std::string generated_result = generateOutput(llm.embd_inp, llm.params, llm.ctx);
        falcon_print_timings(llm.ctx);

        crow::json::wvalue response;
        response["result"] = generated_result;

        // Serialize the JSON response to a string
        std::string responseString = response.dump();
        return crow::response(responseString);
        
    });
    // Start the server
    app.port(8089).multithreaded().run();

    // tokenize the prompt
    

    



    

    
    llama_free(llm.ctx);

    return 0;
}





