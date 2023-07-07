ggllm.cpp is a ggml-based tool to run quantized Falcon Models on CPU and GPU



**Features that differentiate from llama.cpp for now:**
- Support for Falcon 7B and 40B models (inference)
- Fully automated GPU offloading based on available and total VRAM
- Higher efficiency in VRAM usage when using batched processing (more layers being offloaded)
- 16 bit cuBLAs support (takes half the VRAM for those operations)
- Improved loading screen and visualization
- New tokenizer with regex emulation and BPE merge support
- Finetune auto-detection and integrated syntax support (Just load OpenAssistant 7/40 add `-ins` for a chat or `-enc -p "Question"` and optional -sys "System prompt")
- Stopwords support (-S)
- Optimized RAM and VRAM calculation with batch processing support up to 8k
- More command line parameter options (like disabling GPUs)
- Current Falcon inference speed on consumer GPU: up to 54+ tokens/sec for 7B-4-5bit and 18-25 tokens/sec for 40B 3-6 bit, roughly 38/sec and 16/sec at at 1000 tokens generated
  

**Old model support**  
If you use GGML type models (file versions 1-4) you need to place tokenizer.json into the model directory ! (example: https://huggingface.co/OpenAssistant/falcon-40b-sft-mix-1226/blob/main/tokenizer.json)  
If you use updated model binaries they are file version 10+ and called "GGCC", those do not need the load and convert that json file  

**How to just run it?**
1) In most cases you will want to choose a good instruct model, currently the best tunes are from OpenAssist.  
2) Falcon 40B is great even at Q2_K (2 bit) quantization, very good multilingual and reasoning quality.
3) After downloading (and/or converting/quantizing) your model you launch falcon_main with `-enc -p "Your question"` or with `-ins` for multiple questions
4) From there on you can dive into more options, there is a lot to change and optimize.

**The Bloke features fine tuned weights in ggcc v10 with various quantization options:**  
https://huggingface.co/TheBloke/falcon-40b-sft-mix-1226-GGML (OpenAssistant 40B)
https://huggingface.co/TheBloke/falcon-40b-instruct-GGML  
https://huggingface.co/TheBloke/WizardLM-Uncensored-Falcon-40B-GGML  
https://huggingface.co/TheBloke/falcon-7b-instruct-GGML  
https://huggingface.co/TheBloke/WizardLM-Uncensored-Falcon-7B-GGML  
  
**The official HF models are here:**  
https://huggingface.co/tiiuae/falcon-40b/  
https://huggingface.co/tiiuae/falcon-7b/  
https://huggingface.co/tiiuae/falcon-40b-instruct  
https://huggingface.co/tiiuae/falcon-7b-instruct  


**How to compile ggllm.cpp:**
1) Recommended with cmake: (change the CUBLAS flag to 0 to disable CUDA requirements and support)
```
git clone https://github.com/manthanmkulakarni/falcon_ggml_server.git
cd ggllm.cpp
cd ./Crow && mkdir build && cd build && cmake .. -DCROW_BUILD_EXAMPLES=OFF -DCROW_BUILD_TESTS=OFF && make install
cd ..
rm -rf build; mkdir build; cd build
# if you do not have cuda in path:
export PATH="/usr/local/cuda/bin:$PATH"
# in case of problems, this sometimes helped
#export CPATH="/usr/local/cuda/targets/x86_64-linux/include:"
#export LD_LIBRARY_PATH="/usr/local/cuda/lib64:"
cmake -DLLAMA_CUBLAS=1 -DCUDAToolkit_ROOT=/usr/local/cuda/ ..  
cmake --build . --config Release
# find the binaries in ./bin
# falcon_main
```
2) Building with make (fallback):
```
export LLAMA_CUBLAS=1;
# if you do not have "nvcc" in your path:
# export PATH="/usr/local/cuda/bin:$PATH"
make falcon_main falcon_quantize falcon_perplexity
```

3) Installing on WSL (Windows Subsystem for Linux)
```
# Use --no-mmap in WSL OR copy the model into a native directory (not /mnt/) or it will get stuck loading (thanks @nauful)
#Choose a current distro:
wsl.exe --list --online
wsl --install -d distro
# cmake 3.16 is required and the cuda toolset
# If you run an old distro you can upgrade (like apt update; apt upgrade; apt full-upgrade; pico /etc/apt/sources.list/; apt update; apt upgrade; apt full-upgrade; apt autoremove; lsb_release -a); then wsl --shutdown and restart it
# install cuda WSL toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update; apt-get -y install cuda
# you might need to add it to your path:
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda/bin:$PATH"
# now start with a fresh cmake and all should work 
```

**CUDA Optimizing inference speed**
- Thread count will be optimal between 1 and 8. Start with `-t 2` 
- For huge prompts n_batch can speed up processing 10-20 times but additional VRAM of 500-1700 MB is required. That's `-b 512` 
- Multi GPU systems can benefit from single GPU processing when the model is small enough. That's `--override-max-gpu 1`  
- Multi GPU systems with different GPUs benefit from custom tensor splitting to load one GPU heavier. To load the 2nd GPU stronger: `--tensor-split 1,3` `-mg 1`
- Need to squeeze a model into VRAM but 1-2 layers don't fit ? Try `--gpu-reserve-mb-main 1` to reduce reserved VRAM to 1 MB, you can use negative numbers to force VRAM swapping
- Wish to reduce VRAM usage and offload less layers? Use `-ngl 10` to only load 10 layers
- Want to dive into details ? Use `--debug-timings <1,2,3>` to get detailed statistics on performance of each operation, how and where it was performed and it's total impact

 

** To Query the result
```
<path to falcon_main> -t 2 -m <path to model.bin> -n 128 --debug-timings 0 -b 1 --ignore-eos --override-max-gpu 1 -ngl 48

curl --location 'http://localhost:8089/query' \
--header 'Content-Type: application/json' \
--data '{
    "prompt":"tell me about generative AI",
    "n":256
}'
```

CUDA sidenote:  
1) try to use less threads than you have physical processor cores 


This repo is based on https://github.com/cmp-nct/ggllm.cpp.git
