# Gemma2

## Download Gemma2

Download Gemma2 GGUF model
```
wget https://huggingface.co/google/gemma-2-2b-GGUF/resolve/main/2b_pt_v2.gguf -O gemma2-2b.gguf
wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf -O qwen2.5-0.5b-fp16.gguf
```

If you have unauthorized access to the model, you can git clone the repo instead.
```
git clone https://huggingface.co/google/gemma-2-2b-GGUF
mv gemma-2-2b-GGUF/2b_pt_v2.gguf gemma2-2b.gguf
```

## Run Gemma2
```bash
cd build/bin
./gemma2-simple -m ../../examples/simple-gemma2/gemma2-2b.gguf -p "Hello my name is"
```

## Run Qwen2
```bash
cd build/bin
./gemma2-simple -m ../../examples/simple-gemma2/qwen2.5-0.5b-fp16.gguf -p "Hello my name is"
```