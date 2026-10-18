# Deployment Track — Export, Quantize, Serve

> **Book Context**: Ch 8 "Making Transformers Efficient in Production" (pp. 233–272) — distillation, quantization, ONNX, pruning  
> **Core**: Extend Ch 8 with modern tools (GPTQ, AWQ, GGUF, vLLM, TGI) — run after Phases 1, 3, 4, 5, 6, 7

## Book Chapter 8 → Modern Deployment Mapping

| Book Ch 8 Section | Book Technique | Modern Equivalent | Phase |
|-------------------|----------------|-------------------|-------|
| 8.3-8.5 | Knowledge distillation | Distillation + LoRA student | 7 |
| 8.6 | Quantization (INT8) | Dynamic/Static INT8, GPTQ, AWQ | 7 |
| 8.7 | ONNX + ONNX Runtime | ONNX, TensorRT, OpenVINO | 7 |
| 8.8 | Weight pruning | Structured/unstructured, SparseML | 7 |
| — | — | GGUF (llama.cpp) | 4, 6, 9 |
| — | — | vLLM / TGI serving | 4, 6, 10 |
| — | — | Speculative decoding | 4, 10 |

## Export Formats

| Format | Tool | Models | Use Case | Book Ch |
|--------|------|--------|----------|---------|
| **ONNX** | `optimum.onnx` | All | Cross-platform, ORT | 8.7 |
| **TensorRT** | `optimum.tensorrt` | BERT, GPT | NVIDIA max perf | — |
| **OpenVINO** | `optimum.intel` | All | Intel CPU/GPU/NPU | — |
| **CoreML** | `optimum.coreml` | All | Apple Silicon | — |
| **TorchScript** | Native PyTorch | All | PyTorch serving | — |
| **GGUF** | `llama.cpp` | Llama, Mistral | CPU/edge, quantized | — |

## Quantization Methods

| Method | Bits | Tool | Quality Loss | Speedup | Models | Book Ch |
|--------|------|------|--------------|---------|--------|---------|
| **Dynamic INT8** | 8-bit | PyTorch native | ~1% | 1.5-2x | Encoder | 8.6 |
| **Static INT8** | 8-bit | `optimum`/`torch.ao` | ~2% | 2-3x | Encoder | 8.6 |
| **GPTQ** | 4-bit | `auto-gptq` | ~1-2% | 3-4x | Decoder (Llama, Mistral) | — |
| **AWQ** | 4-bit | `autoawq` | ~1% | 3-4x | Decoder | — |
| **GGUF** | 4/5/8-bit | `llama.cpp` | ~1-3% | CPU opt | Decoder | — |
| **QLoRA** | 4-bit + LoRA | `bitsandbytes` | Train-time | N/A | Decoder | — |

## Serving Frameworks

| Framework | Protocol | Batching | Streaming | Models | Best For |
|-----------|----------|----------|-----------|--------|----------|
| **vLLM** | OpenAI API | Continuous | ✓ | Llama, Mistral, GPT | High-throughput LLM |
| **TGI** | OpenAI API | Continuous | ✓ | Llama, Mistral, BLOOM | Production LLM |
| **TensorRT-LLM** | Custom | In-flight | ✓ | Llama, GPT | Max NVIDIA perf |
| **Ollama** | Custom | ✓ | ✓ | GGUF models | Local dev |
| **FastAPI + Transformers** | Custom | Manual | Manual | BERT, Encoder | Custom encoder APIs |
| **ONNX Runtime** | REST/gRPC | ✓ | Limited | BERT, DistilBERT | Cross-platform encoder |

## Quick Start

```bash
# Book Ch 8.7: Export to ONNX
python -m tracks.deployment.export \
  --phase phase7_efficiency \
  --model distilbert-base-uncased \
  --format onnx \
  --config configs/deployment/onnx-intent.yaml \
  --exp-id DEP-PH7-001

# Book Ch 8.6: Quantize (INT8 dynamic)
python -m tracks.deployment.quantize \
  --phase phase7_efficiency \
  --model bert-base-uncased \
  --method dynamic_int8 \
  --config configs/deployment/quant-int8.yaml \
  --exp-id DEP-PH7-002

# Book Ch 8.6: Quantize (INT8 static)
python -m tracks.deployment.quantize \
  --phase phase7_efficiency \
  --model bert-base-uncased \
  --method static_int8 \
  --calibration-data clinc_oos \
  --exp-id DEP-PH7-003

# Modern: GPTQ quantization (4-bit)
python -m tracks.deployment.quantize \
  --phase phase4_generation \
  --model mistralai/Mistral-7B-v0.1 \
  --method gptq \
  --bits 4 \
  --group-size 128 \
  --exp-id DEP-PH4-001

# Modern: AWQ quantization
python -m tracks.deployment.quantize \
  --phase phase4_generation \
  --model mistralai/Mistral-7B-v0.1 \
  --method awq \
  --bits 4 \
  --exp-id DEP-PH4-002

# Modern: Export GGUF for llama.cpp
python -m tracks.deployment.export \
  --phase phase4_generation \
  --model mistralai/Mistral-7B-v0.1 \
  --format gguf \
  --quantization q4_k_m \
  --exp-id DEP-PH4-003

# Serve with vLLM (modern)
vllm serve mistralai/Mistral-7B-Instruct-v0.2 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096

# Serve with TGI
docker run -p 8080:80 ghcr.io/huggingface/text-generation-inference:latest \
  --model-id mistralai/Mistral-7B-Instruct-v0.2

# Serve GGUF with llama.cpp
llama-server -m mistral-7b-instruct-q4_k_m.gguf -c 4096 --port 8080

# Serve encoder (ONNX Runtime)
python -m tracks.deployment.serve_onnx \
  --model distilbert-base-uncased-onnx \
  --port 8080
```

## Deployment Config (Per Phase)

```yaml
# Phase 1/3/7: Encoder deployment (BERT, DistilBERT, XLM-R)
deployment:
  export:
    format: onnx
    opset: 14
    dynamic_axes: true
  quantization:
    method: static_int8  # or dynamic_int8
    calibration_samples: 1000
  serving:
    framework: onnxruntime
    batch_size: 32
    workers: 4

# Phase 4/6/9/10: Decoder deployment (GPT, Llama, Mistral)
deployment:
  export:
    format: gguf  # or onnx for smaller models
    quantization: q4_k_m  # q4_k_m, q5_k_m, q8_0
  quantization:
    method: gptq  # or awq
    bits: 4
    group_size: 128
    dampening: 0.01
  serving:
    framework: vllm  # or tgi, llama.cpp
    tensor_parallel: 2
    max_model_len: 4096
    gpu_memory_utilization: 0.9
```

## Benchmarking Deployment

```bash
# Latency / throughput benchmark
python -m tracks.deployment.benchmark \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --format vllm \
  --batch-sizes 1,2,4,8,16 \
  --input-lens 128,512,1024,2048 \
  --output-lens 128,256,512 \
  --exp-id DEP-BENCH-001

# Compare formats
python -m tracks.deployment.benchmark \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --formats vllm,tgi,llama.cpp,onnx \
  --exp-id DEP-BENCH-002

# Encoder benchmark
python -m tracks.deployment.benchmark \
  --model distilbert-base-uncased \
  --format onnxruntime \
  --batch-sizes 1,8,32,64 \
  --exp-id DEP-BENCH-003
```

## Key Code Patterns

```python
# 1. ONNX Export (Book Ch 8.7)
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model = ORTModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", export=True
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model.save_pretrained("./bert-onnx")
tokenizer.save_pretrained("./bert-onnx")

# 2. Quantization with Optimum (Book Ch 8.6)
from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig

optimizer = ORTOptimizer.from_pretrained("./bert-onnx")
optimization_config = OptimizationConfig(
    optimization_level=99,  # Enable all
    enable_transformers_specific_optimizations=True,
)
optimizer.optimize(save_dir="./bert-onnx-optimized", optimization_config=optimization_config)

# 3. GPTQ Quantization (Modern)
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer

model = AutoGPTQForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantize_config={"bits": 4, "group_size": 128, "damp_percent": 0.01},
)
model.quantize(calibration_dataset)
model.save_pretrained("./mistral-7b-gptq-4bit")

# 4. GGUF Conversion (Modern)
# llama.cpp: python convert.py mistral-7b --outtype q4_k_m --outfile mistral-7b-q4_k_m.gguf

# 5. vLLM Serving (Modern)
from vllm import LLM, SamplingParams

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", tensor_parallel_size=2)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
outputs = llm.generate(prompts, sampling_params)
```

## Quality Validation After Deployment

| Check | Method | Threshold |
|-------|--------|-----------|
| **Accuracy parity** | Eval on validation set | ≤ 1% drop vs FP32 |
| **Latency** | Benchmark p50/p99 | Meet SLA |
| **Throughput** | Benchmark max throughput | Meet capacity |
| **Memory** | Peak VRAM/RAM | Fit device |
| **Numerical stability** | NaN/Inf check | Zero |

## Integration with Experiment Registry

```bash
# Log deployment artifacts
python -m tracks.deployment.export \
  --phase phase7_efficiency \
  --model distilbert-base-uncased \
  --log-artifact \
  --exp-id PH7-002
```

## Next Steps

- Add TensorRT export for encoder models
- Add speculative decoding config for vLLM
- Add multi-GPU / multi-node serving configs
- Add cost estimation (GPU hours per 1M tokens)
- CI: Automated benchmark on model updates