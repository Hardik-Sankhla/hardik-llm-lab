# Phase 6: QA & RAG — Chapter 7

> **Book**: Ch 7 "Question Answering" (pp. 189–231)  
> **Core**: Review-based QA with Haystack — retriever-reader, BM25/DPR, domain adaptation, evaluation

## Book Implementation Checklist

| Section | Book Task | Key Code | Extension Ideas |
|---------|-----------|----------|-----------------|
| 7.1 | Review QA dataset | `load_dataset("amazon_reviews_multi", "en")` | Other domains (SQuAD, HotpotQA) |
| 7.2 | Extractive QA pipeline | `pipeline("question-answering")` | Compare readers (BERT, RoBERTa, DeBERTa) |
| 7.3 | Haystack retriever-reader | `Retriever` + `Reader` + `Pipeline` | BM25 vs DPR vs dense retrievers |
| 7.4 | Retriever evaluation | Recall@k, MRR | Hard negative mining |
| 7.5 | Reader evaluation | EM, F1 on SQuAD | Domain adaptation |
| 7.6 | Domain adaptation | Fine-tune on target domain | Few-shot adaptation |
| 7.7 | Full pipeline eval | End-to-end EM/F1 | Latency vs accuracy |
| 7.8 | Beyond extractive | Generative QA, RAG | Flan-T5, Llama-2 + retrieval |

## Recommended Experiments

| Exp ID | Variant | Retriever | Reader | Dataset | Target EM |
|--------|---------|-----------|--------|---------|-----------|
| PH6-001 | Book baseline | BM25 | `deepset/roberta-base-squad2` | Amazon reviews | ≥ 0.65 |
| PH6-002 | DPR retriever | `facebook/dpr-ctx_encoder` | `deepset/roberta-base-squad2` | Amazon reviews | ≥ 0.70 |
| PH6-003 | Dense retriever | `sentence-transformers/msmarco-distilbert` | `deepset/roberta-base-squad2` | SQuAD 2.0 | ≥ 0.75 |
| PH6-004 | Domain adapted | BM25 | Fine-tuned on reviews | Amazon reviews | +5 EM |
| PH6-005 | Generative QA | BM25 | `google/flan-t5-base` | NQ/TriviaQA | EM/F1 |
| PH6-006 | RAG (book Ch 7.8) | DPR | `facebook/rag-token-base` | NQ | EM/F1 |
| PH6-007 | Modern RAG | `bge-large` + FAISS | `mistral-7b-instruct` | Custom docs | RAGAS |

## Datasets (Book + Extensions)

| Dataset | Type | Domain | Size | Book Ch |
|---------|------|--------|------|---------|
| `amazon_reviews_multi` | Extractive | Product reviews | 180K | Ch 7 |
| `squad_v2` | Extractive | Wikipedia | 130K | — |
| `natural_questions` | Open-domain | Google queries | 300K | — |
| `triviaqa` | Open-domain | Trivia | 95K | — |
| `hotpotqa` | Multi-hop | Wikipedia | 90K | — |
| `coqa` / `quac` | Conversational | Dialogue | 100K+ | — |

## Quick Start

```bash
# Book: Review QA with Haystack BM25 + RoBERTa (Ch 7.1-7.3)
python -m phase6_qa_rag.run \
  --config configs/phase6/haystack-bm25-roberta.yaml \
  --exp-id PH6-001

# Book: DPR retriever (Ch 7.3)
python -m phase6_qa_rag.run \
  --config configs/phase6/haystack-dpr-roberta.yaml \
  --exp-id PH6-002

# Book: Domain adaptation (Ch 7.6)
python -m phase6_qa_rag.run \
  --config configs/phase6/haystack-bm25-roberta.yaml \
  --adapt-domain amazon_reviews \
  --exp-id PH6-004

# Modern: Generative QA with Flan-T5 (Ch 7.8)
python -m phase6_qa_rag.run \
  --config configs/phase6/flan-t5-rag.yaml \
  --exp-id PH6-005

# Modern: Mistral-7B RAG (beyond book)
python -m phase6_qa_rag.run \
  --config configs/phase6/mistral-rag.yaml \
  --exp-id PH6-007
```

## Key Book Code Patterns to Reproduce

```python
# 1. Extractive QA pipeline (Ch 7.2)
from transformers import pipeline
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2"
)
result = qa_pipeline(question="What is the capital of France?", context="Paris is the capital of France.")
# {'answer': 'Paris', 'score': 0.99, 'start': 0, 'end': 5}

# 2. Haystack retriever-reader (Ch 7.3)
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline

document_store = InMemoryDocumentStore()
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
pipeline = ExtractiveQAPipeline(reader, retriever)

# Index documents
documents = [{"content": doc, "meta": {"name": f"doc_{i}"}} for i, doc in enumerate(corpus)]
document_store.write_documents(documents)

# Query
result = pipeline.run(query="What is...", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})

# 3. Retriever evaluation (Ch 7.4)
from haystack.utils import eval_retriever
retriever_eval = eval_retriever(retriever, label_index="eval_labels", top_k=10)
print(f"Recall@10: {retriever_eval['recall']:.3f}, MRR@10: {retriever_eval['mrr']:.3f}")

# 4. Reader evaluation (Ch 7.5)
from haystack.utils import eval_reader
reader_eval = eval_reader(reader, label_index="eval_labels", top_k=5)
print(f"EM: {reader_eval['exact_match']:.3f}, F1: {reader_eval['f1']:.3f}")

# 5. Domain adaptation (Ch 7.6)
# Fine-tune reader on target domain
reader.train(data_dir="data/amazon_reviews", train_filename="train.json", ...)
```

## Retriever Comparison (Book + Modern)

| Retriever | Type | Index Size | Speed | Recall@10 | Best For |
|-----------|------|------------|-------|-----------|----------|
| **BM25** | Sparse | Small | Fast | ~0.65 | Baseline, keyword-heavy |
| **DPR** | Dense | Medium | Medium | ~0.75 | Semantic similarity |
| **Sentence-TF** | Dense | Medium | Medium | ~0.78 | General-purpose |
| **BGE** | Dense | Medium | Medium | ~0.82 | State-of-the-art |
| **Hybrid** | Sparse+Dense | Large | Slow | ~0.85 | Best recall |

## Reader Comparison

| Reader | Architecture | SQuAD 2.0 F1 | Speed | Size |
|--------|--------------|--------------|-------|------|
| `bert-large-uncased` | BERT | 82.0 | Slow | 340M |
| `roberta-large` | RoBERTa | 86.0 | Slow | 355M |
| `deepset/roberta-base-squad2` | RoBERTa-base | 81.0 | Fast | 125M |
| `deepset/deberta-v3-large-squad2` | DeBERTa-v3 | 88.5 | Slow | 300M |
| `microsoft/deberta-v3-small` | DeBERTa-v3-small | 83.0 | Fast | 44M |

## Evaluation Metrics

| Metric | Component | What It Measures |
|--------|-----------|------------------|
| **Recall@k** | Retriever | % relevant docs in top-k |
| **MRR** | Retriever | Mean reciprocal rank |
| **EM** | Reader | Exact answer string match |
| **F1** | Reader | Token overlap with gold |
| **End-to-end EM** | Pipeline | Full system accuracy |
| **Latency** | Pipeline | ms/query (p50, p95, p99) |

## Modern RAG Extensions (Beyond Book Ch 7.8)

| Component | Options | Status |
|-----------|---------|--------|
| **Embedder** | BGE, E5, GTE, Instructor | 🟡 Planned |
| **Vector DB** | FAISS, Chroma, Qdrant, Weaviate | 🟡 Planned |
| **Re-ranker** | Cross-encoder, Cohere, Jina | 🟡 Planned |
| **Generator** | Flan-T5, Mistral, Llama-2, Zephyr | 🟡 Planned |
| **Evaluation** | RAGAS, TruLens, custom | 🟡 Planned |

## Next Phase Dependencies

- Phase 4 (Generation): Generative QA models
- Phase 5 (Summarization): Summarization as QA (long answers)
- Phase 7 (Efficiency): Distill reader, quantize retriever, ONNX export
- Tracks/PEFT: LoRA for reader domain adaptation
- Tracks/Deployment: vLLM/TGI for generative QA serving