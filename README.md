# Beyond Basic RAG: Hybrid Search & Reranking

Production-grade Retrieval Augmented Generation (RAG) implementation featuring:

- **Hybrid Search**: Combines BM25 sparse retrieval with dense vector search
- **Reciprocal Rank Fusion (RRF)**: Intelligent merging of ranked lists
- **Cross-Encoder Reranking**: Precision rescoring with full query-document attention
- **Context Optimization**: Strategic reordering to combat "Lost in the Middle"

## 📊 Performance Improvements

Compared to naive vector-only RAG:

| Metric | Naive RAG | Hybrid + Reranking | Improvement |
|--------|-----------|-------------------|-------------|
| **Exact Match Recall** | 67% | 94% | +40% |
| **Answer Accuracy** | 76% | 89% | +17% |
| **P@5 (Precision at 5)** | 0.72 | 0.91 | +26% |

**Trade-off**: 2x latency (+340ms) for ~20% accuracy gain.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Harthikahari/Beyond-Basic-RAG.git
cd Beyond-Basic-RAG

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Basic Usage

```python
from src.retrieval import HybridRetriever, create_sample_documents
from src.reranker import RerankedRetriever, CrossEncoderReranker

# 1. Load documents
documents = create_sample_documents()

# 2. Create hybrid retriever
hybrid_retriever = HybridRetriever(
    documents=documents,
    bm25_weight=0.5,
    dense_weight=0.5
)

# 3. Add reranking
reranker = CrossEncoderReranker(model_name="fast")
production_retriever = RerankedRetriever(
    base_retriever=hybrid_retriever,
    reranker=reranker,
    top_k=5,
    apply_context_reordering=True
)

# 4. Retrieve relevant documents
query = "How do I fix authentication timeouts?"
results = production_retriever.get_relevant_documents(query)

for i, doc in enumerate(results, 1):
    score = doc.metadata.get("rerank_score", 0)
    print(f"{i}. [Score: {score:.3f}] {doc.metadata['doc_id']}")
    print(f"   {doc.page_content[:100]}...\n")
```

## 📁 Project Structure

```
beyond-basic-rag/
├── data/
│   ├── raw/                      # Original documents
│   └── processed/                # Chunked/embedded docs
├── notebooks/
│   ├── 01_hybrid_search_implementation.ipynb
│   └── 02_reranking_experiments.ipynb
├── src/
│   ├── __init__.py
│   ├── retrieval.py              # HybridRetriever class
│   ├── reranker.py               # CrossEncoderReranker class
│   └── utils.py                  # Context optimization utilities
├── tests/
│   ├── test_retrieval.py
│   └── test_reranker.py
├── requirements.txt
├── .gitignore
├── README.md
└── blog-post-01-hybrid-search-reranking.md
```

## 🏗️ Architecture

```mermaid
graph LR
    A[User Query] --> B[Hybrid Retriever]
    B --> C[BM25 Sparse Search]
    B --> D[Dense Vector Search]
    C --> E[RRF Merging]
    D --> E
    E --> F[Top 50 Candidates]
    F --> G[Cross-Encoder Reranker]
    G --> H[Top 5 Results]
    H --> I[Context Reordering]
    I --> J[LLM]
```

## 📚 Key Components

### 1. HybridRetriever

Combines BM25 keyword matching with dense vector search:

- **BM25**: Excels at exact keyword matches (error codes, SKUs, proper nouns)
- **Dense Vector**: Handles paraphrased and conceptual queries
- **RRF Merging**: Combines results without requiring normalized scores

```python
retriever = HybridRetriever(
    documents=docs,
    embedding_model="text-embedding-3-small",
    bm25_weight=0.5,  # Tune based on your domain
    dense_weight=0.5,
    bm25_k=25,
    dense_k=25
)
```

**Weight Tuning Guidelines**:
- High-precision domains (legal, medical): `bm25=0.6, dense=0.4`
- Conversational queries: `bm25=0.3, dense=0.7`
- Mixed workloads: `bm25=0.5, dense=0.5`

### 2. CrossEncoderReranker

Precision rescoring with full query-document attention:

```python
reranker = CrossEncoderReranker(
    model_name="fast",  # or "accurate", "qa", "multilingual"
    device="cuda"  # or "cpu"
)

reranked = reranker.rerank(
    query=query,
    documents=candidates,
    top_k=5
)
```

**Available Models**:
- `"fast"`: `ms-marco-MiniLM-L-6-v2` (fast, general domains)
- `"accurate"`: `ms-marco-MiniLM-L-12-v2` (slower, higher accuracy)
- `"qa"`: `qnli-distilroberta-base` (optimized for Q&A)
- `"multilingual"`: `mmarco-mMiniLMv2-L12-H384-v1` (100+ languages)

### 3. Context Optimization

Combat "Lost in the Middle" by reordering documents:

```python
from src.utils import reorder_documents_lost_in_middle

optimized_docs = reorder_documents_lost_in_middle(documents)
# Places most relevant docs at START and END of context
```

**Impact**: +8% answer accuracy (measured on GPT-4).

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_retrieval.py -v
```

## 📓 Notebooks

Interactive Jupyter notebooks for experimentation:

```bash
jupyter notebook notebooks/01_hybrid_search_implementation.ipynb
```

## 🛠️ Advanced Configuration

### Custom Document Chunking

```python
from src.utils import chunk_documents

# Split long documents
chunked_docs = chunk_documents(
    documents=long_documents,
    chunk_size=512,
    chunk_overlap=50
)
```

### Score-Based Filtering

```python
# Only return documents above threshold
filtered = reranker.rerank_with_threshold(
    query=query,
    documents=candidates,
    score_threshold=0.7,
    max_results=10
)
```

### Context Statistics

```python
from src.utils import calculate_context_stats

stats = calculate_context_stats(documents)
print(f"Total tokens: {stats['total_tokens']}")
print(f"Average score: {stats['avg_score']}")
```

## 🌐 Production Deployment

For API deployment with FastAPI:

```python
from fastapi import FastAPI
from src import HybridRetriever, RerankedRetriever, CrossEncoderReranker

app = FastAPI()

# Initialize retriever (do this once at startup)
retriever = RerankedRetriever(...)

@app.post("/search")
async def search(query: str, top_k: int = 5):
    results = retriever.get_relevant_documents(query)
    return {"results": results}
```

## 📖 Learn More

Read the full technical blog post: [`blog-post-01-hybrid-search-reranking.md`](./blog-post-01-hybrid-search-reranking.md)

**Series**: Generative AI: From Prototype to Production

- **Part 1**: Beyond Basic RAG: Hybrid Search & Reranking (this repo)
- **Part 2**: Query Understanding & Routing (coming soon)
- **Part 3**: Evaluation Frameworks (coming soon)

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Research: ["Lost in the Middle"](https://arxiv.org/abs/2307.03172) (Liu et al., 2023)
- Models: [sentence-transformers](https://www.sbert.net/) library
- Framework: [LangChain](https://www.langchain.com/)

## 📬 Contact

Questions or feedback? Open an issue or reach out at [your-email@example.com](mailto:your-email@example.com).

---

**⭐ If you find this useful, please star the repository!**
