# Intelligent Corporate Investigation Platform


## Overview

This platform demonstrates a **production‑ready Hybrid Graph ML + Graph RAG system** for risk and compliance investigations. It ingests large‑scale knowledge graphs (corporate registrations, trade data, ownership links), generates GNN‑based entity embeddings, and provides a natural language interface for investigators to uncover hidden relationships, shell companies, and tradecraft patterns.

**Key capabilities:**
- Process subgraphs from a simulated **10‑billion record knowledge graph** via read‑only API.
- Train **Graph Neural Networks (GraphSAGE / GAT)** with PyTorch Geometric for entity embeddings.
- Index tradecraft documents into **Weaviate** vector database with semantic/structural chunking.
- **Hybrid retrieval** combining vector similarity, graph traversal, and cross‑encoder reranking.
- **Golden dataset management** in PostgreSQL with versioning, pool separation (eval/train/holdout), and full lineage.
- **HELM‑style benchmarking** with precision/recall >70% target.

This project directly addresses all technical requirements from the **Tradecraft Evaluation Platform** job description.

---

## 🚀 Quick Start (Local Development)

### Prerequisites
- Docker & Docker Compose (v2.20+)
- Python 3.10+
- 16GB+ RAM (32GB recommended for full KG simulation)
- NVIDIA GPU (optional, for GNN training)

### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/corporate-investigation-platform.git
cd corporate-investigation-platform
cp .env.example .env
# Edit .env with your API keys (OpenAI, Weaviate, etc.)
┌─────────────────────────────────────────────────────────────┐
│                    Investigators (UI / API)                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Hybrid Graph RAG Layer                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │Vector        │  │Graph         │  │Cross‑encoder       │ │
│  │Retriever     │+ │Traversal     │→ │Reranker            │ │
│  │(Weaviate)    │  │(KG API)      │  │(MiniLM)            │ │
│  └──────────────┘  └──────────────┘  └────────────────────┘ │
└─────────────┬─────────────┬─────────────────────┬───────────┘
              │             │                     │
              ▼             ▼                     ▼
┌─────────────────┐  ┌─────────────┐  ┌─────────────────────┐
│  Weaviate       │  │ Knowledge   │  │ PostgreSQL          │
│  • Tradecraft   │  │ Graph       │  │ • Golden datasets   │
│    corpus       │  │ • 10B nodes │  │ • Versioning        │
│  • Hybrid index │  │ • Read‑only │  │ • Pool separation   │
└─────────────────┘  │   API       │  │ • Lineage tracking  │
                     └──────┬──────┘  └─────────────────────┘
                            │
                            ▼
                  ┌─────────────────────┐
                  │ GNN Layer           │
                  │ • PyTorch Geometric │
                  │ • Entity embeddings │
                  └─────────────────────┘