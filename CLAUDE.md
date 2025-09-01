# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a MCP-based RAG Document Management System with vLLM integration for high-performance embedding generation. The system ingests documents, generates vector embeddings, stores them in PostgreSQL with PGvector, and provides semantic search capabilities through MCP tools.

## Key Architecture Components

- **MCP Server**: Node.js/Python server handling protocol communication with LLM clients
- **Document Processor**: Handles PDF, HTML, images (OCR), emails
- **Embedding Service**: Prioritizes vLLM for GPU acceleration, falls back to Ollama/HuggingFace
- **Storage**: PostgreSQL with PGvector extension for vector storage
- **Search Engine**: Hybrid semantic and keyword search
- **File Watcher**: Monitors shared folders for automatic document ingestion

## Development Commands

### Initial Setup
```bash
# Install dependencies (when package.json exists)
npm install

# Set up PostgreSQL with PGvector
docker run -d --name rag-postgres \
  -e POSTGRES_PASSWORD=secure_password \
  -e POSTGRES_DB=rag_system \
  -p 5432:5432 \
  pgvector/pgvector:pg14

# Start vLLM server (requires GPU)
docker run -d --name vllm-server \
  --runtime nvidia --gpus all \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model BAAI/bge-large-en-v1.5

# Install and start Ollama (fallback)
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
ollama pull nomic-embed-text
```

### Development Workflow
```bash
# Build TypeScript project
npm run build

# Run migrations
npm run migrate

# Start development server
npm run dev

# Run tests
npm test

# Run linting
npm run lint

# Type checking
npm run typecheck
```

## Implementation Structure

```
src/
├── embedding/
│   ├── providers/      # vLLM, Ollama, HuggingFace, OpenAI providers
│   ├── router.ts       # Provider selection with fallback chain
│   └── cache.ts        # Embedding cache layer
├── shared-folder/
│   ├── watcher.ts      # File system monitoring
│   └── resolver.ts     # Path alias resolution
├── mcp/
│   └── server.ts       # MCP protocol implementation
├── database/
│   ├── schema.sql      # PGvector tables
│   └── client.ts       # PostgreSQL connection
└── document/
    └── processor.ts    # Document ingestion pipeline
```

## Environment Configuration

Create `.env` file with:
```
# Primary embedding provider (vllm for best performance)
EMBEDDING_PROVIDER=vllm
VLLM_HOST=http://localhost:8000
VLLM_MODEL=BAAI/bge-large-en-v1.5

# Fallback provider
EMBEDDING_FALLBACK_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434

# Database
POSTGRES_HOST=localhost
POSTGRES_DB=rag_system
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=secure_password

# Shared folders for document ingestion
SHARED_FOLDER_BASE=/var/rag-documents
AUTO_INGEST_ENABLED=true
```

## Implementation Priorities

1. **Phase 1: Core Infrastructure**
   - Set up MCP server framework
   - Implement vLLM embedding provider with Ollama fallback
   - Create PostgreSQL schema with PGvector
   - Basic document ingestion pipeline

2. **Phase 2: Document Processing**
   - PDF, HTML, and text document processors
   - Shared folder watcher with auto-ingestion
   - Path alias resolution for client/server mapping
   - Chunking strategies for long documents

3. **Phase 3: Search & Retrieval**
   - Hybrid search (semantic + keyword)
   - MCP tool implementations for search/ingest
   - Embedding cache for performance
   - Batch processing optimization

## Testing Strategy

- Unit tests for each provider
- Integration tests for vLLM/Ollama failover
- Performance benchmarks (target: 500+ embeddings/sec with vLLM)
- End-to-end MCP protocol tests

## Performance Targets

- vLLM: 500-2000 embeddings/second (GPU)
- Ollama: 50-200 embeddings/second (fallback)
- Search latency: <100ms for semantic search
- Auto-ingestion: Process files within 10 seconds of detection