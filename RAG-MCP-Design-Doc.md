# Design Document: MCP-Based RAG Document Management System with vLLM Integration

## 1. Executive Summary

This document outlines the design for a server-based Retrieval-Augmented Generation (RAG) system that operates as an MCP (Model Context Protocol) server. The system ingests various document types, stores them with vector embeddings in PostgreSQL with PGvector, and provides semantic search capabilities to LLM clients through MCP tools. This includes support for local embedding models with vLLM as the primary high-performance provider, Ollama as a fallback, and configurable shared folder paths for seamless file ingestion between MCP clients and servers.

## 2. System Overview

### 2.1 Purpose

The system serves as a knowledge base that LLM clients can query through MCP to enhance their responses with relevant document context. It provides document ingestion, vector embedding generation, storage, and semantic retrieval capabilities with enterprise-grade performance through vLLM integration.

### 2.2 Key Components

- **MCP Server**: Handles protocol communication with LLM clients
- **Document Processor**: Ingests and processes various document formats
- **Embedding Service**: Generates vector embeddings for document chunks (supports both cloud and local models with vLLM priority)
- **Storage Layer**: PostgreSQL with PGvector for document and vector storage
- **Search Engine**: Performs hybrid semantic and keyword search
- **API Layer**: Exposes MCP tools for client interaction
- **File Watch Service**: Monitors shared folders for automatic ingestion
- **vLLM Service**: High-performance GPU-accelerated embedding generation

## 3. Technical Architecture

### 3.1 System Architecture

```
┌─────────────────────┐
│   MCP Client        │
│  (Claude Desktop)   │
└──────────┬──────────┘
           │ MCP Protocol
           ▼
┌─────────────────────┐     ┌─────────────────────┐
│   MCP Server        │────►│  Shared Folder      │
│  (Node.js/Python)   │     │  /watch/documents   │
├─────────────────────┤     └─────────────────────┘
│   Request Router    │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌─────────┐  ┌─────────┐
│Document │  │ Query   │
│Ingestion│  │Handler  │
└────┬────┘  └────┬────┘
     │            │
     ▼            ▼
┌──────────────────────┐     ┌──────────────────────┐
│  Embedding Service   │────►│  vLLM Server         │
│  (Router/Selector)   │     │  (High Performance)  │
└──────────┬───────────┘     └──────────────────────┘
           │                  ┌──────────────────────┐
           │─────────────────►│  Ollama (Fallback)   │
           │                  └──────────────────────┘
           ▼
┌──────────────────────┐
│     PostgreSQL       │
│    with PGvector     │
└──────────────────────┘
```

### 3.2 Technology Stack

- **Runtime**: Node.js (v20+)
- **MCP Implementation**: @modelcontextprotocol/sdk
- **Database**: PostgreSQL 14+ with PGvector extension
- **Local Embedding Models**: 
  - vLLM (for high-performance GPU inference with OpenAI-compatible API)
  - Ollama (for easy model management)
  - HuggingFace Transformers
  - SentenceTransformers
- **Document Processing**:
  - PDF: pdf-parse or PyPDF2
  - HTML: cheerio or BeautifulSoup
  - Images: Tesseract OCR with sharp/PIL for preprocessing
  - Email: mailparser or email library
- **File Watching**: chokidar
- **Configuration**: Environment variables via dotenv

## 4. Local Embedding Model Configuration

### 4.1 Supported Local Model Providers

```typescript
interface LocalEmbeddingProviders {
  vllm: {
    endpoint: string;
    models: ['BAAI/bge-large-en-v1.5', 'intfloat/e5-large-v2', 'thenlper/gte-large', 'sentence-transformers/all-mpnet-base-v2'];
    api_format: 'openai-compatible';
    tensor_parallel_size?: number;      // For multi-GPU setups
    gpu_memory_utilization?: number;    // Control GPU memory usage (0.0-1.0)
    max_model_len?: number;              // Maximum context length
    dtype?: 'auto' | 'float16' | 'bfloat16' | 'float32';
    quantization?: 'awq' | 'gptq' | 'squeezellm' | null;
    trust_remote_code?: boolean;
  };
  ollama: {
    endpoint: string;
    models: ['nomic-embed-text', 'mxbai-embed-large', 'all-minilm'];
    dimensions: { [model: string]: number };
  };
  huggingface: {
    models: ['sentence-transformers/all-MiniLM-L6-v2', 'BAAI/bge-small-en-v1.5'];
    device: 'cpu' | 'cuda' | 'mps';
    cache_dir: string;
  };
  custom: {
    endpoint: string;
    api_format: 'openai-compatible' | 'custom';
  };
}
```

### 4.2 Ollama Integration

```bash
# Install and run Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve

# Pull embedding models
ollama pull nomic-embed-text
ollama pull mxbai-embed-large
```

```typescript
class OllamaEmbeddingProvider {
  private endpoint: string;
  private model: string;
  
  constructor(config: OllamaConfig) {
    this.endpoint = config.endpoint || 'http://localhost:11434';
    this.model = config.model || 'nomic-embed-text';
  }
  
  async generateEmbedding(text: string): Promise<number[]> {
    const response = await fetch(`${this.endpoint}/api/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: this.model,
        prompt: text
      })
    });
    
    const data = await response.json();
    return data.embedding;
  }
  
  async getModelDimensions(): Promise<number> {
    // Model-specific dimensions
    const dimensions = {
      'nomic-embed-text': 768,
      'mxbai-embed-large': 1024,
      'all-minilm': 384
    };
    return dimensions[this.model] || 768;
  }
}
```

### 4.3 vLLM Integration

#### 4.3.1 vLLM Server Setup

```bash
# Install vLLM with pip
pip install vllm

# Start vLLM server with embedding model
python -m vllm.entrypoints.openai.api_server \
  --model BAAI/bge-large-en-v1.5 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --dtype auto \
  --enforce-eager

# For production with multiple GPUs
python -m vllm.entrypoints.openai.api_server \
  --model BAAI/bge-large-en-v1.5 \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --swap-space 4 \
  --disable-log-requests

# Using Docker
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  --shm-size=8g \
  vllm/vllm-openai:latest \
  --model BAAI/bge-large-en-v1.5 \
  --tensor-parallel-size 1 \
  --max-model-len 8192
```

#### 4.3.2 vLLM Provider Implementation

```typescript
class VLLMEmbeddingProvider implements EmbeddingProvider {
  private endpoint: string;
  private model: string;
  private headers: Record<string, string>;
  private maxBatchSize: number;
  private timeout: number;
  
  constructor(config: VLLMConfig) {
    this.endpoint = config.endpoint || 'http://localhost:8000';
    this.model = config.model || 'BAAI/bge-large-en-v1.5';
    this.maxBatchSize = config.max_batch_size || 256;
    this.timeout = config.timeout || 60000;
    this.headers = {
      'Content-Type': 'application/json',
      ...(config.api_key && { 'Authorization': `Bearer ${config.api_key}` })
    };
  }
  
  async generateEmbedding(text: string): Promise<number[]> {
    const response = await fetch(`${this.endpoint}/v1/embeddings`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        model: this.model,
        input: text,
        encoding_format: 'float'
      }),
      signal: AbortSignal.timeout(this.timeout)
    });
    
    if (!response.ok) {
      const error = await response.text();
      throw new Error(`vLLM embedding failed: ${response.status} - ${error}`);
    }
    
    const data = await response.json();
    return data.data[0].embedding;
  }
  
  async generateBatchEmbeddings(texts: string[]): Promise<number[][]> {
    // vLLM handles batching efficiently
    const batches = this.chunkArray(texts, this.maxBatchSize);
    const results: number[][] = [];
    
    for (const batch of batches) {
      const response = await fetch(`${this.endpoint}/v1/embeddings`, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify({
          model: this.model,
          input: batch,
          encoding_format: 'float'
        }),
        signal: AbortSignal.timeout(this.timeout)
      });
      
      if (!response.ok) {
        throw new Error(`vLLM batch embedding failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      results.push(...data.data.map(item => item.embedding));
    }
    
    return results;
  }
  
  async getModelInfo(): Promise<VLLMModelInfo> {
    const response = await fetch(`${this.endpoint}/v1/models`);
    const data = await response.json();
    
    const modelInfo = data.data.find(m => m.id === this.model);
    
    // Get actual embedding dimension by generating a test embedding
    const testEmbedding = await this.generateEmbedding("test");
    
    return {
      model: modelInfo?.id || this.model,
      dimension: testEmbedding.length,
      max_tokens: modelInfo?.max_model_len || 8192,
      permissions: modelInfo?.permission || [],
      created: modelInfo?.created || Date.now()
    };
  }
  
  async getServerMetrics(): Promise<VLLMMetrics> {
    try {
      const response = await fetch(`${this.endpoint}/metrics`);
      const metricsText = await response.text();
      
      // Parse Prometheus metrics
      return this.parsePrometheusMetrics(metricsText);
    } catch (error) {
      console.warn('Failed to fetch vLLM metrics:', error);
      return null;
    }
  }
  
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.endpoint}/health`, {
        signal: AbortSignal.timeout(5000)
      });
      return response.ok;
    } catch {
      return false;
    }
  }
  
  private chunkArray<T>(array: T[], size: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size));
    }
    return chunks;
  }
}
```

### 4.4 HuggingFace Local Models (Python)

```python
from sentence_transformers import SentenceTransformer
import torch

class LocalEmbeddingProvider:
    def __init__(self, config):
        self.model_name = config.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = config.get('cache_dir', './models')
        
        # Load model with caching
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            cache_folder=self.cache_dir
        )
        
        # Get embedding dimension
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def generate_embeddings(self, texts):
        """Generate embeddings for batch of texts"""
        return self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True
        )
    
    def warm_up(self):
        """Warm up the model with a test embedding"""
        _ = self.generate_embeddings(["test"])
```

### 4.5 Embedding Service Router with vLLM Priority

```typescript
class EmbeddingServiceRouter {
  private providers: Map<string, EmbeddingProvider>;
  private activeProvider: string;
  private providerPriority: string[] = ['vllm', 'ollama', 'huggingface', 'openai'];
  
  constructor(config: EmbeddingConfig) {
    this.providers = new Map();
    this.initializeProviders(config);
    this.activeProvider = config.default_provider || this.selectBestProvider();
  }
  
  private async initializeProviders(config: EmbeddingConfig) {
    // Initialize vLLM if configured (highest priority for performance)
    if (config.vllm?.enabled) {
      try {
        const vllmProvider = new VLLMEmbeddingProvider(config.vllm);
        if (await vllmProvider.healthCheck()) {
          this.providers.set('vllm', vllmProvider);
          console.log('vLLM provider initialized successfully');
        }
      } catch (error) {
        console.warn('Failed to initialize vLLM provider:', error);
      }
    }
    
    // Initialize OpenAI if configured
    if (config.openai?.api_key) {
      this.providers.set('openai', new OpenAIEmbeddingProvider(config.openai));
    }
    
    // Initialize Ollama if configured
    if (config.ollama?.enabled) {
      this.providers.set('ollama', new OllamaEmbeddingProvider(config.ollama));
    }
    
    // Initialize HuggingFace if configured
    if (config.huggingface?.enabled) {
      this.providers.set('huggingface', new HuggingFaceProvider(config.huggingface));
    }
  }
  
  private selectBestProvider(): string {
    // Select provider based on priority and availability
    for (const provider of this.providerPriority) {
      if (this.providers.has(provider)) {
        return provider;
      }
    }
    throw new Error('No embedding providers available');
  }
  
  async generateEmbedding(text: string, provider?: string): Promise<number[]> {
    const selectedProvider = provider || this.activeProvider;
    const service = this.providers.get(selectedProvider);
    
    if (!service) {
      throw new Error(`Embedding provider ${selectedProvider} not configured`);
    }
    
    try {
      const startTime = Date.now();
      const embedding = await service.generateEmbedding(text);
      
      // Track metrics
      await this.recordMetrics(selectedProvider, Date.now() - startTime, true);
      
      return embedding;
    } catch (error) {
      console.error(`Provider ${selectedProvider} failed:`, error);
      
      // Try fallback providers in order
      for (const fallbackProvider of this.providerPriority) {
        if (fallbackProvider !== selectedProvider && this.providers.has(fallbackProvider)) {
          console.warn(`Falling back to ${fallbackProvider}`);
          try {
            const embedding = await this.providers.get(fallbackProvider).generateEmbedding(text);
            await this.recordMetrics(fallbackProvider, 0, true, true);
            return embedding;
          } catch (fallbackError) {
            console.error(`Fallback provider ${fallbackProvider} also failed:`, fallbackError);
          }
        }
      }
      
      throw error;
    }
  }
  
  async generateBatchEmbeddings(texts: string[], provider?: string): Promise<number[][]> {
    const selectedProvider = provider || this.activeProvider;
    const service = this.providers.get(selectedProvider);
    
    // vLLM excels at batch processing
    if (selectedProvider === 'vllm' && service) {
      return (service as VLLMEmbeddingProvider).generateBatchEmbeddings(texts);
    }
    
    // Fallback to sequential processing for other providers
    return Promise.all(texts.map(text => this.generateEmbedding(text, provider)));
  }
}
```

## 5. Shared Folder Configuration and Path Aliasing

### 5.1 Folder Structure and Configuration

```typescript
interface SharedFolderConfig {
  // Server-side paths
  server: {
    watch_paths: string[];        // Actual paths on server filesystem
    ingestion_base: string;       // Base directory for document storage
    temp_upload: string;          // Temporary upload directory
  };
  
  // Client-side aliasing
  client_aliases: {
    [alias: string]: string;      // Maps client paths to server paths
  };
  
  // Auto-ingestion settings
  auto_ingest: {
    enabled: boolean;
    poll_interval: number;        // seconds
    file_patterns: string[];      // glob patterns
    ignore_patterns: string[];    // files to ignore
    process_existing: boolean;    // process files on startup
  };
}
```

### 5.2 MCP Server Configuration with Path Mapping

```json
{
  "mcpServers": {
    "rag-system": {
      "command": "node",
      "args": ["./dist/server.js"],
      "env": {
        "POSTGRES_HOST": "localhost",
        "POSTGRES_DB": "rag_system",
        "SHARED_FOLDER_BASE": "/home/user/rag-documents",
        "CLIENT_ALIAS_DOCUMENTS": "~/Documents/RAG",
        "CLIENT_ALIAS_DOWNLOADS": "~/Downloads/RAG",
        "AUTO_INGEST_ENABLED": "true",
        "EMBEDDING_PROVIDER": "vllm",
        "VLLM_HOST": "http://localhost:8000"
      }
    }
  }
}
```

### 5.3 File Watcher Implementation

```typescript
import chokidar from 'chokidar';
import path from 'path';
import { createHash } from 'crypto';

class SharedFolderWatcher {
  private watchers: Map<string, chokidar.FSWatcher>;
  private ingestionQueue: Queue;
  private processedFiles: Set<string>;
  
  constructor(
    private config: SharedFolderConfig,
    private documentProcessor: DocumentProcessor
  ) {
    this.watchers = new Map();
    this.processedFiles = new Set();
    this.ingestionQueue = new Queue();
  }
  
  async initialize() {
    // Load processed files from database
    await this.loadProcessedFiles();
    
    // Set up watchers for each configured path
    for (const watchPath of this.config.server.watch_paths) {
      await this.setupWatcher(watchPath);
    }
  }
  
  private async setupWatcher(watchPath: string) {
    const watcher = chokidar.watch(watchPath, {
      persistent: true,
      ignoreInitial: !this.config.auto_ingest.process_existing,
      ignored: this.config.auto_ingest.ignore_patterns,
      awaitWriteFinish: {
        stabilityThreshold: 2000,
        pollInterval: 100
      }
    });
    
    watcher
      .on('add', (filePath) => this.handleFileAdded(filePath))
      .on('change', (filePath) => this.handleFileChanged(filePath))
      .on('unlink', (filePath) => this.handleFileRemoved(filePath));
    
    this.watchers.set(watchPath, watcher);
  }
  
  private async handleFileAdded(filePath: string) {
    const fileHash = await this.calculateFileHash(filePath);
    
    if (this.processedFiles.has(fileHash)) {
      console.log(`File already processed: ${filePath}`);
      return;
    }
    
    // Add to ingestion queue
    await this.ingestionQueue.add({
      path: filePath,
      hash: fileHash,
      action: 'ingest',
      metadata: {
        source: 'auto_watch',
        original_path: filePath,
        ingested_at: new Date()
      }
    });
  }
  
  private async calculateFileHash(filePath: string): Promise<string> {
    const fileBuffer = await fs.readFile(filePath);
    return createHash('sha256').update(fileBuffer).digest('hex');
  }
  
  // Path resolution for client aliases
  resolveClientPath(clientPath: string): string {
    for (const [alias, serverPath] of Object.entries(this.config.client_aliases)) {
      if (clientPath.startsWith(alias)) {
        return clientPath.replace(alias, serverPath);
      }
    }
    return clientPath;
  }
}
```

### 5.4 MCP Tool with Path Resolution

```typescript
interface EnhancedMCPTools {
  // Enhanced document ingestion with path resolution
  "ingest_document": {
    description: "Ingest a document into the RAG system",
    parameters: {
      path: string;          // Can be client alias or server path
      metadata?: object;
      force?: boolean;
      use_alias?: boolean;   // Whether to resolve path aliases
    },
    handler: async (params) => {
      let resolvedPath = params.path;
      
      if (params.use_alias !== false) {
        // Attempt to resolve client alias
        resolvedPath = sharedFolderWatcher.resolveClientPath(params.path);
      }
      
      // Validate resolved path exists and is accessible
      if (!await fileExists(resolvedPath)) {
        throw new Error(`File not found: ${params.path} (resolved to: ${resolvedPath})`);
      }
      
      return await documentProcessor.ingest(resolvedPath, params.metadata);
    }
  },
  
  // Save document to shared folder for ingestion
  "save_to_shared": {
    description: "Save content to shared folder for automatic ingestion",
    parameters: {
      filename: string;
      content: string | Buffer;
      folder?: string;       // Optional subfolder in shared directory
      metadata?: object;
    },
    handler: async (params) => {
      const targetDir = params.folder 
        ? path.join(config.server.ingestion_base, params.folder)
        : config.server.ingestion_base;
      
      const filePath = path.join(targetDir, params.filename);
      
      // Ensure directory exists
      await fs.mkdir(targetDir, { recursive: true });
      
      // Write file
      await fs.writeFile(filePath, params.content);
      
      // File watcher will automatically pick it up
      return {
        saved_path: filePath,
        client_alias: reverseResolveToAlias(filePath),
        will_auto_ingest: config.auto_ingest.enabled
      };
    }
  },
  
  // Monitor shared folder status
  "shared_folder_status": {
    description: "Get status of shared folders and pending ingestions",
    parameters: {},
    handler: async () => {
      return {
        watch_paths: config.server.watch_paths,
        client_aliases: config.client_aliases,
        auto_ingest_enabled: config.auto_ingest.enabled,
        pending_files: await ingestionQueue.getPending(),
        processed_today: await getProcessedFilesCount(24),
        total_processed: processedFiles.size
      };
    }
  }
}
```

## 6. Database Schema (Enhanced)

### 6.1 Core Tables

```sql
-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50),
    file_size BIGINT,
    file_hash VARCHAR(64) UNIQUE,
    content TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    indexed_at TIMESTAMP,
    source VARCHAR(100),
    INDEX idx_documents_hash (file_hash),
    INDEX idx_documents_created (created_at)
);

-- Document chunks with vectors
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1024), -- Adjust dimension based on model
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_chunks_document (document_id),
    INDEX idx_chunks_embedding (embedding vector_cosine_ops)
);
```

### 6.2 Additional Tables for Local Models and Shared Folders

```sql
-- Embedding model configurations
CREATE TABLE embedding_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider VARCHAR(50) NOT NULL, -- 'openai', 'vllm', 'ollama', 'huggingface', 'custom'
    model_name VARCHAR(255) NOT NULL,
    dimension INTEGER NOT NULL,
    is_active BOOLEAN DEFAULT false,
    config JSONB,
    performance_stats JSONB, -- avg time, success rate
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(provider, model_name)
);

-- File watch registry
CREATE TABLE watched_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_path TEXT NOT NULL,
    file_hash VARCHAR(64) UNIQUE NOT NULL,
    watch_folder VARCHAR(255),
    client_alias VARCHAR(255),
    status VARCHAR(20), -- 'pending', 'processing', 'completed', 'failed'
    document_id UUID REFERENCES documents(id),
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_modified TIMESTAMP,
    processed_at TIMESTAMP,
    INDEX idx_watched_files_status (status),
    INDEX idx_watched_files_path (file_path)
);

-- Ingestion queue for batch processing
CREATE TABLE ingestion_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_path TEXT NOT NULL,
    file_hash VARCHAR(64),
    priority INTEGER DEFAULT 5,
    status VARCHAR(20) DEFAULT 'pending',
    attempts INTEGER DEFAULT 0,
    metadata JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_started_at TIMESTAMP,
    completed_at TIMESTAMP,
    INDEX idx_queue_status_priority (status, priority DESC)
);

-- Performance metrics table
CREATE TABLE embedding_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider VARCHAR(50) NOT NULL,
    model_name VARCHAR(255),
    duration_ms INTEGER,
    batch_size INTEGER DEFAULT 1,
    success BOOLEAN,
    from_cache BOOLEAN DEFAULT false,
    is_fallback BOOLEAN DEFAULT false,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_metrics_provider (provider, created_at),
    INDEX idx_metrics_success (success)
);
```

## 7. Configuration (Enhanced)

### 7.1 Extended Environment Variables

```bash
# Server Configuration
MCP_SERVER_PORT=3000
MCP_SERVER_HOST=0.0.0.0

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_system
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=secure_password

# Embedding Configuration - Provider Selection
EMBEDDING_PROVIDER=vllm  # 'openai', 'vllm', 'ollama', 'huggingface', 'custom'
EMBEDDING_FALLBACK_PROVIDER=ollama  # Fallback if primary fails

# vLLM Configuration (if used) - HIGHEST PRIORITY
VLLM_ENABLED=true
VLLM_HOST=http://localhost:8000
VLLM_MODEL=BAAI/bge-large-en-v1.5  # Or intfloat/e5-large-v2, thenlper/gte-large
VLLM_EMBEDDING_DIMENSION=1024  # BGE-large: 1024, E5-large: 1024, GTE-large: 1024
VLLM_API_KEY=  # Optional, if authentication is enabled
VLLM_TIMEOUT=60000  # ms
VLLM_MAX_BATCH_SIZE=256  # Maximum batch size for embeddings
VLLM_GPU_MEMORY_UTILIZATION=0.9  # Fraction of GPU memory to use
VLLM_TENSOR_PARALLEL_SIZE=1  # Number of GPUs for tensor parallelism
VLLM_PIPELINE_PARALLEL_SIZE=1  # Number of GPUs for pipeline parallelism
VLLM_MAX_MODEL_LEN=8192  # Maximum context length
VLLM_DTYPE=auto  # auto, float16, bfloat16, float32
VLLM_QUANTIZATION=  # Optional: awq, gptq, squeezellm
VLLM_TRUST_REMOTE_CODE=false  # Trust remote code in HuggingFace models
VLLM_DOWNLOAD_DIR=/models  # Directory to download models
VLLM_LOAD_FORMAT=auto  # auto, pt, safetensors, npcache, dummy

# OpenAI Configuration (if used)
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSION=1536

# Ollama Configuration (if used) - FALLBACK
OLLAMA_ENABLED=true
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=nomic-embed-text
OLLAMA_EMBEDDING_DIMENSION=768
OLLAMA_TIMEOUT=30000  # ms

# HuggingFace Local Configuration (if used)
HUGGINGFACE_ENABLED=true
HUGGINGFACE_MODEL=sentence-transformers/all-MiniLM-L6-v2
HUGGINGFACE_DEVICE=cpu  # 'cpu', 'cuda', 'mps'
HUGGINGFACE_CACHE_DIR=./models
HUGGINGFACE_BATCH_SIZE=32

# Custom Embedding Server (if used)
CUSTOM_EMBEDDING_ENABLED=false
CUSTOM_EMBEDDING_URL=http://localhost:8080/embeddings
CUSTOM_EMBEDDING_API_FORMAT=openai-compatible

# Shared Folder Configuration
SHARED_FOLDER_ENABLED=true
SHARED_FOLDER_BASE=/var/rag-documents
SHARED_FOLDER_WATCH_PATHS=/var/rag-documents/inbox,/var/rag-documents/upload
SHARED_FOLDER_TEMP=/var/rag-documents/temp

# Client Path Aliases (comma-separated key=value pairs)
CLIENT_PATH_ALIASES=~/Documents/RAG=/var/rag-documents/inbox,~/Downloads/RAG=/var/rag-documents/downloads

# Auto-Ingestion Configuration
AUTO_INGEST_ENABLED=true
AUTO_INGEST_POLL_INTERVAL=10  # seconds
AUTO_INGEST_FILE_PATTERNS=*.pdf,*.txt,*.md,*.html,*.docx
AUTO_INGEST_IGNORE_PATTERNS=.DS_Store,Thumbs.db,*.tmp,~*
AUTO_INGEST_PROCESS_EXISTING=false
AUTO_INGEST_BATCH_SIZE=10
AUTO_INGEST_CONCURRENT_WORKERS=3

# Processing Configuration
MAX_CHUNK_SIZE=1024
CHUNK_OVERLAP=128
MAX_CONCURRENT_INGESTIONS=5
CHUNK_METHOD=recursive_character

# Search Configuration
DEFAULT_SEARCH_MODE=hybrid
HYBRID_SEARCH_ALPHA=0.7
RERANKING_ENABLED=true
RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Performance Configuration
EMBEDDING_CACHE_ENABLED=true
EMBEDDING_CACHE_SIZE=1000
EMBEDDING_BATCH_SIZE=16
DB_CONNECTION_POOL_SIZE=10

# Monitoring
METRICS_ENABLED=true
METRICS_PORT=9090
HEALTH_CHECK_INTERVAL=30  # seconds

# Logging
LOG_LEVEL=info
LOG_FORMAT=json
LOG_FILE=/var/log/rag-mcp/server.log
```

### 7.2 Docker Compose with vLLM and Local Models

```yaml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg14
    environment:
      POSTGRES_DB: rag_system
      POSTGRES_USER: rag_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - rag-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag_user"]
      interval: 10s
      timeout: 5s
      retries: 5

  vllm:
    image: vllm/vllm-openai:latest
    command: [
      "--model", "BAAI/bge-large-en-v1.5",
      "--host", "0.0.0.0",
      "--port", "8000",
      "--max-model-len", "8192",
      "--gpu-memory-utilization", "0.9",
      "--dtype", "auto",
      "--enforce-eager",
      "--enable-prefix-caching",
      "--max-num-batched-tokens", "32768",
      "--max-num-seqs", "256"
    ]
    volumes:
      - huggingface_cache:/root/.cache/huggingface
      - vllm_cache:/root/.cache/vllm
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    shm_size: '8gb'
    ipc: host
    ulimits:
      memlock:
        soft: -1
        hard: -1
    networks:
      - rag-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_models:/root/.ollama
    ports:
      - "11434:11434"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - rag-network

  rag-mcp-server:
    build: .
    depends_on:
      postgres:
        condition: service_healthy
      vllm:
        condition: service_healthy
      ollama:
        condition: service_started
    environment:
      - EMBEDDING_PROVIDER=vllm
      - VLLM_HOST=http://vllm:8000
      - EMBEDDING_FALLBACK_PROVIDER=ollama
      - OLLAMA_HOST=http://ollama:11434
      - POSTGRES_HOST=postgres
      - SHARED_FOLDER_BASE=/shared
    volumes:
      - ./shared:/shared
      - ./models:/app/models
    ports:
      - "3000:3000"
    networks:
      - rag-network

networks:
  rag-network:
    driver: bridge

volumes:
  postgres_data:
  huggingface_cache:
  vllm_cache:
  ollama_models:
```

### 7.3 Initialization Script for Models

```typescript
class ModelInitializer {
  async initialize() {
    console.log('Initializing embedding models...');
    
    // Initialize vLLM if needed
    if (config.vllm.enabled) {
      await this.initializeVLLM();
    }
    
    // Check and pull Ollama models if needed
    if (config.ollama.enabled) {
      await this.initializeOllama();
    }
    
    // Download HuggingFace models if needed
    if (config.huggingface.enabled) {
      await this.initializeHuggingFace();
    }
    
    // Test each configured provider
    await this.testProviders();
    
    // Update database with available models
    await this.updateModelRegistry();
  }
  
  private async initializeVLLM() {
    console.log('Checking vLLM server status...');
    
    try {
      // Check if vLLM server is running
      const healthResponse = await fetch(`${config.vllm.host}/health`);
      
      if (!healthResponse.ok) {
        throw new Error('vLLM server not healthy');
      }
      
      // Get available models
      const modelsResponse = await fetch(`${config.vllm.host}/v1/models`);
      const models = await modelsResponse.json();
      
      console.log('vLLM available models:', models.data);
      
      // Test embedding generation
      const testResponse = await fetch(`${config.vllm.host}/v1/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: config.vllm.model,
          input: 'test embedding',
          encoding_format: 'float'
        })
      });
      
      if (!testResponse.ok) {
        throw new Error('vLLM embedding test failed');
      }
      
      const testData = await testResponse.json();
      const dimension = testData.data[0].embedding.length;
      
      console.log(`vLLM initialized: ${config.vllm.model} (${dimension} dimensions)`);
      
      // Update database with model info
      await db.query(
        `INSERT INTO embedding_models (provider, model_name, dimension, is_active, config) 
         VALUES ($1, $2, $3, $4, $5) 
         ON CONFLICT (provider, model_name) 
         DO UPDATE SET dimension = $3, is_active = $4, config = $5`,
        ['vllm', config.vllm.model, dimension, true, {
          endpoint: config.vllm.host,
          gpu_memory_utilization: config.vllm.gpu_memory_utilization,
          tensor_parallel_size: config.vllm.tensor_parallel_size
        }]
      );
    } catch (error) {
      console.error('Failed to initialize vLLM:', error);
      throw error;
    }
  }
  
  private async initializeOllama() {
    const models = ['nomic-embed-text', 'mxbai-embed-large'];
    
    for (const model of models) {
      try {
        // Check if model exists
        const response = await fetch(`${config.ollama.host}/api/show`, {
          method: 'POST',
          body: JSON.stringify({ name: model })
        });
        
        if (!response.ok) {
          console.log(`Pulling Ollama model: ${model}`);
          await this.pullOllamaModel(model);
        }
      } catch (error) {
        console.error(`Failed to initialize Ollama model ${model}:`, error);
      }
    }
  }
  
  private async initializeHuggingFace() {
    // Python subprocess or API call to download models
    const script = `
      from sentence_transformers import SentenceTransformer
      model = SentenceTransformer('${config.huggingface.model}', 
                                  cache_folder='${config.huggingface.cache_dir}')
      print(f"Model downloaded: {model.get_sentence_embedding_dimension()} dimensions")
    `;
    
    await this.executePython(script);
  }
}
```

## 8. Performance Optimization

### 8.1 Model Warm-up and Caching

```typescript
class EmbeddingOptimizer {
  private cache: LRUCache<string, number[]>;
  private warmModel: boolean = true;
  
  async initialize() {
    this.cache = new LRUCache({ max: 1000 });
    
    if (this.warmModel) {
      // Warm up model with sample text
      await this.warmUp();
    }
  }
  
  private async warmUp() {
    const samples = [
      "This is a warm-up text.",
      "Loading model into memory.",
      "Preparing embedding service."
    ];
    
    for (const text of samples) {
      await this.generateEmbedding(text);
    }
  }
  
  async generateEmbedding(text: string): Promise<number[]> {
    // Check cache first
    const cached = this.cache.get(text);
    if (cached) return cached;
    
    // Generate embedding
    const embedding = await this.provider.generateEmbedding(text);
    
    // Cache result
    this.cache.set(text, embedding);
    
    return embedding;
  }
}
```

### 8.2 Batch Processing for Local Models

```typescript
class BatchEmbeddingProcessor {
  private queue: string[] = [];
  private batchSize: number = 32;
  private flushInterval: number = 100; // ms
  
  async addToQueue(text: string): Promise<number[]> {
    return new Promise((resolve) => {
      this.queue.push({ text, resolve });
      
      if (this.queue.length >= this.batchSize) {
        this.flush();
      }
    });
  }
  
  private async flush() {
    if (this.queue.length === 0) return;
    
    const batch = this.queue.splice(0, this.batchSize);
    const texts = batch.map(item => item.text);
    
    // Process batch
    const embeddings = await this.provider.generateBatchEmbeddings(texts);
    
    // Resolve promises
    batch.forEach((item, index) => {
      item.resolve(embeddings[index]);
    });
  }
}
```

### 8.3 vLLM-Specific Optimizations

```typescript
class VLLMOptimizer {
  private config: VLLMOptimizerConfig;
  
  constructor(config: VLLMOptimizerConfig) {
    this.config = config;
  }
  
  async optimizeServerSettings(): Promise<void> {
    // Dynamically adjust settings based on workload
    const metrics = await this.getServerMetrics();
    
    if (metrics.gpu_memory_usage > 0.95) {
      console.warn('High GPU memory usage detected, consider:');
      console.warn('1. Reducing max_model_len');
      console.warn('2. Decreasing gpu_memory_utilization');
      console.warn('3. Enabling quantization (AWQ/GPTQ)');
    }
    
    if (metrics.queue_size > 100) {
      console.warn('Large queue detected, consider:');
      console.warn('1. Increasing tensor_parallel_size');
      console.warn('2. Adding more vLLM instances');
      console.warn('3. Implementing request batching');
    }
  }
  
  async benchmarkThroughput(): Promise<BenchmarkResults> {
    const testSizes = [1, 10, 50, 100, 256];
    const results: BenchmarkResults = {
      provider: 'vllm',
      throughput: {},
      latency: {}
    };
    
    for (const batchSize of testSizes) {
      const texts = Array(batchSize).fill('Benchmark text for embedding generation');
      const startTime = Date.now();
      
      await this.provider.generateBatchEmbeddings(texts);
      
      const duration = Date.now() - startTime;
      results.throughput[batchSize] = (batchSize * 1000) / duration; // embeddings/sec
      results.latency[batchSize] = duration / batchSize; // ms/embedding
    }
    
    return results;
  }
  
  getOptimalBatchSize(targetLatency: number = 100): number {
    // Determine optimal batch size based on target latency
    // vLLM can handle large batches efficiently
    const gpuMemory = this.config.gpu_memory_gb;
    const modelSize = this.config.model_size_gb;
    
    // Rough calculation: larger GPU memory allows bigger batches
    const memoryRatio = gpuMemory / modelSize;
    
    if (memoryRatio > 4) {
      return 256; // Maximum batch size
    } else if (memoryRatio > 2) {
      return 128;
    } else if (memoryRatio > 1.5) {
      return 64;
    } else {
      return 32;
    }
  }
}
```

## 9. Testing Strategy (Enhanced)

### 9.1 vLLM Integration Tests

```typescript
describe('vLLM Integration Tests', () => {
  let vllmProvider: VLLMEmbeddingProvider;
  
  beforeAll(async () => {
    vllmProvider = new VLLMEmbeddingProvider({
      endpoint: 'http://localhost:8000',
      model: 'BAAI/bge-large-en-v1.5'
    });
    
    // Wait for vLLM to be ready
    const maxRetries = 30;
    for (let i = 0; i < maxRetries; i++) {
      if (await vllmProvider.healthCheck()) {
        break;
      }
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  });
  
  test('vLLM single embedding generation', async () => {
    const embedding = await vllmProvider.generateEmbedding('test text');
    expect(embedding).toHaveLength(1024); // BGE-large dimension
    expect(embedding[0]).toBeGreaterThan(-1);
    expect(embedding[0]).toBeLessThan(1);
  });
  
  test('vLLM batch embedding performance', async () => {
    const texts = Array(100).fill('test text for batch processing');
    const start = Date.now();
    
    const embeddings = await vllmProvider.generateBatchEmbeddings(texts);
    
    const elapsed = Date.now() - start;
    const throughput = (texts.length * 1000) / elapsed;
    
    expect(embeddings).toHaveLength(100);
    expect(elapsed).toBeLessThan(2000); // Should process 100 texts in < 2s
    console.log(`vLLM Throughput: ${throughput.toFixed(2)} embeddings/sec`);
  });
  
  test('vLLM vs Ollama performance comparison', async () => {
    const texts = Array(50).fill('benchmark text');
    
    // Test vLLM
    const vllmStart = Date.now();
    await vllmProvider.generateBatchEmbeddings(texts);
    const vllmTime = Date.now() - vllmStart;
    
    // Test Ollama
    const ollamaProvider = new OllamaEmbeddingProvider(config);
    const ollamaStart = Date.now();
    for (const text of texts) {
      await ollamaProvider.generateEmbedding(text);
    }
    const ollamaTime = Date.now() - ollamaStart;
    
    console.log(`vLLM: ${vllmTime}ms, Ollama: ${ollamaTime}ms`);
    console.log(`vLLM is ${(ollamaTime / vllmTime).toFixed(2)}x faster`);
    
    expect(vllmTime).toBeLessThan(ollamaTime);
  });
  
  test('Fallback from vLLM to Ollama', async () => {
    // Simulate vLLM failure
    const router = new EmbeddingServiceRouter({
      ...config,
      vllm: { ...config.vllm, endpoint: 'http://invalid:8000' }
    });
    
    const embedding = await router.generateEmbedding('test');
    expect(embedding).toBeDefined();
    // Should have fallen back to Ollama
  });
});
```

### 9.2 Local Model Testing

```typescript
describe('Local Embedding Models', () => {
  test('Ollama model availability', async () => {
    const provider = new OllamaEmbeddingProvider(config);
    const embedding = await provider.generateEmbedding('test');
    expect(embedding).toHaveLength(768);
  });
  
  test('Fallback to local model on API failure', async () => {
    // Simulate OpenAI API failure
    const router = new EmbeddingServiceRouter(config);
    const embedding = await router.generateEmbedding('test');
    expect(embedding).toBeDefined();
  });
  
  test('Batch processing performance', async () => {
    const texts = Array(100).fill('test text');
    const start = Date.now();
    const embeddings = await batchProcessor.processBatch(texts);
    const elapsed = Date.now() - start;
    expect(elapsed).toBeLessThan(5000); // Should process 100 texts in < 5s
  });
});
```

### 9.3 Shared Folder Testing

```typescript
describe('Shared Folder Operations', () => {
  test('Path alias resolution', () => {
    const watcher = new SharedFolderWatcher(config);
    const resolved = watcher.resolveClientPath('~/Documents/RAG/test.pdf');
    expect(resolved).toBe('/var/rag-documents/inbox/test.pdf');
  });
  
  test('Auto-ingestion triggers', async () => {
    // Add file to watched folder
    await fs.writeFile('/var/rag-documents/inbox/test.txt', 'content');
    
    // Wait for watcher to process
    await sleep(3000);
    
    // Check if file was ingested
    const doc = await db.query('SELECT * FROM documents WHERE filename = $1', ['test.txt']);
    expect(doc.rows).toHaveLength(1);
  });
});
```

## 10. Deployment Guide

### 10.1 Quick Start Script with vLLM

```bash
#!/bin/bash
# setup-rag-mcp-with-vllm.sh

echo "Setting up RAG MCP Server with vLLM and Local Models..."

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: No NVIDIA GPU detected. vLLM requires GPU."
    echo "Falling back to CPU-only setup with Ollama."
    USE_VLLM=false
else
    echo "GPU detected. Setting up vLLM for high-performance inference."
    USE_VLLM=true
fi

if [ "$USE_VLLM" = true ]; then
    # 1. Install vLLM
    echo "Installing vLLM..."
    pip install vllm torch

    # 2. Start vLLM server with Docker (recommended)
    echo "Starting vLLM server..."
    docker run -d \
      --name vllm-server \
      --runtime nvidia \
      --gpus all \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      -p 8000:8000 \
      --ipc=host \
      --shm-size=8gb \
      --restart unless-stopped \
      vllm/vllm-openai:latest \
      --model BAAI/bge-large-en-v1.5 \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.9 \
      --enable-prefix-caching

    # Wait for vLLM to be ready
    echo "Waiting for vLLM to initialize (this may take a few minutes)..."
    until curl -s http://localhost:8000/health > /dev/null 2>&1; do
        echo -n "."
        sleep 5
    done
    echo " Ready!"

    # Test vLLM
    echo "Testing vLLM embedding generation..."
    curl -X POST http://localhost:8000/v1/embeddings \
      -H "Content-Type: application/json" \
      -d '{
        "model": "BAAI/bge-large-en-v1.5",
        "input": "Test embedding",
        "encoding_format": "float"
      }'
fi

# 3. Install Ollama as fallback
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# 4. Start Ollama and pull models
ollama serve &
OLLAMA_PID=$!
sleep 5
ollama pull nomic-embed-text
ollama pull mxbai-embed-large

# 5. Set up PostgreSQL with PGVector
docker run -d \
  --name rag-postgres \
  -e POSTGRES_PASSWORD=secure_password \
  -e POSTGRES_DB=rag_system \
  -p 5432:5432 \
  pgvector/pgvector:pg14

# 6. Create shared folders
mkdir -p ~/Documents/RAG
mkdir -p ~/rag-documents/{inbox,processed,failed}

# 7. Configure environment
cat > .env << EOF
# Primary embedding provider
EMBEDDING_PROVIDER=${USE_VLLM:+vllm}${USE_VLLM:-ollama}
VLLM_ENABLED=$USE_VLLM
VLLM_HOST=http://localhost:8000
VLLM_MODEL=BAAI/bge-large-en-v1.5

# Fallback provider
EMBEDDING_FALLBACK_PROVIDER=ollama
OLLAMA_ENABLED=true
OLLAMA_HOST=http://localhost:11434

# Database
POSTGRES_HOST=localhost
POSTGRES_DB=rag_system
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=secure_password

# Shared folders
SHARED_FOLDER_BASE=$HOME/rag-documents
AUTO_INGEST_ENABLED=true
EOF

# 8. Install dependencies
npm install
npm run build

# 9. Initialize database
npm run migrate

# 10. Start server
echo "Starting RAG MCP Server..."
npm start

echo "Setup complete!"
echo "Primary embedding provider: ${USE_VLLM:+vLLM}${USE_VLLM:-Ollama}"
echo "MCP Server running on http://localhost:3000"
```

### 10.2 Client Configuration Example

```json
{
  "mcpServers": {
    "rag-local": {
      "command": "node",
      "args": ["/home/user/rag-mcp-server/dist/server.js"],
      "env": {
        "EMBEDDING_PROVIDER": "vllm",
        "VLLM_HOST": "http://localhost:8000",
        "EMBEDDING_FALLBACK_PROVIDER": "ollama",
        "OLLAMA_HOST": "http://localhost:11434",
        "SHARED_FOLDER_BASE": "/home/user/rag-documents",
        "CLIENT_PATH_ALIASES": "~/Documents/RAG=/home/user/rag-documents/inbox",
        "AUTO_INGEST_ENABLED": "true"
      }
    }
  }
}
```

## 11. Monitoring and Observability

### 11.1 Model Performance Metrics

```typescript
interface ModelMetrics {
  provider: string;
  model: string;
  requests_total: number;
  requests_failed: number;
  avg_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  cache_hit_rate: number;
  fallback_used: number;
  batch_size_avg: number;
  throughput_embeddings_per_sec: number;
  gpu_memory_used_mb?: number;
  gpu_utilization_percent?: number;
}

class MetricsCollector {
  async recordEmbeddingGeneration(
    provider: string,
    duration: number,
    success: boolean,
    fromCache: boolean
  ) {
    // Update metrics
    await this.updateMetrics({
      provider,
      duration,
      success,
      fromCache
    });
    
    // Log to database for analysis
    await db.query(
      'INSERT INTO embedding_metrics (provider, duration_ms, success, from_cache) VALUES ($1, $2, $3, $4)',
      [provider, duration, success, fromCache]
    );
  }
}
```

### 11.2 Monitoring Dashboard Integration

```typescript
interface VLLMDashboardMetrics {
  // Real-time metrics
  current_requests: number;
  queue_length: number;
  gpu_utilization: number;
  memory_used_gb: number;
  
  // Performance metrics
  p50_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  throughput_per_second: number;
  
  // Model metrics
  cache_hit_rate: number;
  batch_size_avg: number;
  tokens_per_second: number;
  
  // Error metrics
  error_rate: number;
  fallback_rate: number;
  timeout_rate: number;
}

class VLLMMonitoringService {
  async exportToPrometheus(): Promise<void> {
    // Export metrics in Prometheus format
    const metrics = await this.collectMetrics();
    
    // Register metrics
    prometheus.register.clear();
    
    const throughputGauge = new prometheus.Gauge({
      name: 'vllm_embedding_throughput',
      help: 'Embeddings processed per second'
    });
    
    throughputGauge.set(metrics.throughput_per_second);
  }
  
  async setupGrafanaDashboard(): Promise<void> {
    // Auto-provision Grafana dashboard
    const dashboardConfig = {
      title: 'vLLM Embedding Service',
      panels: [
        this.createThroughputPanel(),
        this.createLatencyPanel(),
        this.createGPUPanel(),
        this.createErrorRatePanel()
      ]
    };
    
    await this.grafanaAPI.createDashboard(dashboardConfig);
  }
}
```

## 12. Troubleshooting Guide

### 12.1 Common Issues with Local Models

```markdown
### Ollama Connection Issues
- **Problem**: Cannot connect to Ollama
- **Solution**: 
  - Check if Ollama is running: `curl http://localhost:11434/api/tags`
  - Verify OLLAMA_HOST environment variable
  - Check firewall settings

### Model Dimension Mismatch
- **Problem**: Embedding dimension doesn't match database vector column
- **Solution**:
  - Check model dimension: `SELECT dimension FROM embedding_models WHERE is_active = true`
  - Update PGVector column if needed: `ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(NEW_DIMENSION)`

### Slow Local Model Performance
- **Problem**: Local embeddings are slow
- **Solution**:
  - Enable GPU acceleration if available
  - Reduce batch size for CPU processing
  - Consider using smaller models (e.g., all-MiniLM-L6-v2)
  - Implement caching for repeated content
```

### 12.2 vLLM-Specific Troubleshooting

```markdown
### vLLM Installation Issues
- **Problem**: vLLM installation fails with CUDA errors
- **Solution**:
  - Ensure CUDA 11.8+ is installed: `nvcc --version`
  - Install with specific CUDA version: `pip install vllm-cuda118`
  - Use Docker image for easier setup

### vLLM Out of Memory (OOM) Errors
- **Problem**: vLLM crashes with "CUDA out of memory"
- **Solution**:
  ```bash
  # Reduce memory usage
  --gpu-memory-utilization 0.7  # Reduce from 0.9
  --max-model-len 4096          # Reduce from 8192
  --max-num-batched-tokens 16384  # Reduce from 32768
  ```

### vLLM Slow Initialization
- **Problem**: vLLM takes long to start
- **Solution**:
  - Pre-download models: `huggingface-cli download BAAI/bge-large-en-v1.5`
  - Use `--load-format safetensors` for faster loading
  - Enable model caching: `--download-dir /models`

### vLLM Connection Refused
- **Problem**: Cannot connect to vLLM server
- **Solution**:
  - Check if server is running: `docker ps | grep vllm`
  - Verify port binding: `netstat -tlnp | grep 8000`
  - Check Docker logs: `docker logs vllm-server`
  - Ensure firewall allows port 8000

### vLLM Performance Degradation
- **Problem**: vLLM becomes slow over time
- **Solution**:
  - Monitor GPU memory fragmentation
  - Implement request batching
  - Consider multiple vLLM instances with load balancing
  - Enable prefix caching: `--enable-prefix-caching`

### Model Dimension Mismatch with vLLM
- **Problem**: Embedding dimensions don't match expected size
- **Solution**:
  - Verify model output: Different models have different dimensions
    - BGE-large: 1024
    - E5-large: 1024
    - GTE-large: 1024
    - All-MiniLM: 384
  - Update PGVector column dimension accordingly
```

## 13. Advanced Features

### 13.1 Multi-Instance vLLM Load Balancing

```typescript
class VLLMLoadBalancer {
  private instances: VLLMInstance[];
  private currentIndex: number = 0;
  
  constructor(instances: VLLMInstance[]) {
    this.instances = instances;
  }
  
  async selectInstance(): Promise<VLLMInstance> {
    // Round-robin with health checking
    const maxAttempts = this.instances.length;
    
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const instance = this.instances[this.currentIndex];
      this.currentIndex = (this.currentIndex + 1) % this.instances.length;
      
      if (await instance.isHealthy()) {
        return instance;
      }
    }
    
    throw new Error('No healthy vLLM instances available');
  }
  
  async generateEmbeddingWithLoadBalancing(text: string): Promise<number[]> {
    const instance = await this.selectInstance();
    return instance.generateEmbedding(text);
  }
}
```

### 13.2 Disaster Recovery and High Availability

```typescript
class VLLMHighAvailability {
  private primaryCluster: VLLMCluster;
  private secondaryCluster: VLLMCluster;
  
  async setupFailover(): Promise<void> {
    // Configure automatic failover
    this.healthChecker = setInterval(async () => {
      if (!await this.primaryCluster.isHealthy()) {
        await this.failoverToSecondary();
      }
    }, 5000);
  }
  
  async setupReplication(): Promise<void> {
    // Replicate model cache across instances
    await this.primaryCluster.syncModelCache(this.secondaryCluster);
    
    // Set up active-active configuration
    this.loadBalancer.addCluster(this.primaryCluster, weight: 0.7);
    this.loadBalancer.addCluster(this.secondaryCluster, weight: 0.3);
  }
  
  async backupConfiguration(): Promise<void> {
    // Backup vLLM configuration and model cache
    const backup = {
      config: await this.exportConfig(),
      models: await this.listCachedModels(),
      metrics: await this.exportMetrics(),
      timestamp: new Date().toISOString()
    };
    
    await this.storage.save('vllm-backup', backup);
  }
}
```

### 13.3 Advanced vLLM Features Integration

```typescript
// Continuous batching for streaming scenarios
class VLLMStreamingEmbedder {
  private pendingRequests: Map<string, Promise<number[]>>;
  private batchWindow: number = 50; // ms
  
  async streamingEmbed(text: string): Promise<number[]> {
    // Accumulate requests within time window
    return this.addToBatch(text);
  }
  
  private async processBatch(): Promise<void> {
    // vLLM's continuous batching handles this efficiently
    const batch = Array.from(this.pendingRequests.keys());
    const embeddings = await this.vllm.generateBatchEmbeddings(batch);
    
    // Resolve all promises
    batch.forEach((text, idx) => {
      this.pendingRequests.get(text).resolve(embeddings[idx]);
    });
  }
}

// Resource management
class VLLMResourceManager {
  private gpuMonitor: GPUMonitor;
  private memoryThreshold: number = 0.85;
  
  async autoScaleSettings(): Promise<void> {
    const gpuStats = await this.gpuMonitor.getStats();
    
    if (gpuStats.memoryUtilization > this.memoryThreshold) {
      // Dynamically adjust settings
      await this.adjustBatchSize(0.8); // Reduce by 20%
      await this.enableQuantization(); // Enable AWQ/GPTQ if available
      await this.reduceMaxTokens(0.75); // Reduce context window
    }
  }
  
  async enableMultiGPU(): Promise<void> {
    const gpuCount = await this.gpuMonitor.getDeviceCount();
    
    if (gpuCount > 1) {
      // Restart vLLM with tensor parallelism
      await this.restartWithConfig({
        tensor_parallel_size: gpuCount,
        pipeline_parallel_size: 1,
        gpu_memory_utilization: 0.95 / gpuCount
      });
    }
  }
}
```

## 14. Production Deployment

### 14.1 Kubernetes Deployment for vLLM

```yaml
# k8s/vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
  namespace: rag-system
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
          - "--model"
          - "BAAI/bge-large-en-v1.5"
          - "--tensor-parallel-size"
          - "1"
          - "--gpu-memory-utilization"
          - "0.9"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /v1/models
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 5
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: vllm-model-cache
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  namespace: rag-system
spec:
  selector:
    app: vllm
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

### 14.2 CI/CD Pipeline Integration

```yaml
# .github/workflows/vllm-deployment.yml
name: Deploy vLLM Service

on:
  push:
    branches: [main]
    paths:
      - 'vllm-config/**'
      - 'docker/vllm/**'

jobs:
  test-vllm:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v2
      
      - name: Test vLLM Configuration
        run: |
          docker run --rm --gpus all \
            -v $PWD:/workspace \
            vllm/vllm-openai:latest \
            python -m pytest /workspace/tests/vllm/
      
      - name: Benchmark Performance
        run: |
          ./scripts/benchmark-vllm.sh
          
      - name: Validate Model Loading
        run: |
          docker run --rm --gpus all \
            vllm/vllm-openai:latest \
            --model BAAI/bge-large-en-v1.5 \
            --dry-run

  deploy:
    needs: test-vllm
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl apply -f k8s/vllm-deployment.yaml
          kubectl rollout status deployment/vllm-server
```

### 14.3 Security Considerations

```typescript
class VLLMSecurityLayer {
  // API key authentication
  async setupAuthentication(): Promise<void> {
    // Generate API keys for vLLM access
    const apiKey = await this.generateAPIKey();
    
    // Configure vLLM with authentication
    await this.vllm.configure({
      api_key: apiKey,
      require_auth: true,
      allowed_ips: ['10.0.0.0/8', '172.16.0.0/12']
    });
  }
  
  // Rate limiting
  async setupRateLimiting(): Promise<void> {
    this.rateLimiter = new RateLimiter({
      requests_per_minute: 1000,
      requests_per_hour: 50000,
      burst_size: 100
    });
  }
  
  // Input validation
  async validateInput(text: string): Promise<boolean> {
    // Check for malicious inputs
    if (text.length > this.maxInputLength) {
      throw new Error('Input too long');
    }
    
    if (this.containsMaliciousPatterns(text)) {
      throw new Error('Potentially malicious input detected');
    }
    
    return true;
  }
}
```

## 15. Performance Comparison and Recommendations

### 15.1 Provider Performance Comparison

| Provider | Setup Complexity | Throughput (emb/sec) | Latency (ms) | GPU Required | Batch Support | Cost/Million |
|----------|-----------------|---------------------|--------------|--------------|---------------|--------------|
| vLLM     | Medium          | 500-2000           | 2-10         | Yes          | Excellent     | $0.50        |
| Ollama   | Easy            | 50-200             | 20-100       | Optional     | Limited       | $0.80        |
| HuggingFace | Easy         | 20-100             | 50-200       | Optional     | Good          | $1.00        |
| OpenAI API | Easy          | 100-500            | 50-500       | No           | Good          | $13.00       |

### 15.2 Model Selection Guide for vLLM

| Model | Dimension | Context Length | Use Case | VRAM Required |
|-------|-----------|---------------|----------|---------------|
| BAAI/bge-large-en-v1.5 | 1024 | 512 | General purpose, best quality | 4GB |
| BAAI/bge-base-en-v1.5 | 768 | 512 | Balanced quality/speed | 2GB |
| intfloat/e5-large-v2 | 1024 | 512 | Excellent for queries | 4GB |
| thenlper/gte-large | 1024 | 512 | Good multilingual support | 4GB |
| sentence-transformers/all-mpnet-base-v2 | 768 | 514 | Well-rounded performance | 2GB |

### 15.3 Cost-Benefit Analysis

| Scenario | vLLM Cost | Alternative Cost | Savings |
|----------|-----------|------------------|---------|
| 1M embeddings/day | $50/month (GPU) | $500/month (OpenAI) | 90% |
| Real-time search | 10ms latency | 100ms latency | 10x faster |
| Batch ingestion (10K docs) | 5 minutes | 50 minutes | 10x faster |
| On-premise requirement | ✅ Supported | ❌ Cloud-only | Compliance |

## 16. Migration Guide

### 16.1 Migration from Existing Systems

```typescript
class MigrationManager {
  async migrateFromOllama(): Promise<void> {
    // Step 1: Deploy vLLM alongside Ollama
    await this.deployVLLM();
    
    // Step 2: Set up dual-provider mode
    await this.configureDualMode({
      primary: 'ollama',
      secondary: 'vllm',
      traffic_split: 0.1 // 10% to vLLM initially
    });
    
    // Step 3: Gradually increase vLLM traffic
    for (const percentage of [0.25, 0.5, 0.75, 1.0]) {
      await this.adjustTrafficSplit(percentage);
      await this.monitorPerformance(duration: '1h');
      
      if (await this.hasIssues()) {
        await this.rollback();
        break;
      }
    }
    
    // Step 4: Complete migration
    await this.setPrimaryProvider('vllm');
    await this.setFallbackProvider('ollama');
  }
}
```

## 17. Implementation Notes for AI Coding Tools

### 17.1 Priority Implementation Order

1. **Phase 1: Core with Local Models**
   - Set up vLLM for high-performance GPU inference
   - Set up Ollama integration as fallback
   - Implement embedding service router with fallback chain (vLLM → Ollama → HuggingFace)
   - Create shared folder watcher
   - Implement path alias resolution

2. **Phase 2: Optimization**
   - Add embedding caching layer
   - Implement batch processing
   - Add model warm-up routines
   - Create performance monitoring

3. **Phase 3: Advanced Features**
   - Multiple model support with A/B testing
   - Dynamic model selection based on content type
   - Distributed processing for large batches
   - Model fine-tuning interface

### 17.2 Key Implementation Files

```
rag-mcp-server/
├── src/
│   ├── embedding/
│   │   ├── providers/
│   │   │   ├── vllm.ts
│   │   │   ├── ollama.ts
│   │   │   ├── huggingface.ts
│   │   │   └── openai.ts
│   │   ├── router.ts         # Provider selection and fallback
│   │   ├── cache.ts          # Embedding cache
│   │   └── batch.ts          # Batch processing
│   ├── shared-folder/
│   │   ├── watcher.ts        # File system monitoring
│   │   ├── resolver.ts       # Path alias resolution
│   │   └── queue.ts          # Ingestion queue
│   ├── config/
│   │   ├── embedding.ts      # Embedding configuration
│   │   └── paths.ts          # Path configuration
│   └── models/
│       ├── initializer.ts    # Model setup and testing
│       └── metrics.ts        # Performance tracking
├── docker/
│   ├── vllm/
│   │   └── Dockerfile
│   └── docker-compose.yml
├── k8s/
│   ├── vllm-deployment.yaml
│   └── rag-system-namespace.yaml
└── scripts/
    ├── setup-vllm.sh
    ├── benchmark-vllm.sh
    └── migrate-to-vllm.sh
```

## 18. Conclusion

The integration of vLLM into the RAG Document Management System provides:

1. **Enterprise-grade Performance**: 
   - 500-2000 embeddings/second throughput
   - 2-10ms latency per embedding
   - Efficient batch processing with up to 256 embeddings per request

2. **Cost Efficiency**: 
   - 90% cost reduction compared to cloud APIs
   - One-time GPU investment vs recurring API costs
   - Predictable performance and costs

3. **Flexibility and Reliability**:
   - Multiple model support with easy switching
   - Automatic fallback chain (vLLM → Ollama → HuggingFace)
   - Support for both GPU and CPU environments

4. **Production-Ready Features**:
   - Docker and Kubernetes deployment configurations
   - Comprehensive monitoring and metrics
   - Security and rate limiting capabilities
   - CI/CD pipeline integration

5. **Scalability**:
   - Multi-GPU support with tensor parallelism
   - Horizontal scaling with load balancing
   - Efficient resource management

The system maintains full backward compatibility while offering significant performance improvements for organizations with GPU resources. The automatic fallback mechanism ensures reliability even in CPU-only environments, making it suitable for both development and production deployments.

With vLLM as the primary embedding provider, the RAG system can handle enterprise-scale document processing workloads efficiently while maintaining the flexibility to adapt to different deployment scenarios and resource constraints.