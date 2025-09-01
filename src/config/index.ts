import { config as dotenvConfig } from 'dotenv';
import { z } from 'zod';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenvConfig({ path: path.resolve(__dirname, '../../.env') });

const configSchema = z.object({
  server: z.object({
    port: z.number().default(3000),
    host: z.string().default('0.0.0.0'),
  }),
  database: z.object({
    host: z.string(),
    port: z.number(),
    database: z.string(),
    user: z.string(),
    password: z.string(),
    maxConnections: z.number().default(10),
  }),
  embedding: z.object({
    provider: z.enum(['vllm', 'ollama', 'huggingface', 'openai']).default('vllm'),
    fallbackProvider: z.enum(['vllm', 'ollama', 'huggingface', 'openai', 'none']).default('ollama'),
    cache: z.object({
      enabled: z.boolean().default(true),
      size: z.number().default(1000),
    }),
  }),
  vllm: z.object({
    enabled: z.boolean().default(true),
    host: z.string().default('http://localhost:8000'),
    model: z.string().default('BAAI/bge-large-en-v1.5'),
    dimension: z.number().default(1024),
    timeout: z.number().default(60000),
    maxBatchSize: z.number().default(256),
  }),
  ollama: z.object({
    enabled: z.boolean().default(true),
    host: z.string().default('http://localhost:11434'),
    model: z.string().default('nomic-embed-text'),
    dimension: z.number().default(768),
    timeout: z.number().default(30000),
  }),
  openai: z.object({
    apiKey: z.string().optional(),
    model: z.string().default('text-embedding-3-small'),
    dimension: z.number().default(1536),
  }),
  sharedFolder: z.object({
    enabled: z.boolean().default(true),
    base: z.string().default('/var/rag-documents'),
    watchPaths: z.array(z.string()).default([]),
    temp: z.string().default('/var/rag-documents/temp'),
  }),
  autoIngest: z.object({
    enabled: z.boolean().default(true),
    pollInterval: z.number().default(10),
    filePatterns: z.array(z.string()).default(['*.pdf', '*.txt', '*.md', '*.html']),
    ignorePatterns: z.array(z.string()).default(['.DS_Store', 'Thumbs.db', '*.tmp']),
    processExisting: z.boolean().default(false),
    batchSize: z.number().default(10),
    concurrentWorkers: z.number().default(3),
  }),
  processing: z.object({
    maxChunkSize: z.number().default(1024),
    chunkOverlap: z.number().default(128),
    maxConcurrentIngestions: z.number().default(5),
  }),
  search: z.object({
    defaultMode: z.enum(['semantic', 'keyword', 'hybrid']).default('hybrid'),
    hybridAlpha: z.number().min(0).max(1).default(0.7),
  }),
  logging: z.object({
    level: z.enum(['debug', 'info', 'warn', 'error']).default('info'),
    format: z.enum(['json', 'pretty']).default('json'),
  }),
});

export type Config = z.infer<typeof configSchema>;

function loadConfig(): Config {
  const rawConfig = {
    server: {
      port: parseInt(process.env.MCP_SERVER_PORT || '3000', 10),
      host: process.env.MCP_SERVER_HOST || '0.0.0.0',
    },
    database: {
      host: process.env.POSTGRES_HOST || 'localhost',
      port: parseInt(process.env.POSTGRES_PORT || '5432', 10),
      database: process.env.POSTGRES_DB || 'rag_system',
      user: process.env.POSTGRES_USER || 'rag_user',
      password: process.env.POSTGRES_PASSWORD || 'secure_password',
      maxConnections: parseInt(process.env.DB_CONNECTION_POOL_SIZE || '10', 10),
    },
    embedding: {
      provider: process.env.EMBEDDING_PROVIDER as any || 'vllm',
      fallbackProvider: process.env.EMBEDDING_FALLBACK_PROVIDER as any || 'ollama',
      cache: {
        enabled: process.env.EMBEDDING_CACHE_ENABLED !== 'false',
        size: parseInt(process.env.EMBEDDING_CACHE_SIZE || '1000', 10),
      },
    },
    vllm: {
      enabled: process.env.VLLM_ENABLED !== 'false',
      host: process.env.VLLM_HOST || 'http://localhost:8000',
      model: process.env.VLLM_MODEL || 'BAAI/bge-large-en-v1.5',
      dimension: parseInt(process.env.VLLM_EMBEDDING_DIMENSION || '1024', 10),
      timeout: parseInt(process.env.VLLM_TIMEOUT || '60000', 10),
      maxBatchSize: parseInt(process.env.VLLM_MAX_BATCH_SIZE || '256', 10),
    },
    ollama: {
      enabled: process.env.OLLAMA_ENABLED !== 'false',
      host: process.env.OLLAMA_HOST || 'http://localhost:11434',
      model: process.env.OLLAMA_MODEL || 'nomic-embed-text',
      dimension: parseInt(process.env.OLLAMA_EMBEDDING_DIMENSION || '768', 10),
      timeout: parseInt(process.env.OLLAMA_TIMEOUT || '30000', 10),
    },
    openai: {
      apiKey: process.env.OPENAI_API_KEY,
      model: process.env.OPENAI_EMBEDDING_MODEL || 'text-embedding-3-small',
      dimension: parseInt(process.env.OPENAI_EMBEDDING_DIMENSION || '1536', 10),
    },
    sharedFolder: {
      enabled: process.env.SHARED_FOLDER_ENABLED !== 'false',
      base: process.env.SHARED_FOLDER_BASE || '/var/rag-documents',
      watchPaths: process.env.SHARED_FOLDER_WATCH_PATHS?.split(',') || [],
      temp: process.env.SHARED_FOLDER_TEMP || '/var/rag-documents/temp',
    },
    autoIngest: {
      enabled: process.env.AUTO_INGEST_ENABLED !== 'false',
      pollInterval: parseInt(process.env.AUTO_INGEST_POLL_INTERVAL || '10', 10),
      filePatterns: process.env.AUTO_INGEST_FILE_PATTERNS?.split(',') || ['*.pdf', '*.txt', '*.md', '*.html'],
      ignorePatterns: process.env.AUTO_INGEST_IGNORE_PATTERNS?.split(',') || ['.DS_Store', 'Thumbs.db', '*.tmp'],
      processExisting: process.env.AUTO_INGEST_PROCESS_EXISTING === 'true',
      batchSize: parseInt(process.env.AUTO_INGEST_BATCH_SIZE || '10', 10),
      concurrentWorkers: parseInt(process.env.AUTO_INGEST_CONCURRENT_WORKERS || '3', 10),
    },
    processing: {
      maxChunkSize: parseInt(process.env.MAX_CHUNK_SIZE || '1024', 10),
      chunkOverlap: parseInt(process.env.CHUNK_OVERLAP || '128', 10),
      maxConcurrentIngestions: parseInt(process.env.MAX_CONCURRENT_INGESTIONS || '5', 10),
    },
    search: {
      defaultMode: process.env.DEFAULT_SEARCH_MODE as any || 'hybrid',
      hybridAlpha: parseFloat(process.env.HYBRID_SEARCH_ALPHA || '0.7'),
    },
    logging: {
      level: process.env.LOG_LEVEL as any || 'info',
      format: process.env.LOG_FORMAT as any || 'json',
    },
  };

  return configSchema.parse(rawConfig);
}

export const config = loadConfig();