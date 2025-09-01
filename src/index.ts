#!/usr/bin/env node

import { MCPServer } from './mcp/server.js';
import { createLogger } from './utils/logger.js';
import { config } from './config/index.js';

const logger = createLogger('main');

async function main() {
  logger.info('Starting RAG MCP Server', {
    version: '0.1.0',
    provider: config.embedding.provider,
    fallback: config.embedding.fallbackProvider,
  });

  const server = new MCPServer();

  // Handle graceful shutdown
  process.on('SIGINT', async () => {
    logger.info('Received SIGINT, shutting down gracefully...');
    await server.stop();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    logger.info('Received SIGTERM, shutting down gracefully...');
    await server.stop();
    process.exit(0);
  });

  process.on('uncaughtException', (error) => {
    logger.error('Uncaught exception', error);
    process.exit(1);
  });

  process.on('unhandledRejection', (reason, promise) => {
    logger.error('Unhandled rejection', { reason, promise });
    process.exit(1);
  });

  try {
    await server.start();
  } catch (error) {
    logger.error('Failed to start server', error);
    process.exit(1);
  }
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});