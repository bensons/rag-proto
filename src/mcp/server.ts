import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from '@modelcontextprotocol/sdk/types.js';
import { createLogger } from '../utils/logger.js';
import { db } from '../database/client.js';
import { config } from '../config/index.js';

const logger = createLogger('mcp-server');

export class MCPServer {
  private server: Server;
  private tools: Map<string, Tool>;

  constructor() {
    this.server = new Server(
      {
        name: 'rag-mcp-server',
        version: '0.1.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.tools = new Map();
    this.registerTools();
    this.setupHandlers();
  }

  private registerTools() {
    // Register ingest_document tool
    this.tools.set('ingest_document', {
      name: 'ingest_document',
      description: 'Ingest a document into the RAG system',
      inputSchema: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'Path to the document file',
          },
          metadata: {
            type: 'object',
            description: 'Optional metadata for the document',
            additionalProperties: true,
          },
          force: {
            type: 'boolean',
            description: 'Force re-ingestion even if document exists',
            default: false,
          },
        },
        required: ['path'],
      },
    });

    // Register search_documents tool
    this.tools.set('search_documents', {
      name: 'search_documents',
      description: 'Search for documents using semantic search',
      inputSchema: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'Search query',
          },
          limit: {
            type: 'number',
            description: 'Maximum number of results',
            default: 10,
          },
          mode: {
            type: 'string',
            enum: ['semantic', 'keyword', 'hybrid'],
            description: 'Search mode',
            default: 'hybrid',
          },
        },
        required: ['query'],
      },
    });

    // Register list_documents tool
    this.tools.set('list_documents', {
      name: 'list_documents',
      description: 'List all indexed documents',
      inputSchema: {
        type: 'object',
        properties: {
          limit: {
            type: 'number',
            description: 'Maximum number of documents to return',
            default: 50,
          },
          offset: {
            type: 'number',
            description: 'Offset for pagination',
            default: 0,
          },
        },
      },
    });

    // Register delete_document tool
    this.tools.set('delete_document', {
      name: 'delete_document',
      description: 'Delete a document from the system',
      inputSchema: {
        type: 'object',
        properties: {
          document_id: {
            type: 'string',
            description: 'ID of the document to delete',
          },
        },
        required: ['document_id'],
      },
    });

    // Register shared_folder_status tool
    this.tools.set('shared_folder_status', {
      name: 'shared_folder_status',
      description: 'Get status of shared folders and pending ingestions',
      inputSchema: {
        type: 'object',
        properties: {},
      },
    });

    logger.info(`Registered ${this.tools.size} MCP tools`);
  }

  private setupHandlers() {
    // Handle list tools request
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: Array.from(this.tools.values()),
      };
    });

    // Handle tool calls
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;
      const tool = this.tools.get(name);

      if (!tool) {
        throw new Error(`Unknown tool: ${name}`);
      }

      logger.info(`Executing tool: ${name}`, { args });

      try {
        switch (name) {
          case 'ingest_document':
            return await this.handleIngestDocument(args);
          
          case 'search_documents':
            return await this.handleSearchDocuments(args);
          
          case 'list_documents':
            return await this.handleListDocuments(args);
          
          case 'delete_document':
            return await this.handleDeleteDocument(args);
          
          case 'shared_folder_status':
            return await this.handleSharedFolderStatus();
          
          default:
            throw new Error(`Tool not implemented: ${name}`);
        }
      } catch (error) {
        logger.error(`Tool execution failed: ${name}`, error);
        throw error;
      }
    });
  }

  private async handleIngestDocument(args: any) {
    // This will be implemented when we have the document processor
    return {
      content: [
        {
          type: 'text',
          text: `Document ingestion initiated for: ${args.path}`,
        },
      ],
    };
  }

  private async handleSearchDocuments(args: any) {
    const { query, limit = 10, mode = 'hybrid' } = args;
    
    // For now, return a placeholder response
    // This will be implemented when we have the embedding service
    return {
      content: [
        {
          type: 'text',
          text: `Search for "${query}" with mode "${mode}" (limit: ${limit})`,
        },
      ],
    };
  }

  private async handleListDocuments(args: any) {
    const { limit = 50, offset = 0 } = args;
    
    try {
      const result = await db.query(
        `SELECT id, filename, file_type, file_size, created_at, status 
         FROM documents 
         ORDER BY created_at DESC 
         LIMIT $1 OFFSET $2`,
        [limit, offset]
      );

      const countResult = await db.query('SELECT COUNT(*) FROM documents');
      const totalCount = parseInt(countResult.rows[0].count, 10);

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              documents: result.rows,
              total: totalCount,
              limit,
              offset,
            }, null, 2),
          },
        ],
      };
    } catch (error) {
      logger.error('Failed to list documents', error);
      throw error;
    }
  }

  private async handleDeleteDocument(args: any) {
    const { document_id } = args;
    
    try {
      const result = await db.query(
        'DELETE FROM documents WHERE id = $1 RETURNING filename',
        [document_id]
      );

      if (result.rowCount === 0) {
        return {
          content: [
            {
              type: 'text',
              text: `Document not found: ${document_id}`,
            },
          ],
        };
      }

      return {
        content: [
          {
            type: 'text',
            text: `Deleted document: ${result.rows[0].filename}`,
          },
        ],
      };
    } catch (error) {
      logger.error('Failed to delete document', error);
      throw error;
    }
  }

  private async handleSharedFolderStatus() {
    try {
      const pendingResult = await db.query(
        "SELECT COUNT(*) FROM ingestion_queue WHERE status = 'pending'"
      );
      const processedResult = await db.query(
        "SELECT COUNT(*) FROM documents WHERE created_at > NOW() - INTERVAL '24 hours'"
      );
      const totalResult = await db.query('SELECT COUNT(*) FROM documents');

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              watch_paths: config.sharedFolder.watchPaths,
              auto_ingest_enabled: config.autoIngest.enabled,
              pending_files: parseInt(pendingResult.rows[0].count, 10),
              processed_today: parseInt(processedResult.rows[0].count, 10),
              total_processed: parseInt(totalResult.rows[0].count, 10),
            }, null, 2),
          },
        ],
      };
    } catch (error) {
      logger.error('Failed to get shared folder status', error);
      throw error;
    }
  }

  async start() {
    logger.info('Starting MCP server...');
    
    // Initialize database connection
    await db.initialize();
    
    // Start the MCP server
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    
    logger.info('MCP server started successfully');
  }

  async stop() {
    logger.info('Stopping MCP server...');
    await db.close();
    logger.info('MCP server stopped');
  }
}