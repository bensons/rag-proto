import pg from 'pg';
import pgvector from 'pgvector/pg';
import { config } from '../config/index.js';
import { createLogger } from '../utils/logger.js';

const logger = createLogger('database');
const { Pool } = pg;

export class DatabaseClient {
  private pool: pg.Pool;

  constructor() {
    this.pool = new Pool({
      host: config.database.host,
      port: config.database.port,
      database: config.database.database,
      user: config.database.user,
      password: config.database.password,
      max: config.database.maxConnections,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
    });

    this.pool.on('error', (err) => {
      logger.error('Unexpected error on idle database client', err);
    });

    this.pool.on('connect', async (client) => {
      await pgvector.registerType(client);
    });
  }

  async query<T extends pg.QueryResultRow = any>(text: string, params?: any[]): Promise<pg.QueryResult<T>> {
    const start = Date.now();
    try {
      const result = await this.pool.query<T>(text, params);
      const duration = Date.now() - start;
      logger.debug('Query executed', { text, duration, rows: result.rowCount });
      return result;
    } catch (error) {
      const duration = Date.now() - start;
      logger.error('Query failed', { text, duration, error });
      throw error;
    }
  }

  async getClient(): Promise<pg.PoolClient> {
    const client = await this.pool.connect();
    await pgvector.registerType(client);
    return client;
  }

  async transaction<T>(callback: (client: pg.PoolClient) => Promise<T>): Promise<T> {
    const client = await this.getClient();
    try {
      await client.query('BEGIN');
      const result = await callback(client);
      await client.query('COMMIT');
      return result;
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }

  async healthCheck(): Promise<boolean> {
    try {
      const result = await this.query('SELECT 1');
      return result.rowCount === 1;
    } catch (error) {
      logger.error('Health check failed', error);
      return false;
    }
  }

  async initialize(): Promise<void> {
    try {
      await this.query('SELECT 1');
      
      // Check if pgvector extension is installed
      const extensionResult = await this.query(
        "SELECT * FROM pg_extension WHERE extname = 'vector'"
      );
      
      if (extensionResult.rowCount === 0) {
        logger.warn('PGVector extension not found. Please ensure it is installed.');
      } else {
        logger.info('Database connected with PGVector extension');
      }
    } catch (error) {
      logger.error('Failed to initialize database', error);
      throw error;
    }
  }

  async close(): Promise<void> {
    await this.pool.end();
    logger.info('Database connection pool closed');
  }
}

export const db = new DatabaseClient();