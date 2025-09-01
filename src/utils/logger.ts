import winston from 'winston';
import { config } from '../config/index.js';

const { combine, timestamp, json, colorize, printf } = winston.format;

const prettyFormat = printf(({ level, message, timestamp, ...metadata }) => {
  let msg = `${timestamp} [${level}]: ${message}`;
  if (Object.keys(metadata).length > 0) {
    msg += ` ${JSON.stringify(metadata)}`;
  }
  return msg;
});

const formats = {
  json: combine(timestamp(), json()),
  pretty: combine(
    timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
    colorize(),
    prettyFormat
  ),
};

export const logger = winston.createLogger({
  level: config.logging.level,
  format: formats[config.logging.format],
  transports: [
    new winston.transports.Console(),
  ],
});

export function createLogger(component: string) {
  return logger.child({ component });
}