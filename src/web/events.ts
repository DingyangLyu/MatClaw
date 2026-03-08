/**
 * Global event bus for real-time agent monitoring.
 * Container runner emits events; web server broadcasts to WebSocket clients.
 */
import { EventEmitter } from 'events';

export interface AgentEvent {
  type:
    | 'agent:start'
    | 'agent:stdout'
    | 'agent:stderr'
    | 'agent:output'
    | 'agent:end'
    | 'status';
  group: string;
  groupFolder: string;
  timestamp: string;
  data: Record<string, unknown>;
}

export const agentEvents = new EventEmitter();
