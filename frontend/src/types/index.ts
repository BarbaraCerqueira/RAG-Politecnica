/**
 * Type definitions for the PoliGPT application
 */

export interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatHistory {
  messages: Message[];
  sessionId: string;
}

export interface Suggestion {
  text: string;
  query: string;
}