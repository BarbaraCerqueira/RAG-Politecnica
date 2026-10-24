/**
 * API service for the PoliGPT application
 * Contains methods for interacting with the backend API
 */

import { getFromStorage, saveToStorage, removeFromStorage } from '../utils/storage';

// Using the environment variable for the API endpoint
const API_URL = import.meta.env.VITE_API_ENDPOINT;

// The API requires an access code (a key of the Azure Function). It is never part of the build:
// the user types it, or opens a shared link with "#acesso=<code>", and it is kept in local storage.
const ACCESS_CODE_STORAGE_KEY = 'accessCode';
const ACCESS_CODE_URL_PARAM = 'acesso';

// Event dispatched when the API rejects the stored access code
export const ACCESS_DENIED_EVENT = 'poligpt:access-denied';

export class AccessDeniedError extends Error {
  constructor() {
    super('Invalid or missing access code');
  }
}

export interface ChatRequest {
  query: string;
  chat_history: Array<{ role: string; content: string }>;
  session_id: string;
}

export interface ChatResponse {
  response: string;
}

export const getAccessCode = (): string | null => getFromStorage<string | null>(ACCESS_CODE_STORAGE_KEY, null);

export const saveAccessCode = (code: string): void => saveToStorage(ACCESS_CODE_STORAGE_KEY, code);

/**
 * Saves the access code from a shared link (#acesso=<code>) and removes it from the address bar
 */
export const captureAccessCodeFromUrl = (): void => {
  const params = new URLSearchParams(window.location.hash.slice(1));
  const code = params.get(ACCESS_CODE_URL_PARAM);
  if (code) {
    saveAccessCode(code);
    window.history.replaceState(null, '', window.location.pathname + window.location.search);
  }
};

const postToApi = (body: unknown, accessCode: string | null): Promise<Response> =>
  fetch(`${API_URL}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(accessCode ? { 'x-functions-key': accessCode } : {}),
    },
    body: JSON.stringify(body),
  });

/**
 * Checks an access code without generating a response: an empty request is rejected by the
 * API with 400 when the code is valid, and with 401 when it is not
 */
export const validateAccessCode = async (code: string): Promise<boolean> => {
  const response = await postToApi({}, code);
  return response.status !== 401 && response.status !== 403;
};

/**
 * Sends a message to the API and returns the response
 * @param data Chat request data containing query, history and session ID
 * @returns The API response with the assistant's message
 */
export const sendChatMessage = async (data: ChatRequest): Promise<ChatResponse> => {
  try {
    const response = await postToApi(data, getAccessCode());

    if (response.status === 401 || response.status === 403) {
      removeFromStorage(ACCESS_CODE_STORAGE_KEY);
      window.dispatchEvent(new Event(ACCESS_DENIED_EVENT));
      throw new AccessDeniedError();
    }

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error sending chat message:', error);
    throw error;
  }
};
