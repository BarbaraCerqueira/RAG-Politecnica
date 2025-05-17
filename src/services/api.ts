/**
 * API service for the PoliGPT application
 * Contains methods for interacting with the backend API
 */

// The API endpoint should be moved to environment variables in production
const API_URL = 'https://api-endpoint.example.com';

export interface ChatRequest {
  query: string;
  chat_history: Array<{ role: string; content: string }>;
  session_id: string;
}

export interface ChatResponse {
  response: string;
}

/**
 * Sends a message to the API and returns the response
 * @param data Chat request data containing query, history and session ID
 * @returns The API response with the assistant's message
 */
export const sendChatMessage = async (data: ChatRequest): Promise<ChatResponse> => {
  try {
    const response = await fetch(`${API_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error sending chat message:', error);
    throw error;
  }
};