/**
 * API service for the PoliGPT application
 * Contains methods for interacting with the backend API
 */

// Using the environment variable for the API endpoint
const API_URL = import.meta.env.VITE_API_ENDPOINT;

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
    const response = await fetch(`${API_URL}`, {
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