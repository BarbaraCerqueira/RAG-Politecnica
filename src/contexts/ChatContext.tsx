import React, { createContext, useContext, useState, ReactNode } from 'react';
import { v4 as uuidv4 } from 'uuid';

// Define types for our context
export interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface Chat {
  id: string;
  title: string;
  messages: Message[];
}

interface ChatContextType {
  chats: Chat[];
  currentChatId: string | null;
  messages: Message[]; // Current chat messages for backward compatibility
  isLoading: boolean;
  switchChat: (id: string) => void;
  newChat: () => void;
  sendMessage: (content: string) => Promise<void>;
  clearMessages: () => void;
}

// Create the context with a default value
const ChatContext = createContext<ChatContextType | undefined>(undefined);

// Hook to use the chat context
export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};

interface ChatProviderProps {
  children: ReactNode;
}

// API endpoint - should be moved to env variable in production
const API_ENDPOINT = 'https://api-endpoint.example.com/chat';

export const ChatProvider: React.FC<ChatProviderProps> = ({ children }) => {
  const [chats, setChats] = useState<Chat[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Initialize with a new chat
  React.useEffect(() => {
    if (chats.length === 0) {
      const newChatId = uuidv4();
      setChats([{
        id: newChatId,
        title: 'Chat 1',
        messages: []
      }]);
      setCurrentChatId(newChatId);
    }
  }, []);

  // Get current chat messages
  const messages = currentChatId 
    ? chats.find(chat => chat.id === currentChatId)?.messages || []
    : [];

  const getCurrentChatIndex = () => {
    return chats.findIndex(chat => chat.id === currentChatId);
  };

  const switchChat = (id: string) => {
    setCurrentChatId(id);
  };

  const newChat = () => {
    const newChatId = uuidv4();
    setChats(prevChats => [
      ...prevChats,
      {
        id: newChatId,
        title: `Chat ${prevChats.length + 1}`,
        messages: []
      }
    ]);
    setCurrentChatId(newChatId);
  };

  const appendMessage = (message: Message) => {
    const chatIndex = getCurrentChatIndex();
    if (chatIndex === -1) return;

    setChats(prevChats => {
      const newChats = [...prevChats];
      newChats[chatIndex] = {
        ...newChats[chatIndex],
        messages: [...newChats[chatIndex].messages, message]
      };
      return newChats;
    });
  };

  const clearMessages = () => {
    const chatIndex = getCurrentChatIndex();
    if (chatIndex === -1) return;

    setChats(prevChats => {
      const newChats = [...prevChats];
      newChats[chatIndex] = {
        ...newChats[chatIndex],
        messages: []
      };
      return newChats;
    });
  };

  const sendMessage = async (content: string) => {
    if (!currentChatId || isLoading) return;

    setIsLoading(true);
    
    // Add user message
    const userMessage: Message = { role: 'user', content };
    appendMessage(userMessage);

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Simulate assistant response
      const assistantMessage: Message = {
        role: 'assistant',
        content: 'Desculpe, ocorreu um erro ao processar sua mensagem. Por favor, tente novamente.'
      };
      appendMessage(assistantMessage);
    } catch (error) {
      console.error('Error sending message:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const value = {
    chats,
    currentChatId,
    messages,
    isLoading,
    switchChat,
    newChat,
    sendMessage,
    clearMessages
  };

  return (
    <ChatContext.Provider value={value}>
      {children}
    </ChatContext.Provider>
  );
};