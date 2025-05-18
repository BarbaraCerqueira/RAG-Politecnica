import React, { useState, FormEvent, useRef, useEffect } from 'react';
import { Send, Trash2 } from 'lucide-react';
import { useChat } from '../contexts/ChatContext';

const ChatInput: React.FC = () => {
  const [message, setMessage] = useState('');
  const { sendMessage, isLoading, clearMessages, messages } = useChat();
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    // Focus the textarea when the component mounts
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  }, []);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    }
  }, [message]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isLoading) {
      await sendMessage(message);
      setMessage('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="flex items-start gap-2 mb-8 px-4">
      {messages.length > 0 && (
        <button
          onClick={clearMessages}
          className="p-3 text-gray-600 dark:text-gray-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-full transition-colors self-end"
          title="Limpar conversa"
        >
          <Trash2 size={20} />
        </button>
      )}
      <form 
        onSubmit={handleSubmit} 
        className="relative flex items-end flex-1 bg-white dark:bg-gray-800 rounded-2xl shadow-lg"
      >
        <textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Escreva uma mensagem..."
          disabled={isLoading}
          rows={1}
          className="flex-1 py-4 px-4 pr-12 bg-transparent border border-gray-300 dark:border-gray-600 rounded-2xl focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:border-transparent disabled:opacity-50 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 resize-none min-h-[56px] max-h-[150px] [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]"
        />
        <button
          type="submit"
          disabled={!message.trim() || isLoading}
          className="absolute right-2 bottom-2 p-2 rounded-xl text-blue-600 dark:text-blue-400 hover:bg-blue-100 dark:hover:bg-blue-900/50 disabled:opacity-50 disabled:hover:bg-transparent transition-colors"
        >
          <Send size={24} />
        </button>
      </form>
    </div>
  );
};

export default ChatInput;