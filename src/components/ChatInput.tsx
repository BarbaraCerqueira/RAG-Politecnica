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
    <form onSubmit={handleSubmit} className="relative">
      <div className="flex items-end space-x-2">
        <div className="flex-1 min-h-[60px]">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Digite sua mensagem..."
            className="w-full p-4 pr-12 rounded-lg bg-gray-100 dark:bg-gray-800 
              border border-gray-300 dark:border-gray-700 resize-none
              focus:outline-none focus:border-blue-500 dark:focus:border-blue-400
              text-gray-800 dark:text-gray-200"
            style={{ minHeight: '60px', maxHeight: '150px' }}
          />
        </div>
        
        <div className="flex space-x-2 pb-3">
          {messages.length > 0 && (
            <button
              type="button"
              onClick={clearMessages}
              className="p-2 text-gray-500 hover:text-red-600 dark:text-gray-400 
                dark:hover:text-red-400 transition-colors"
              title="Limpar conversa"
            >
              <Trash2 className="h-5 w-5" />
            </button>
          )}
          
          <button
            type="submit"
            disabled={!message.trim() || isLoading}
            className={`p-2 rounded-lg transition-colors ${
              message.trim() && !isLoading
                ? 'text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300'
                : 'text-gray-400 dark:text-gray-600 cursor-not-allowed'
            }`}
          >
            <Send className="h-5 w-5" />
          </button>
        </div>
      </div>
    </form>
  );
};

export default ChatInput;