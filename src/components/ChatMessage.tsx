import React from 'react';
import { Message } from '../contexts/ChatContext';
import userAvatar from '../assets/user_avatar.png';
import assistantAvatar from '../assets/assistant_avatar.png';

interface ChatMessageProps {
  message: Message;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const avatar = message.role === 'user' ? userAvatar : assistantAvatar;
  const name = message.role === 'user' ? 'Você' : 'PoliGPT';
  
  return (
    <div 
      className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} animate-fadeIn mb-6`}
    >
      <div 
        className={`flex max-w-full ${message.role === 'user' ? 'flex-row-reverse' : 'flex-row'} items-end`}
      >
        <div className="flex flex-col items-center mx-3">
          <div className="h-10 w-10 rounded-lg overflow-hidden bg-white/80 dark:bg-gray-200 p-0.5 shadow-sm">
            <img src={avatar} alt={message.role} className="h-full w-full object-cover" />
          </div>
          <span className="text-xs text-gray-500 dark:text-gray-400 mt-1">{name}</span>
        </div>
        <div 
          className={`rounded-2xl px-5 py-3 ${
            message.role === 'user' 
              ? 'bg-blue-600 dark:bg-blue-700 text-white rounded-tr-none shadow-md' 
              : 'bg-white/80 dark:bg-gray-700 text-gray-700 dark:text-gray-100 rounded-tl-none shadow-md'
          }`}
        >
          <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;