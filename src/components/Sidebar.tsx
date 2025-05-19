import React from 'react';
import { MessageSquare, Info, Link2, Plus } from 'lucide-react';
import poliLogo from '../assets/poligpt_logo.png';
import { useChat } from '../contexts/ChatContext';

interface SidebarProps {
  onCloseMobile: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ onCloseMobile }) => {
  const { chats, currentChatId, newChat, switchChat } = useChat();

  const handleChatSwitch = (chatId: string) => {
    switchChat(chatId);
    onCloseMobile();
  };

  return (
    <div className="flex flex-col h-full">
      {/* Logo */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <img src={poliLogo} alt="PoliGPT" className="mx-auto w-40" />
      </div>
      
      {/* New Chat Button */}
      <div className="px-2 py-4">
        <button
          onClick={() => {
            newChat();
            onCloseMobile();
          }}
          className="w-full flex items-center px-4 py-3 rounded-lg transition-colors bg-blue-600 hover:bg-blue-700 text-white"
        >
          <Plus className="mr-3 h-5 w-5" />
          <span className="font-medium">Novo Chat</span>
        </button>
      </div>

      {/* Chat List */}
      <nav className="flex-1 px-2 overflow-y-auto">
        {chats.map(chat => (
          <button
            key={chat.id}
            onClick={() => handleChatSwitch(chat.id)}
            className={`w-full flex items-center px-4 py-3 rounded-lg transition-colors mb-1 ${
              chat.id === currentChatId
                ? 'bg-blue-600/15 dark:bg-blue-900/50 text-blue-800 dark:text-blue-300'
                : 'hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            <MessageSquare className="mr-3 h-5 w-5" />
            <span className="font-medium">{chat.title}</span>
          </button>
        ))}
      </nav>
      
      {/* Information */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-700">
        <div className="flex items-start space-x-2 text-sm">
          <Info className="h-5 w-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="font-semibold mb-2 text-gray-900 dark:text-gray-100">Sobre o PoliGPT:</h4>
            <p className="text-gray-600 dark:text-gray-400 mb-3">
              O PoliGPT é um assistente virtual criado para auxiliar alunos da Escola Politécnica da Universidade 
              Federal do Rio de Janeiro (UFRJ) em suas dúvidas. Este chatbot foi alimentado com dados obtidos a 
              partir do website da Escola Politécnica.
            </p>
            <a 
              href="https://www.poli.ufrj.br/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center py-2 px-3 text-sm bg-gray-200 dark:bg-gray-700 rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors text-blue-700 dark:text-blue-300"
            >
              <Link2 className="mr-2 h-4 w-4" />
              Escola Politécnica
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;