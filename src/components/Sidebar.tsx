import React from 'react';
import { MessageSquare, Info, Link2, PlusSquare, Trash2 } from 'lucide-react';
import poliLogo from '../assets/poligpt_logo.png';
import { useChat } from '../hooks/useChat';

interface SidebarProps {
  onCloseMobile: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ onCloseMobile }) => {
  const { chats, currentChatId, newChat, switchChat, deleteChat } = useChat();
  
  return (
    <div className="flex flex-col h-full">
      {/* Logo */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <img src={poliLogo} alt="PoliGPT" className="mx-auto w-40" />
      </div>
      
      {/* New Chat Button */}
      <div className="px-4 py-4">
        <button
          onClick={() => { newChat(); onCloseMobile(); }}
          className="flex items-center w-full px-4 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
        >
          <PlusSquare className="mr-3 h-5 w-5" />
          <span className="font-medium">Nova Conversa</span>
        </button>
      </div>
      
      {/* Chat List */}
      <div className="flex-1 flex flex-col min-h-0">

        <div className="flex-1 overflow-y-auto space-y-1 scrollbar scrollbar-thumb-gray-300 dark:scrollbar-thumb-gray-700 
          scrollbar-track-transparent hover:scrollbar-thumb-gray-400 
          dark:hover:scrollbar-thumb-gray-600 scrollbar-thumb-rounded-full px-2 pb-2 pt-2">
          {chats.map(chat => (
            <div 
              key={chat.id} 
              className="flex items-center relative group"
            >
              <button
                onClick={() => { switchChat(chat.id); onCloseMobile(); }}
                className={`flex items-center w-full px-4 py-2 rounded-lg transition-colors ${
                  chat.id === currentChatId 
                    ? 'bg-blue-600/15 dark:bg-blue-900/50 text-blue-800 dark:text-blue-300' 
                    : 'hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
              >
                <MessageSquare className="mr-3 h-4 w-4 flex-shrink-0" />
                <span className={`font-medium truncate ${chat.id === currentChatId ? 'mr-7' : 'group-hover:mr-7'}`}>
                  {chat.title}
                </span>
              </button>
              <button
                onClick={(e) => { 
                  e.stopPropagation();
                  deleteChat(chat.id); 
                }}
                className={`absolute right-2 p-1.5 text-gray-500 hover:text-red-500 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-full transition-all ${
                  chat.id === currentChatId 
                    ? 'opacity-100' 
                    : 'opacity-0 group-hover:opacity-100'
                }`}
                title="Excluir conversa"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </div>
          ))}
        </div>
      </div>
      
      {/* Information */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-700 mt-auto">
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