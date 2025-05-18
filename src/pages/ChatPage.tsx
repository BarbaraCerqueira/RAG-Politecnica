import React from 'react';
import ChatInput from '../components/ChatInput';
import ChatMessages from '../components/ChatMessages';
import SuggestionButtons from '../components/SuggestionButtons';
import poliLogo from '../assets/poligpt_logo.png';
import { useChat } from '../contexts/ChatContext';

const ChatPage: React.FC = () => {
  const { messages } = useChat();
  const suggestionButtons = [
    {
      text: "Como posso me inscrever em novas matérias?",
      query: "Como posso me inscrever em matérias?"
    },
    {
      text: "Como faço para trancar uma disciplina?",
      query: "Como faço para trancar uma disciplina?"
    },
    {
      text: "Qual é o procedimento para registrar estágio?",
      query: "Qual é o procedimento para registrar um estágio?"
    }
  ];

  return (
    <div className="flex flex-col h-[calc(100vh-2rem)]">
      <div className="flex-1 flex flex-col min-h-0">
        {messages.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center px-6">
            <div className="flex flex-col items-center mb-8 text-center">
              <img src={poliLogo} alt="PoliGPT" className="w-48 mb-8" />
              <h1 className="text-4xl font-bold text-gray-800 dark:text-gray-100 mb-3">
                Como posso ajudar você hoje?
              </h1>
              <p className="text-gray-600 dark:text-gray-400 max-w-2xl text-center">
                Sou o assistente virtual da Escola Politécnica da UFRJ, 
                pronto para ajudar com questões acadêmicas e administrativas.
              </p>
            </div>

            <div className="w-full max-w-4xl mb-8">
              <SuggestionButtons suggestions={suggestionButtons} />
            </div>
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto pt-6 scrollbar-thin scrollbar-thumb-gray-300 dark:scrollbar-thumb-gray-700 
            scrollbar-track-transparent hover:scrollbar-thumb-gray-400 
            dark:hover:scrollbar-thumb-gray-600
            scrollbar-thumb-rounded-full">
            <ChatMessages />
          </div>
        )}
        
        <div className="flex-shrink-0 bg-white dark:bg-gray-900">
          <ChatInput />
        </div>
      </div>
    </div>
  );
};

export default ChatPage;