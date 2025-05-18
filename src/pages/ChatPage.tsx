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
    <div className="h-[calc(100%-144px)] relative overflow-y-auto scrollbar-thin scrollbar-thumb-gray-300 dark:scrollbar-thumb-gray-700 
      scrollbar-track-transparent hover:scrollbar-thumb-gray-400 
      dark:hover:scrollbar-thumb-gray-600
      scrollbar-thumb-rounded-full">
      <div className="flex flex-col min-h-full max-w-5xl mx-auto px-4">
        {messages.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center">
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

            <div className="w-full">
              <SuggestionButtons suggestions={suggestionButtons} />
            </div>
          </div>
        ) : (
          <div className="flex-1">
            <ChatMessages />
          </div>
        )}
      </div>
      
      <div className="fixed bottom-0 left-0 md:left-64 right-0 bg-white dark:bg-gray-900">
        <div className="max-w-5xl mx-auto px-4">
          <ChatInput />
        </div>
      </div>
    </div>
  );
};

export default ChatPage;