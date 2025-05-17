import React from 'react';
import ChatInput from '../components/ChatInput';
import ChatMessages from '../components/ChatMessages';
import SuggestionButtons from '../components/SuggestionButtons';
import poliLogo from '../assets/poligpt_logo.png';

const ChatPage: React.FC = () => {
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
    <div className="flex flex-col h-[calc(100vh-2rem)] relative">
      <div className="flex-1 overflow-hidden flex flex-col">
        {/* Empty chat state */}
        <div className="flex-1 flex flex-col items-center justify-center px-6 pb-24">
          {/* Logo and Title */}
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

          {/* Suggestion buttons in a grid */}
          <div className="w-full max-w-4xl">
            <SuggestionButtons suggestions={suggestionButtons} />
          </div>
        </div>

        {/* Messages container */}
        <ChatMessages />
      </div>

      {/* Input area - fixed at bottom */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-white dark:from-gray-900 pt-6">
        <ChatInput />
      </div>
    </div>
  );
};

export default ChatPage;