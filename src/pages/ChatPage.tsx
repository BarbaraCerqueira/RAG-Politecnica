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
    <div className="grid grid-rows-[1fr_auto] h-full max-h-screen overflow-hidden bg-gray-200/70 dark:bg-gray-900">
      
      {/* Área de mensagens com a scrollbar propria */}
      <div className="overflow-y-auto scrollbar scrollbar-thumb-gray-300 dark:scrollbar-thumb-gray-700 
        scrollbar-track-transparent hover:scrollbar-thumb-gray-400 
        dark:hover:scrollbar-thumb-gray-600 scrollbar-thumb-rounded-full">
        <div className="flex flex-col min-h-full max-w-5xl mx-auto px-4">
          {messages.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center py-10">
              <img src={poliLogo} alt="PoliGPT" className="w-48 mb-8" />
              <h1 className="text-4xl font-bold text-gray-700 dark:text-gray-200/70 mb-3">
                Como posso ajudar você hoje?
              </h1>
              <p className="text-gray-600 dark:text-gray-400 max-w-2xl">
                Sou o assistente virtual da Escola Politécnica da UFRJ, 
                pronto para ajudar com questões acadêmicas e administrativas.
              </p>
              <div className="mt-8 w-full">
                <SuggestionButtons suggestions={suggestionButtons} />
              </div>
            </div>
          ) : (
            <div className="flex flex-col space-y-4 pt-16 pb-12">
              <ChatMessages />
            </div>
          )}
        </div>
      </div>

      {/* Chat Input fixo na base da pagina*/}
      <div>
        <div className="max-w-5xl mx-auto mb-8 px-8">
          <ChatInput />
        </div>
      </div>
    </div>
  );
};

export default ChatPage;