import React from 'react';
import { BarChart } from 'lucide-react';

const EvaluationPage: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center py-12">
      <div className="bg-blue-100 p-4 rounded-full mb-6">
        <BarChart className="h-16 w-16 text-blue-600" />
      </div>
      <h1 className="text-3xl font-bold text-gray-800 mb-4">Em construção...</h1>
      <p className="text-gray-600 text-center max-w-md">
        A página de avaliação está sendo desenvolvida e estará disponível em breve.
      </p>
    </div>
  );
};

export default EvaluationPage;