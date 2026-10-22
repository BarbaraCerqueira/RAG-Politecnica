import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ChatProvider } from './contexts/ChatContext';
import Layout from './components/Layout';
import ChatPage from './pages/ChatPage';

function App() {
  return (
    <Router>
      <ChatProvider>
        <Layout>
          <Routes>
            <Route path="/" element={<ChatPage />} />
          </Routes>
        </Layout>
      </ChatProvider>
    </Router>
  );
}

export default App;