import { ChatProvider } from './contexts/ChatContext';
import Layout from './components/Layout';
import AccessGate from './components/AccessGate';
import ChatPage from './pages/ChatPage';

function App() {
  return (
    <ChatProvider>
      <Layout>
        <AccessGate>
          <ChatPage />
        </AccessGate>
      </Layout>
    </ChatProvider>
  );
}

export default App;
