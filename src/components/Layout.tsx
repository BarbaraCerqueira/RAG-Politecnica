import React, { useState, ReactNode, useEffect } from 'react';
import Sidebar from './Sidebar';
import { Menu } from 'lucide-react';

interface LayoutProps {
  children: ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    // Check local storage for dark mode preference
    if (typeof window !== 'undefined') {
      const savedMode = localStorage.getItem('darkMode');
      return savedMode ? JSON.parse(savedMode) : true;
    }
    return true;
  });

  const toggleSidebar = () => setSidebarOpen(!sidebarOpen);
  
  const toggleTheme = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    localStorage.setItem('darkMode', JSON.stringify(newDarkMode));
  };

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
      {/* Mobile sidebar toggle and theme toggle */}
      <div className="fixed z-50 top-4 right-4 flex gap-2">
        <button
          className="p-2 rounded-full bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 shadow-lg transition-colors duration-200"
          onClick={toggleTheme}
        >
          {darkMode ? '☀️' : '🌙'}
        </button>
        <button
          className="md:hidden p-2 rounded-full bg-blue-600 dark:bg-blue-500 text-white shadow-lg"
          onClick={toggleSidebar}
        >
          <Menu size={24} />
        </button>
      </div>
      
      {/* Sidebar */}
      <div
        className={`fixed md:static inset-y-0 left-0 transform ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } md:translate-x-0 transition duration-200 ease-in-out z-30 w-64 bg-gray-100 dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 overflow-y-auto`}
      >
        <Sidebar onCloseMobile={() => setSidebarOpen(false)} />
      </div>
      
      {/* Backdrop */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-20 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      
      {/* Main content */}
      <div className="flex-1 overflow-y-auto">
        <main className="max-w-4xl mx-auto p-4 text-gray-900 dark:text-gray-100">{children}</main>
      </div>
    </div>
  );
};

export default Layout;