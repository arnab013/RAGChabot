import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import ChatInterface from './components/ChatInterface';

function App() {
  return (
    <Router>
      <div className="h-screen bg-gradient-to-br from-gray-900 via-slate-800 to-gray-900">
        <Routes>
          <Route path="/" element={<ChatInterface />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
