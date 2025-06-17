import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';
import MessageBubble from './MessageBubble';
import TypingIndicator from './TypingIndicator';
import axios from 'axios';

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    {
      id: '1',
      content: 'Hello there. I am here to dig into Sustainable Development Goals for Patents with you. What do you want to start the digging with?',
      sender: 'ai',
      timestamp: new Date(Date.now() - 60000),
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const sendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage = {
      id: Date.now().toString(),
      content: inputValue,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    try {
      const response = await axios.post('/api/chat', { query: inputValue });
      
      const aiMessage = {
        id: (Date.now() + 1).toString(),
        content: response.data.response,
        sender: 'ai',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = {
        id: (Date.now() + 1).toString(),
        content: 'I apologize, but I encountered an error processing your message. Please try again.',
        sender: 'ai',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const adjustTextareaHeight = () => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = Math.min(inputRef.current.scrollHeight, 120) + 'px';
    }
  };

  useEffect(() => {
    adjustTextareaHeight();
  }, [inputValue]);

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex-shrink-0 bg-white/80 backdrop-blur-sm border-b border-gray-200/50 px-6 py-4">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">            <img 
              src={process.env.PUBLIC_URL + '/robot.png'} 
              alt="GoalDigger" 
              className="w-6 h-6 rounded-full"
            />
          </div>          <div>
            <h1 className="text-xl font-semibold text-gray-900">GoalDigger</h1>
            <p className="text-sm text-gray-500">
              {isTyping ? 'Typing...' : 'Online'}
            </p>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        
        {isTyping && <TypingIndicator />}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="flex-shrink-0 bg-white/80 backdrop-blur-sm border-t border-gray-200/50 px-6 py-4">
        <div className="flex items-end space-x-4 max-w-3xl mx-auto">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message..."
              className="w-full resize-none bg-gray-50 border border-gray-200 rounded-2xl px-4 py-3 pr-12
                       focus:ring-2 focus:ring-blue-500 focus:border-transparent focus:bg-white
                       transition-all duration-200 placeholder-gray-500 text-gray-900
                       min-h-[48px] max-h-[120px] leading-relaxed"
              rows={1}
              disabled={isTyping}
              style={{ height: '48px' }}
            />
            <button
              onClick={sendMessage}
              disabled={!inputValue.trim() || isTyping}
              className="absolute right-3 bottom-3 p-2 bg-gradient-to-r from-blue-500 to-purple-600 
                       text-white rounded-xl hover:from-blue-600 hover:to-purple-700
                       disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:from-blue-500 
                       disabled:hover:to-purple-600 transform transition-all duration-200 
                       hover:scale-105 active:scale-95 shadow-lg hover:shadow-xl"
            >
              {isTyping ? (
                <Loader2 size={18} className="animate-spin" />
              ) : (
                <Send size={18} />
              )}
            </button>
          </div>
        </div>
        
        {/* Input Helper Text */}
        <div className="text-center mt-3">
          <p className="text-xs text-gray-400">
            Press Enter to send • Shift + Enter for new line
          </p>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
