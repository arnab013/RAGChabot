import React from 'react';
import { User } from 'lucide-react';

const MessageBubble = ({ message }) => {
  const isUser = message.sender === 'user';
  
  const formatTime = (date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const formatContent = (content) => {
    return content.split('\n').map((line, i) => (
      <div key={i} className={i > 0 ? 'mt-2' : ''}>
        {line || <br />}
      </div>
    ));
  };

  return (
    <div className={`flex items-start space-x-4 animate-slideIn ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`}>
      {/* Avatar */}
      <div className={`
        w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 shadow-md
        ${isUser 
          ? 'bg-gradient-to-br from-blue-500 to-blue-600' 
          : 'bg-gradient-to-br from-purple-500 to-pink-500'
        }
      `}>
        {isUser ? (
          <User size={18} className="text-white" />
        ) : (
          <img 
            src={process.env.PUBLIC_URL + '/robot.png'} 
            alt="Bot" 
            className="w-6 h-6 rounded-full"
          />
        )}
      </div>

      {/* Message Content */}
      <div className={`
        max-w-2xl group
        ${isUser ? 'items-end' : 'items-start'}
      `}>
        <div className={`
          px-5 py-3 rounded-2xl shadow-sm relative
          ${isUser 
            ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-tr-md' 
            : 'bg-white text-gray-800 rounded-tl-md border border-gray-100'
          }
          transform transition-all duration-300 hover:shadow-md hover:-translate-y-0.5
        `}>
          <div className="text-[15px] leading-relaxed">
            {formatContent(message.content)}
          </div>
        </div>
        
        {/* Timestamp */}
        <div className={`
          text-xs text-gray-400 mt-2 px-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200
          ${isUser ? 'text-right' : 'text-left'}
        `}>
          {formatTime(message.timestamp)}
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;
