import React from 'react';

const TypingIndicator = () => {
  return (
    <div className="flex items-start space-x-4 animate-slideIn">
      {/* Avatar */}
      <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 
                    flex items-center justify-center flex-shrink-0 shadow-md">
        <img 
          src={process.env.PUBLIC_URL + '/robot.png'} 
          alt="Bot" 
          className="w-6 h-6 rounded-full"
        />
      </div>

      {/* Typing Animation */}
      <div className="bg-white rounded-2xl rounded-tl-md px-5 py-4 shadow-sm border border-gray-100">
        <div className="flex space-x-1.5">
          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" 
               style={{ animationDelay: '0ms' }}></div>
          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" 
               style={{ animationDelay: '150ms' }}></div>
          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" 
               style={{ animationDelay: '300ms' }}></div>
        </div>
      </div>
    </div>
  );
};

export default TypingIndicator;
