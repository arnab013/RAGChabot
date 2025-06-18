import React from 'react';

const TypingIndicator = () => {
  return (
    <div className="flex items-start space-x-4 animate-slideIn">
      {/* Avatar */}
      <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-400 to-blue-500 
                    flex items-center justify-center flex-shrink-0 shadow-lg shadow-cyan-400/30">
        <img 
          src={process.env.PUBLIC_URL + '/robot.png'} 
          alt="Bot" 
          className="w-6 h-6 rounded-full"
        />
      </div>

      {/* Typing Animation */}
      <div className="bg-gradient-to-r from-cyan-400/20 via-blue-500/20 to-cyan-400/20 rounded-2xl rounded-tl-md px-5 py-4 shadow-lg shadow-cyan-400/30 border border-cyan-400/30 backdrop-blur-md">
        <div className="flex space-x-2">
          <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce shadow-lg shadow-cyan-400/50" 
               style={{ animationDelay: '0ms' }}></div>
          <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce shadow-lg shadow-cyan-400/50" 
               style={{ animationDelay: '150ms' }}></div>
          <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce shadow-lg shadow-cyan-400/50" 
               style={{ animationDelay: '300ms' }}></div>
        </div>
      </div>
    </div>
  );
};

export default TypingIndicator;
