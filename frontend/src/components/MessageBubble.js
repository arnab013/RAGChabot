import React from 'react';
import { User } from 'lucide-react';
import ChartFactory from './charts/ChartFactory';

const MessageBubble = ({ message }) => {
  const isUser = message.sender === 'user';
  
  const formatTime = (date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  // Function to extract chart data from content
  const extractChartFromContent = (content) => {
    try {
      if (typeof content === 'object' && content !== null) {
        // Check for structured chart content
        if (content.type === 'chart_text' && content.content) {
          const { chart_data, chart_config, title } = content.content;
          if (chart_data) {
            return {
              type: chart_config?.type || 'bar',
              title: chart_config?.title || title || 'Patent Statistics',
              data: chart_data
            };
          }
        }
        // Check for direct chart data
        if (content.chart && typeof content.chart === 'object') {
          return content.chart;
        }
      }
    } catch (error) {
      console.error('Error extracting chart from content:', error);
    }
    return null;  };  // Get chart data from content or message.chartData
  const chartData = !isUser ? (extractChartFromContent(message.content) || message.chartData) : null;

  const formatContent = (content) => {
    try {
      console.log('Formatting content:', content);
      
      // Handle user messages - always simple format
      if (isUser) {
        return (
          <div className="text-[15px] leading-relaxed">
            {formatTextContent(content)}
          </div>
        );
      }
        // Check if this is a simple bot message (string content)
      if (typeof content === 'string') {
        // Check if this message has chart data and insight/takeaway fields
        if (chartData && (message.insight || message.takeaway)) {
          // Create a structured response with insight and takeaway sections
          return (
            <div className="space-y-4">
              {/* Main message content */}
              <div className="text-[15px] leading-relaxed">
                {formatTextContent(content)}
              </div>
              
              {/* Key Insights Section */}
              {message.insight && (
                <div className="bg-gradient-to-r from-blue-500/10 to-cyan-500/10 border border-blue-400/20 rounded-lg p-4 mb-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                    <h4 className="font-semibold text-blue-300 text-sm">Key Insights</h4>
                  </div>
                  <div className="text-gray-200 text-sm">
                    {formatTextContent(message.insight)}
                  </div>
                </div>
              )}
              
              {/* Takeaway Section */}
              {message.takeaway && (
                <div className="bg-gradient-to-r from-purple-900/20 to-indigo-900/20 border border-purple-500/20 rounded-lg p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    <h4 className="font-semibold text-purple-300 text-sm">Key Takeaway</h4>
                  </div>
                  <div className="text-gray-200 text-sm italic">
                    {formatTextContent(message.takeaway)}
                  </div>
                </div>
              )}
            </div>
          );
        }
        
        // All string content should be rendered directly without extra containers
        return (
          <div className="text-[15px] leading-relaxed">
            {formatTextContent(content)}
          </div>
        );
      }
      
      // Handle structured content from the new API format
      if (typeof content === 'object' && content !== null) {
        // New structured format
        if (content.type && content.content) {
          const { type, content: contentData } = content;
          console.log(`Processing ${type} content:`, contentData);
          
          switch (type) {            case 'text':
              const textData = contentData.body || contentData.title || 'No content';
              return (
                <div className="text-[15px] leading-relaxed">
                  {formatTextContent(textData)}
                </div>
              );
            
            case 'semantic_text':
              return formatSemanticTextContent(contentData);
            
            case 'chart_text':
              return formatChartTextContent(contentData);
            
            default:
              console.warn('Unknown content type:', type);
              return formatTextContent(contentData.body || contentData.title || JSON.stringify(content));
          }
        }        // Check if this is chart data directly - extract for separate display
        if (content.chart && typeof content.chart === 'object') {
          console.log('Direct chart data found:', content.chart);
          // Create enhanced content data with insight and takeaway for chart display
          const chartContentData = {
            title: content.chart.title || "Analysis Result",
            description: content.message,
            insight: content.insight || "",
            takeaway: content.takeaway || "",
            chart_data: content.chart.data
          };
          return formatChartTextContent(chartContentData);
        }// Handle legacy format or malformed objects
        if (content.body || content.title) {
          const textContent = content.body || content.title;
          return (
            <div className="text-[15px] leading-relaxed">
              {formatTextContent(textContent)}
            </div>
          );
        }
      }      
      // Fallback for any other format
      const fallbackContent = String(content || '');
      return (
        <div className="text-[15px] leading-relaxed">
          {formatTextContent(fallbackContent)}
        </div>
      );
    } catch (error) {
      console.error('Error formatting content:', error);
      return <div className="text-red-500 text-sm">Error displaying message content</div>;
    }
  };
  const formatTextContent = (text) => {
    if (!text) return <div></div>;
    
    return text.split('\n').map((line, i) => {
      // Handle different formatting styles
      if (line.trim() === '') {
        return <br key={i} />;
      }
      
      let formattedLine = line;
      
      // Handle headers
      if (line.startsWith('# ')) {
        return (
          <h1 key={i} className="text-xl font-bold text-cyan-300 mt-4 mb-3 border-b border-cyan-400/30 pb-2">
            {line.substring(2)}
          </h1>
        );
      } else if (line.startsWith('## ')) {
        return (
          <h2 key={i} className="text-lg font-semibold text-blue-300 mt-4 mb-2">
            {line.substring(3)}
          </h2>
        );
      } else if (line.startsWith('### ')) {
        return (
          <h3 key={i} className="text-base font-medium text-gray-300 mt-3 mb-2">
            {line.substring(4)}
          </h3>
        );
      }
      
      // Handle bullet points
      if (line.trim().startsWith('• ')) {
        return (
          <div key={i} className="flex items-start space-x-2 ml-4 mt-1">
            <span className="text-cyan-400 mt-1 text-sm">•</span>
            <span className="text-gray-200">{formatInlineText(line.trim().substring(2))}</span>
          </div>
        );
      }
      
      // Handle numbered lists
      const numberedMatch = line.match(/^(\d+)\.\s(.+)$/);
      if (numberedMatch) {
        return (
          <div key={i} className="flex items-start space-x-2 ml-4 mt-1">
            <span className="text-cyan-400 font-medium">{numberedMatch[1]}.</span>
            <span className="text-gray-200">{formatInlineText(numberedMatch[2])}</span>
          </div>
        );
      }
      
      // Handle horizontal rules
      if (line.trim() === '---') {
        return <hr key={i} className="border-gray-600 my-4" />;
      }
      
      // Handle regular text with inline formatting
      formattedLine = formatInlineText(line);
      
      return (
        <div key={i} className={`${i > 0 ? 'mt-2' : ''} text-gray-200 leading-relaxed`}>
          {formattedLine}
        </div>
      );
    });
  };

  const formatInlineText = (text) => {
    if (!text) return text;
    
    // Handle both bold text and patent numbers
    const parts = [];
    let currentIndex = 0;
    
    // Combined regex to find both bold patterns and patent numbers
    const combinedRegex = /(\*\*(.*?)\*\*)|([A-Z]{2}[0-9]{7}[A-Z]?\d?)/g;
    let match;
    
    while ((match = combinedRegex.exec(text)) !== null) {
      // Add text before the match
      if (match.index > currentIndex) {
        parts.push(text.substring(currentIndex, match.index));
      }
      
      if (match[1]) {
        // This is a bold match
        parts.push(
          <strong key={`bold-${match.index}`} className="font-semibold text-white">
            {match[2]}
          </strong>
        );
      } else if (match[3]) {
        // This is a patent number match
        parts.push(
          <span 
            key={`patent-${match.index}`} 
            className="font-medium text-cyan-400 bg-cyan-900/30 px-1 py-0.5 rounded text-sm"
          >
            {match[3]}
          </span>
        );
      }
      
      currentIndex = match.index + match[0].length;
    }
    
    // Add remaining text
    if (currentIndex < text.length) {
      parts.push(text.substring(currentIndex));
    }
    
    // If no matches were found, return original text
    return parts.length === 0 ? text : (parts.length === 1 ? parts[0] : parts);
  };  const formatSemanticTextContent = (contentData) => {
    const { title, body } = contentData;
    
    return (
      <div className="space-y-4">
        {title && (
          <div className="border-l-4 border-green-400 pl-4 mb-4">
            <h3 className="text-lg font-bold text-white mb-1">{title}</h3>
            <p className="text-xs text-gray-400 uppercase tracking-wide">Search Result</p>
          </div>
        )}
        
        {body && (
          <div className="bg-gradient-to-r from-gray-800/30 to-gray-700/30 border border-gray-600/30 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-3">
              <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <h4 className="font-semibold text-gray-300 text-sm">Information</h4>
            </div>
            <div className="text-gray-200 text-sm leading-relaxed">
              {formatTextContent(body)}
            </div>
          </div>
        )}
      </div>
    );
  };  const formatChartTextContent = (contentData) => {
    // Use insight and takeaway from message properties if available, otherwise from contentData
    const insight = message.insight || contentData.insight || '';
    const takeaway = message.takeaway || contentData.takeaway || '';
    const { title, description, chart_data } = contentData;
    
    // Log chart data to help with debugging
    console.log('Chart text content data:', contentData);
    console.log('Message insight:', message.insight);
    console.log('Message takeaway:', message.takeaway);
    
    return (
      <div className="space-y-4">
        {/* Response Header */}
        {title && (
          <div className="border-l-4 border-cyan-400 pl-4 mb-4">
            <h3 className="text-lg font-bold text-white mb-1">{title}</h3>
            <p className="text-xs text-gray-400 uppercase tracking-wide">Analysis Result</p>
          </div>
        )}
        
        {/* Key Insights Section */}
        {insight && (
          <div className="bg-gradient-to-r from-blue-500/10 to-cyan-500/10 border border-blue-400/20 rounded-lg p-4 mb-4">
            <div className="flex items-center space-x-2 mb-2">
              <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
              <h4 className="font-semibold text-blue-300 text-sm">Key Insights</h4>
            </div>
            <div className="text-gray-200 text-sm">
              {formatTextContent(insight)}
            </div>
          </div>
        )}
        
        {/* Detailed Response */}
        {description && (
          <div className="bg-gradient-to-r from-gray-800/30 to-gray-700/30 border border-gray-600/30 rounded-lg p-4 mb-4">
            <div className="flex items-center space-x-2 mb-3">
              <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <h4 className="font-semibold text-gray-300 text-sm">Detailed Response</h4>
            </div>
            <div className="text-gray-200 text-sm leading-relaxed">
              {formatTextContent(description)}
            </div>
          </div>
        )}
        
        {/* Data Summary */}
        {chart_data && (
          <div className="bg-gradient-to-r from-green-900/20 to-emerald-900/20 border border-green-500/20 rounded-lg p-4 mb-4">
            <div className="flex items-center space-x-2 mb-3">
              <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
              </svg>
              <h4 className="font-semibold text-green-300 text-sm">Data Summary</h4>
            </div>
            <div className="text-gray-200 text-xs">
              <p className="mb-2">Found <span className="font-bold text-green-400">{chart_data.length}</span> data points for visualization</p>
              {chart_data.length > 0 && (
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-gray-400">Categories:</span>
                    <span className="ml-1 text-green-300 font-medium">
                      {chart_data.length > 5 ? `${chart_data.slice(0, 3).map(item => Object.keys(item)[0]).join(', ')}...` : chart_data.map(item => Object.keys(item)[0]).join(', ')}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Data Range:</span>
                    <span className="ml-1 text-green-300 font-medium">
                      {Math.min(...chart_data.map(item => Object.values(item)[0] || 0))} - {Math.max(...chart_data.map(item => Object.values(item)[0] || 0))}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* Takeaway */}
        {takeaway && (
          <div className="bg-gradient-to-r from-purple-900/20 to-indigo-900/20 border border-purple-500/20 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              <h4 className="font-semibold text-purple-300 text-sm">Key Takeaway</h4>
            </div>
            <div className="text-gray-200 text-sm italic">
              {formatTextContent(takeaway)}
            </div>
          </div>
        )}
      </div>
    );
  };
  return (
    <div className="w-full mb-6">
      {/* Message Row */}
      <div className={`flex items-start space-x-4 animate-slideIn ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`}>
        {/* Avatar */}
        <div className={`
          w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg
          ${isUser 
            ? 'bg-gradient-to-br from-yellow-400 to-amber-500 shadow-yellow-400/30' 
            : 'bg-gradient-to-br from-cyan-400 to-blue-500 shadow-cyan-400/30'
          }
        `}>
          {isUser ? (
            <User size={18} className="text-gray-900" />
          ) : (
            <img 
              src={process.env.PUBLIC_URL + '/robot.png'} 
              alt="Bot" 
              className="w-6 h-6 rounded-full"
            />
          )}
        </div>        {/* Message Content */}
        <div className={`
          group
          ${isUser ? 'flex flex-col items-end w-full' : 'flex flex-col items-start w-full'}
        `}>
          <div className={`
            px-6 py-5 rounded-2xl shadow-lg relative backdrop-blur-sm
            ${isUser 
              ? 'bg-gradient-to-r from-yellow-400 via-amber-400 to-yellow-500 text-gray-900 rounded-tr-md shadow-yellow-400/30 border border-yellow-300/50 max-w-[85%] min-w-[150px] inline-block' 
              : 'bg-gradient-to-r from-cyan-400/20 via-blue-500/20 to-cyan-400/20 text-white rounded-tl-md shadow-cyan-400/30 border border-cyan-400/30 backdrop-blur-md w-full max-w-5xl'
            }
            transform transition-all duration-300 hover:shadow-xl hover:-translate-y-1 hover:scale-[1.01]
            ${isUser ? 'hover:shadow-yellow-400/40' : 'hover:shadow-cyan-400/40'}
            min-h-[80px]
          `}>
            <div className="text-[15px] leading-relaxed">
              {formatContent(message.content)}
            </div>
          </div>
          
          {/* Timestamp */}
          <div className={`
            text-xs text-gray-400 mt-2 px-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200
            ${isUser ? 'text-right self-end' : 'text-left self-start'}
          `}>
            {formatTime(message.timestamp)}
          </div>
        </div>
      </div>      {/* Dedicated Chart Container - Separate from message bubble */}
      {!isUser && chartData && (
        <div className="flex items-start space-x-4 mt-8 animate-fadeIn">
          {/* Spacer for avatar alignment */}
          <div className="w-10 h-10 flex-shrink-0"></div>
          
          {/* Chart Container - matching message width */}
          <div className="w-full max-w-5xl">
            <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-sm rounded-2xl shadow-2xl border border-cyan-400/10 p-1 hover:border-cyan-400/20 transition-all duration-300 w-full">
              <div className="bg-gradient-to-r from-gray-800/80 to-gray-900/80 rounded-xl p-6 border border-gray-700/50">
                {/* Chart Header */}
                <div className="flex items-center justify-between mb-6 pb-4 border-b border-gray-600/30">
                  <div className="flex items-center space-x-4">
                    <div className="w-10 h-10 bg-gradient-to-br from-cyan-400 to-blue-500 rounded-xl flex items-center justify-center shadow-lg shadow-cyan-400/25">
                      <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                    </div>
                    <div>
                      <h4 className="text-white font-semibold text-base">Data Visualization</h4>
                      <p className="text-gray-400 text-sm">Interactive Chart Analysis</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <span className="px-4 py-1.5 bg-gradient-to-r from-cyan-400/20 to-blue-500/20 text-cyan-300 text-sm rounded-full border border-cyan-400/30 font-medium">
                      {chartData.type?.toUpperCase() || 'CHART'}
                    </span>
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse shadow-lg shadow-green-400/50"></div>
                  </div>
                </div>
                
                {/* Chart Content */}
                <div className="relative bg-gradient-to-br from-gray-900/20 to-gray-800/20 rounded-xl p-2 border border-gray-700/30">
                  <ChartFactory chartData={chartData} />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MessageBubble;
