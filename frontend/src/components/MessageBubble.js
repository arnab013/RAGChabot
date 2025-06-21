import React from 'react';
import { User } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkBreaks from 'remark-breaks';
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
            {formatTextContent(content, true)}
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
  const formatTextContent = (text, isUserMessage = false) => {
    if (!text) return <div></div>;
    
    // Pre-process text to add line breaks if missing
    let processedText = text;
    
    // If the text has very few line breaks, add them manually
    if (text.length > 500 && text.split('\n').length < 5) {
      // Add line breaks before headers
      processedText = processedText.replace(/(#{1,6}\s+)/g, '\n\n$1');
      
      // Add line breaks after headers (if not already present)
      processedText = processedText.replace(/(#{1,6}\s+[^\n]+)(?!\n)/g, '$1\n\n');
      
      // Add line breaks before bullet points
      processedText = processedText.replace(/([^-]\s*)(-\s+)/g, '$1\n$2');
      
      // Add line breaks after sentences that are likely paragraph boundaries
      processedText = processedText.replace(/(\.\s+)([A-Z][a-z])/g, '$1\n\n$2');
      
      // Clean up multiple consecutive newlines
      processedText = processedText.replace(/\n{3,}/g, '\n\n');
    }

    // Define colors based on message type
    const textColors = isUserMessage ? {
      primary: 'text-black',
      secondary: 'text-gray-800',
      accent: 'text-gray-900',
      link: 'text-blue-700',
      code: 'text-purple-700',
      codeBackground: 'bg-gray-200'
    } : {
      primary: 'text-white',
      secondary: 'text-gray-200',
      accent: 'text-gray-300',
      link: 'text-blue-400',
      code: 'text-cyan-300',
      codeBackground: 'bg-gray-700'
    };
    
    // Use ReactMarkdown for proper markdown rendering
    return (
      <ReactMarkdown
        className="prose prose-invert prose-sm max-w-none"
        remarkPlugins={[remarkBreaks]} // Enable line breaks conversion
        components={{
          // Custom components for better styling
          h1: ({children}) => <h1 className={`text-xl font-bold ${textColors.primary} mb-3 mt-4`}>{children}</h1>,
          h2: ({children}) => <h2 className={`text-lg font-semibold ${textColors.primary} mb-2 mt-3`}>{children}</h2>,
          h3: ({children}) => <h3 className={`text-md font-medium ${textColors.primary} mb-2 mt-2`}>{children}</h3>,
          h4: ({children}) => <h4 className={`text-sm font-medium ${textColors.secondary} mb-1 mt-2`}>{children}</h4>,
          ul: ({children}) => <ul className={`list-disc list-inside space-y-1 ${textColors.secondary} mb-3`}>{children}</ul>,
          ol: ({children}) => <ol className={`list-decimal list-inside space-y-1 ${textColors.secondary} mb-3`}>{children}</ol>,
          li: ({children}) => <li className={`${textColors.secondary} mb-1`}>{children}</li>,
          p: ({children}) => <p className={`${textColors.secondary} mb-3 leading-relaxed`}>{children}</p>,
          strong: ({children}) => <strong className={`font-bold ${textColors.primary}`}>{children}</strong>,
          em: ({children}) => <em className={`italic ${textColors.accent}`}>{children}</em>,
          code: ({children}) => <code className={`${textColors.codeBackground} ${textColors.code} px-1 py-0.5 rounded text-sm`}>{children}</code>,
          pre: ({children}) => <pre className={`bg-gray-800 p-3 rounded-lg overflow-x-auto text-sm ${textColors.secondary} mb-3`}>{children}</pre>,
          blockquote: ({children}) => <blockquote className={`border-l-4 border-blue-400 pl-4 italic ${textColors.accent} mb-3`}>{children}</blockquote>,
          a: ({href, children}) => <a href={href} className={`${textColors.link} hover:${textColors.link} underline`} target="_blank" rel="noopener noreferrer">{children}</a>,
          // Handle line breaks properly
          br: () => <br className="my-1" />
        }}
      >
        {processedText}
      </ReactMarkdown>
    );
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
    
    // Process chart data for rendering
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
    <div className="w-full mb-6" data-message-id={message.id}>
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
              ? 'bg-gradient-to-r from-yellow-400 via-amber-400 to-yellow-500 text-black rounded-tr-md shadow-yellow-400/30 border border-yellow-300/50 max-w-[75%] min-w-[150px] inline-block' 
              : 'bg-gradient-to-r from-cyan-400/20 via-blue-500/20 to-cyan-400/20 text-white rounded-tl-md shadow-cyan-400/30 border border-cyan-400/30 backdrop-blur-md w-full max-w-full'
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
          
          {/* Chart Container */}
          <div className="w-full max-w-full">
            <ChartFactory chartData={chartData} />
          </div>
        </div>
      )}
    </div>
  );
};

export default MessageBubble;
