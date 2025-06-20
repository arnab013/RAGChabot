import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Menu, X, Search, BarChart3, MessageSquare, FileText, Download } from 'lucide-react';
import MessageBubble from './MessageBubble';
import TypingIndicator from './TypingIndicator';
import axios from 'axios';

// Configure axios to send cookies with requests
axios.defaults.withCredentials = true;

// Welcome message constant to avoid duplication
const WELCOME_MESSAGE = 'Hello there. I am here to dig into Sustainable Development Goals for Patents with you. What do you want to start the digging with?';

// Example prompts organized by category
const examplePrompts = [
  {
    category: "Patent Search",
    icon: "Search",
    prompts: [
      "Find patents about solar energy storage",
      "Show me patents related to water purification",
      "Search for patents about sustainable agriculture",
      "Find patents about electric vehicle batteries",
      "Look for patents on renewable energy systems",
      "Show patents about carbon capture technology"
    ]
  },  {
    category: "Bar Charts & Analytics",
    icon: "BarChart3", 
    prompts: [
      "Show me patent publication trends by year",
      "Which SDG has the most patents?",
      "Show SDG distribution across patents",
      "What are the top technology fields?",
      "Show top 10 inventors by patent count",
      "Display patent counts by country",
      "Show technology analysis by IPC classification",
      "Who are the most prolific assignees?"
    ]
  },  {
    category: "Line Charts & Trends",
    icon: "BarChart3",
    prompts: [
      "Show patent publication trends in last 12 months",
      "Display publication trends for the last 6 months",
      "Show publication trends in 2023",
      "Compare patent publication trends in 2020 and 2021",
      "Show SDG patent trends over time",
      "Plot patent filing trends by technology",
      "Compare trends between 2019 and 2022",
      "Show publication trends in the last 24 months"
    ]
  },  {
    category: "Enhanced Analytics",
    icon: "TrendingUp",
    prompts: [
      "Compare patent publication trends in 2020 and 2000", 
      "Show me patent publication trends in last 12 months",
      "What are the top 5 inventors?",
      "Show geographical distribution of patents",
      "Technology analysis by IPC sections",
      "Top assignees and their patent counts",
      "Patents by applicant countries",
      "Show IPC classification distribution"
    ]
  },
  {
    category: "Pie Charts & Distribution",
    icon: "BarChart3",
    prompts: [
      "Show percentage distribution of patents by SDG",
      "What's the proportion of patents by technology type?",
      "Display patent distribution by geographic region",
      "Show breakdown of renewable vs non-renewable patents",
      "What percentage of patents are in each category?",
      "Show distribution of patents by filing organization"
    ]
  },
  {
    category: "Area Charts & Volume",
    icon: "BarChart3",
    prompts: [
      "Show cumulative patent growth over time",
      "Display overlapping SDG patent volumes",
      "Show stacked patent trends by technology",
      "Plot cumulative innovation in clean energy",
      "Display overlapping patent categories over time",
      "Show volume growth in different research areas"
    ]
  },
  {
    category: "Advanced Visualizations",
    icon: "BarChart3",
    prompts: [
      "Create a treemap of patent technology categories",
      "Show stacked bar chart of SDG patents by year",
      "Display hierarchical view of patent classifications",
      "Create a comparative analysis of patent volumes",
      "Show multi-dimensional patent data visualization",
      "Display complex patent relationship mappings"
    ]
  },  {
    category: "SDG Analysis", 
    icon: "FileText",
    prompts: [
      "Which patents contribute to SDG 7 (Clean Energy)?",
      "Show patents related to SDG 6 (Clean Water)",
      "Find SDG 13 (Climate Action) patents",
      "How do patents map to SDG 3 (Good Health)?",
      "Show SDG 9 (Industry Innovation) patents",
      "Which patents support SDG 2 (Zero Hunger)?",
      "Show SDG distribution with percentages",
      "SDG trends over the last 5 years"
    ]
  },
  {
    category: "Conversational",
    icon: "MessageSquare",
    prompts: [
      "What can you help me with?",
      "Explain how patents relate to SDGs",
      "How does this patent search system work?",
      "What types of charts can you generate?",
      "Tell me about the database coverage",
      "How are patents classified by SDG?"
    ]
  }
];

const ChatInterface = () => {  const [messages, setMessages] = useState([
    {
      id: '1',
      content: WELCOME_MESSAGE,
      sender: 'ai',
      timestamp: new Date(Date.now() - 60000),
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [sessionInfo, setSessionInfo] = useState(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  // Load session info on component mount
  useEffect(() => {    loadSessionInfo();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  
  const loadSessionInfo = async () => {
    try {
      // Our backend might not have a session-info endpoint, so we'll handle that case
      try {
        const response = await axios.get('/api/session-info');
        setSessionInfo(response.data);
      } catch (err) {
        console.log('Session info endpoint not available, using default session info');
        // Just use a default value
        setSessionInfo({
          message_count: messages.filter(m => m.sender === 'user').length
        });
      }
    } catch (error) {
      console.error('Error loading session info:', error);
    }
  };
  const clearSession = async () => {
    try {
      await axios.post('/api/reset');      setMessages([
        {
          id: '1',
          content: WELCOME_MESSAGE,
          sender: 'ai',
          timestamp: new Date(),
        }
      ]);
      await loadSessionInfo();
    } catch (error) {
      console.error('Error clearing session:', error);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  useEffect(() => {
    scrollToBottom();
  }, [messages, isThinking]);
  const sendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage = {
      id: Date.now().toString(),
      content: inputValue,
      sender: 'user',
      timestamp: new Date(),
    };    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsThinking(true);
    
    try {      // No need to send session_id anymore - Flask handles it via cookies
      const response = await axios.post('/api/search', { 
        query: inputValue 
      });
      
      console.log('API Response:', response.data);
      
      // Process chart data if available
      let chartData = null;
      if (response.data.chart) {
        console.log('Chart data found in response:', response.data.chart);
        chartData = response.data.chart;
      } else if (
        response.data.message && 
        typeof response.data.message === 'object' && 
        response.data.message.type === 'chart_text' &&
        response.data.message.content &&
        response.data.message.content.chart_data
      ) {
        console.log('Chart data found in message content:', response.data.message.content.chart_data);
        chartData = {
          type: response.data.message.content.chart_config?.type || 'line',
          title: response.data.message.content.chart_config?.title || 'Patent Statistics',
          data: response.data.message.content.chart_data
        };
      }
      
      const aiMessage = {
        id: (Date.now() + 1).toString(),
        content: response.data.message, // Keep the full structured response
        sender: 'ai',
        timestamp: new Date(),
        chartData: chartData, // Include chart data if available
        insight: response.data.insight || '', // Include insight from API response
        takeaway: response.data.takeaway || '', // Include takeaway from API response
      };

      setMessages(prev => [...prev, aiMessage]);
      
      // Update session info after successful message
      await loadSessionInfo();
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = {
        id: (Date.now() + 1).toString(),
        content: 'I apologize, but I encountered an error processing your message. Please try again.',
        sender: 'ai',
        timestamp: new Date(),
      };      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsThinking(false);
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

  const handlePromptSelect = (prompt) => {
    setInputValue(prompt);
    setIsSidebarOpen(false);
    // Focus the input after a brief delay to ensure the sidebar animation completes
    setTimeout(() => {
      if (inputRef.current) {
        inputRef.current.focus();
      }
    }, 100);
  };

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };
  const getIcon = (iconName) => {
    const iconProps = { size: 16, className: "text-gray-400" };
    switch (iconName) {
      case 'Search': return <Search {...iconProps} />;
      case 'BarChart3': return <BarChart3 {...iconProps} />;
      case 'FileText': return <FileText {...iconProps} />;
      case 'MessageSquare': return <MessageSquare {...iconProps} />;
      default: return <MessageSquare {...iconProps} />;
    }
  };
    const downloadConversation = async () => {
    try {
      // Show loading indicator
      console.log('Starting PDF generation...');
      
      // Check if messages exist
      if (messages.length <= 1) {
        alert('No conversation to download.');
        return;
      }
      
      console.log('Preparing print view...');
      
      // Create a new window for printing
      const printWindow = window.open('', '_blank');
      
      if (!printWindow) {
        alert('Please allow popups for this site to enable PDF download.');
        return;
      }
      
      // Build the HTML content for printing
      let printHTML = `
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>GoalDigger Conversation - ${new Date().toLocaleDateString()}</title>
          <style>
            @page {
              margin: 20mm;
              size: A4;
            }
            
            body {
              font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
              line-height: 1.6;
              color: #333;
              margin: 0;
              padding: 0;
              background: white;
            }
            
            .header {
              text-align: center;
              margin-bottom: 30px;
              padding-bottom: 20px;
              border-bottom: 2px solid #e5e7eb;
            }
            
            .header h1 {
              font-size: 24px;
              font-weight: bold;
              margin: 0;
              color: #0891b2;
            }
            
            .header p {
              font-size: 14px;
              color: #6b7280;
              margin: 10px 0 0 0;
            }
            
            .message {
              margin-bottom: 25px;
              page-break-inside: avoid;
            }
            
            .message-container {
              display: flex;
              align-items: flex-start;
              gap: 15px;
            }
            
            .message-container.user {
              flex-direction: row-reverse;
            }
            
            .avatar {
              width: 32px;
              height: 32px;
              border-radius: 50%;
              display: flex;
              align-items: center;
              justify-content: center;
              font-weight: bold;
              font-size: 14px;
              flex-shrink: 0;
            }
            
            .avatar.user {
              background: linear-gradient(135deg, #fbbf24, #f59e0b);
              color: #1f2937;
            }
            
            .avatar.ai {
              background: linear-gradient(135deg, #0891b2, #0284c7);
              color: white;
            }
            
            .content-wrapper {
              flex: 1;
              max-width: calc(100% - 50px);
            }
            
            .message-bubble {
              padding: 15px 20px;
              border-radius: 16px;
              word-wrap: break-word;
              line-height: 1.6;
              font-size: 14px;
              margin-bottom: 8px;
            }
            
            .message-bubble.user {
              background: #fef3c7;
              border: 1px solid #fcd34d;
              color: #1f2937;
              max-width: 70%;
              margin-left: auto;
            }
            
            .message-bubble.ai {
              background: #f0f9ff;
              border: 1px solid #0891b2;
              color: #1f2937;
            }
            
            .timestamp {
              font-size: 11px;
              color: #6b7280;
              text-align: right;
            }
            
            .timestamp.ai {
              text-align: left;
            }
            
            .chart-container {
              margin-top: 16px;
              padding: 16px;
              background: #f8fafc;
              border-radius: 12px;
              border: 1px solid #e2e8f0;
              text-align: center;
            }
            
            .chart-title {
              font-weight: bold;
              color: #0891b2;
              margin-bottom: 8px;
            }
            
            .chart-info {
              font-size: 12px;
              color: #6b7280;
            }
            
            @media print {
              body {
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
              }
              
              .message {
                page-break-inside: avoid;
              }
            }
          </style>
        </head>
        <body>
          <div class="header">
            <h1>GoalDigger Conversation</h1>
            <p>Patent & SDG Analytics - ${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString()}</p>
          </div>
      `;
      
      // Process each message
      messages.forEach((message, index) => {
        if (index === 0 && message.content.includes('Hello there')) return; // Skip welcome message
        
        const isUser = message.sender === 'user';
        
        // Process message content
        let content = '';
        if (typeof message.content === 'string') {
          content = message.content;
        } else if (message.content && typeof message.content === 'object') {
          if (message.content.text) {
            content = message.content.text;
          } else if (message.content.content && message.content.content.text) {
            content = message.content.content.text;
          } else {
            content = JSON.stringify(message.content, null, 2);
          }
        }
        
        // Format content with line breaks and escape HTML
        content = content.replace(/&/g, '&amp;')
                        .replace(/</g, '&lt;')
                        .replace(/>/g, '&gt;')
                        .replace(/\n/g, '<br>');
        
        printHTML += `
          <div class="message">
            <div class="message-container ${isUser ? 'user' : ''}">
              <div class="avatar ${isUser ? 'user' : 'ai'}">
                ${isUser ? 'U' : 'AI'}
              </div>
              <div class="content-wrapper">
                <div class="message-bubble ${isUser ? 'user' : 'ai'}">
                  ${content}
                </div>
                <div class="timestamp ${isUser ? 'user' : 'ai'}">
                  ${message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            </div>
        `;
        
        // Add chart if present
        if (!isUser && message.chartData) {
          printHTML += `
            <div class="chart-container">
              <div class="chart-title">📊 ${message.chartData.title || 'Chart'}</div>
              <div class="chart-info">Chart data visualization (${message.chartData.type || 'chart'})</div>
            </div>
          `;
        }
        
        printHTML += '</div>';
      });
      
      printHTML += `
        </body>
        </html>
      `;
      
      // Write content to the new window
      printWindow.document.write(printHTML);
      printWindow.document.close();
      
      // Wait for content to load, then trigger print
      printWindow.onload = () => {
        setTimeout(() => {
          printWindow.print();
          
          // Close the window after printing (user can cancel this)
          printWindow.onafterprint = () => {
            printWindow.close();
          };
        }, 500);
      };
      
      console.log('Print dialog opened');
      
    } catch (error) {
      console.error('Error generating PDF:', error);
      console.error('Error details:', error.message);
      console.error('Error stack:', error.stack);
      
      const useTextFallback = window.confirm(
        `PDF generation failed: ${error.message}\n\nWould you like to download the conversation as a text file instead?`
      );
      
      if (useTextFallback) {
        downloadAsText();
      }
    }
  };

  // Simple text export as fallback
  const downloadAsText = () => {
        try {
          let textContent = `GoalDigger Conversation Export\n`;
          textContent += `Date: ${new Date().toLocaleDateString()}\n`;
          textContent += `${'-'.repeat(50)}\n\n`;
          
          messages.forEach((message, index) => {
            if (index === 0 && message.content.includes('Hello there')) return; // Skip welcome message
            
            const sender = message.sender === 'user' ? 'USER' : 'AI';
            const time = message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            
            textContent += `[${time}] ${sender}:\n`;
            
            let content = '';
            if (typeof message.content === 'string') {
              content = message.content;
            } else if (message.content && typeof message.content === 'object') {
              if (message.content.text) {
                content = message.content.text;
              } else if (message.content.content && message.content.content.text) {
                content = message.content.content.text;
              } else {
                content = JSON.stringify(message.content, null, 2);
              }
            }
            
            textContent += `${content}\n`;
            
            if (message.chartData) {
              textContent += `📊 Chart: ${message.chartData.title || 'Data Visualization'}\n`;
            }
            
            textContent += `\n${'-'.repeat(30)}\n\n`;
          });
          
          const blob = new Blob([textContent], { type: 'text/plain' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `GoalDigger_Conversation_${new Date().toISOString().split('T')[0]}.txt`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
          
          console.log('Text export completed');
        } catch (error) {
          console.error('Text export failed:', error);
          alert('Export failed. Please try again.');        }
      };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 flex items-center justify-center p-2">
      {/* Main Chat Container - Centered with proper sizing */}
      <div className="flex h-[calc(100vh-1rem)] w-full max-w-5xl bg-gray-800/30 backdrop-blur-lg border border-gray-700/50 rounded-xl shadow-2xl overflow-hidden relative">
        
        {/* Main Chat Area - Full Width */}
        <div className="flex flex-col flex-1 w-full">
          {/* Header */}
          <div className="flex-shrink-0 bg-gray-800/90 backdrop-blur-sm border-b border-gray-700/50 px-9 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-cyan-400 to-blue-500 rounded-full flex items-center justify-center shadow-lg">
                <img 
                  src={process.env.PUBLIC_URL + '/robot.png'} 
                  alt="GoalDigger" 
                  className="w-6 h-6 rounded-full"
                />
              </div>              <div>
                <h1 className="text-xl font-semibold text-white">GoalDigger</h1>
                <p className="text-sm text-gray-300">
                  {isThinking ? 'Thinking...' : 'Online'}
                </p>
              </div>
            </div>            {/* Session controls */}          <div className="flex items-center space-x-2">
            <button
              onClick={toggleSidebar}
              className="p-2 text-gray-300 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
              title="Example prompts"
            >
              <Menu size={20} />
            </button>
            {messages.length > 1 && (
              <button
                onClick={downloadConversation}
                className="p-2 text-gray-300 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
                title="Download conversation as PDF"
              >
                <Download size={20} />
              </button>
            )}
            {messages.length > 1 && (
              <button
                onClick={clearSession}
                className="p-2 text-gray-300 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
                title="Clear conversation"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </button>
            )}
          </div></div>
        </div>

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto px-9 py-6">
            <div className="max-w-4xl mx-auto space-y-6">
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
              
              {isThinking && <TypingIndicator />}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Input Area */}
          <div className="flex-shrink-0 bg-gray-800/90 backdrop-blur-sm border-t border-gray-700/50 px-9 py-4">
            <div className="flex items-end space-x-4 max-w-4xl mx-auto">
              <div className="flex-1 relative chat-input-container">
                <textarea
                  ref={inputRef}
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type your message..."
                  className="w-full resize-none bg-gray-700 border border-gray-600 rounded-2xl px-4 py-3 pr-12
                           focus:ring-2 focus:ring-cyan-400 focus:border-transparent focus:bg-gray-600
                           transition-all duration-200 placeholder-gray-400 text-white
                           min-h-[49px] max-h-[120px] leading-relaxed overflow-hidden"
                  rows={1}
                  disabled={isThinking}
                  style={{ height: '49px', resize: 'none' }}
                />
                <button
                  onClick={sendMessage}
                  disabled={!inputValue.trim() || isThinking}
                  className="absolute right-3 bottom-3 p-2 bg-gradient-to-r from-cyan-400 to-blue-500 
                           text-white rounded-xl hover:from-cyan-500 hover:to-blue-600
                           disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:from-cyan-400 
                           disabled:hover:to-blue-500 transform transition-all duration-200 
                           hover:scale-105 active:scale-95 shadow-lg hover:shadow-xl z-10"
                >
                  {isThinking ? (
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
        
        {/* Backdrop Overlay */}
        {isSidebarOpen && (
          <div 
            className="absolute inset-0 bg-black/50 backdrop-blur-sm z-40 transition-opacity duration-300"
            onClick={toggleSidebar}
          />
        )}
        
        {/* Sidebar Overlay with Example Prompts */}
        <div className={`absolute top-0 left-0 w-80 h-full bg-gray-800/95 backdrop-blur-sm border-r border-gray-700/50 transform transition-transform duration-300 ease-in-out z-50 ${
          isSidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}>
          <div className="flex items-center justify-between p-4 border-b border-gray-700">
            <h2 className="text-lg font-semibold text-white">Example Prompts</h2>
            <button
              onClick={toggleSidebar}
              className="p-2 text-gray-300 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
            >
              <X size={20} />
            </button>
          </div>
          
          <div className="p-4 h-full overflow-y-auto pb-20">
            {examplePrompts.map((category, categoryIndex) => (
              <div key={categoryIndex} className="mb-6">
                <div className="flex items-center space-x-2 mb-3">
                  {getIcon(category.icon)}
                  <h3 className="text-sm font-medium text-gray-300">{category.category}</h3>
                </div>
                <div className="space-y-2">
                  {category.prompts.map((prompt, promptIndex) => (
                    <button
                      key={promptIndex}
                      onClick={() => handlePromptSelect(prompt)}
                      className="w-full text-left p-3 text-sm text-gray-300 hover:bg-gray-700 hover:text-white rounded-lg transition-colors border border-gray-600 hover:border-gray-500"
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
        
      {/* End of Main Container */}
      </div>
    </div>
  );
};

export default ChatInterface;
