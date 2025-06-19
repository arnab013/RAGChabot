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
