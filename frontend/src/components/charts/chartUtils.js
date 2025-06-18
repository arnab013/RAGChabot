/**
 * Shared utilities for chart components
 */

/**
 * Modern dark theme color palette for charts
 */
export const CHART_COLORS = [
  '#06b6d4', // cyan-500
  '#3b82f6', // blue-500  
  '#8b5cf6', // violet-500
  '#10b981', // emerald-500
  '#f59e0b', // amber-500
  '#ef4444', // red-500
  '#ec4899', // pink-500
  '#84cc16', // lime-500
  '#6366f1', // indigo-500
  '#14b8a6', // teal-500
  '#f97316', // orange-500
  '#a855f7', // purple-500
  '#22d3ee', // cyan-400
  '#60a5fa', // blue-400
  '#a78bfa'  // violet-400
];

/**
 * Get colors for chart data
 */
export const getChartColors = (count = 8) => {
  return Array.from({ length: count }, (_, i) => CHART_COLORS[i % CHART_COLORS.length]);
};

/**
 * Convert string to title case
 */
export const toTitleCase = (str) => {
  if (!str) return '';
  return str
    .split(/[_\s-]/)
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ');
};

/**
 * Common chart styling for dark theme
 */
export const CHART_THEME = {
  grid: {
    stroke: '#374151',
    strokeDasharray: '3 3'
  },
  axis: {
    tick: { fontSize: 12, fill: '#d1d5db' },
    stroke: '#6b7280'
  },
  label: {
    style: { fill: '#d1d5db', fontSize: '12px' }
  },
  legend: {
    wrapperStyle: { color: '#d1d5db' }
  }
};

/**
 * Enhanced custom tooltip for dark theme
 */
export const CustomTooltip = ({ active, payload, label, labelFormatter, valueFormatter }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-gray-800 p-4 border border-cyan-400/30 rounded-xl shadow-2xl backdrop-blur-md max-w-xs">
        <p className="font-semibold text-white mb-3 text-sm border-b border-gray-700 pb-2">
          {labelFormatter ? labelFormatter(label) : toTitleCase(String(label))}
        </p>
        {payload.map((entry, index) => (
          <div key={index} className="flex items-center justify-between text-sm mb-1">
            <span className="flex items-center">
              <div 
                className="w-3 h-3 rounded-full mr-3 shadow-sm" 
                style={{ 
                  backgroundColor: entry.color, 
                  boxShadow: `0 0 6px ${entry.color}30` 
                }}
              />
              <span className="text-gray-300">{toTitleCase(entry.name)}:</span>
            </span>
            <span className="font-medium ml-3 text-white">
              {valueFormatter ? 
                valueFormatter(entry.value, entry.name) : 
                (typeof entry.value === 'number' ? entry.value.toLocaleString() : entry.value)
              }
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

/**
 * Transform backend data to chart-ready format
 */
export const transformChartData = (data) => {
  if (!data) return [];
  
  // If data is already in array format
  if (Array.isArray(data) && data.length > 0 && typeof data[0] === 'object') {
    return data;
  }
  
  // Handle Chart.js format with labels and datasets
  if (data.labels && data.datasets) {
    const labels = data.labels || [];
    const datasets = data.datasets || [];
    
    if (datasets.length === 0) return [];
    
    return labels.map((label, index) => {
      const dataPoint = { category: label };
      datasets.forEach((dataset, datasetIndex) => {
        const key = dataset.label || `value${datasetIndex}`;
        const value = dataset.data?.[index];
        dataPoint[key] = isNaN(Number(value)) ? value : Number(value);
      });
      return dataPoint;
    });
  }
  
  // Handle simple key-value object
  if (typeof data === 'object' && !Array.isArray(data)) {
    return Object.entries(data).map(([key, value]) => ({
      category: key,
      value: typeof value === 'number' ? value : Number(value) || 0
    }));
  }
  
  return [];
};

/**
 * Get data keys for multi-series charts
 */
export const getDataKeys = (data, excludeKeys = ['category']) => {
  if (!data || !Array.isArray(data) || data.length === 0) return [];
  
  const firstItem = data[0];
  return Object.keys(firstItem).filter(key => !excludeKeys.includes(key));
};

/**
 * Limit data points for better visualization
 */
export const limitDataPoints = (data, maxPoints = 20) => {
  if (!Array.isArray(data)) return [];
  return data.length > maxPoints ? data.slice(0, maxPoints) : data;
};

/**
 * Determine appropriate axis labels based on chart data and context
 */
export const getAxisLabels = (chartData, data) => {
  const title = (chartData?.title || '').toLowerCase();
  const firstItem = data?.[0] || {};
  const categoryKey = 'category';
  const dataKeys = getDataKeys(data);
  
  // Default labels
  let xAxisLabel = 'Category';
  let yAxisLabel = dataKeys[0] ? toTitleCase(dataKeys[0]) : 'Value';
  
  // Publication trends specific logic
  if (title.includes('publication') && title.includes('year')) {
    xAxisLabel = 'Year';
    yAxisLabel = 'Patent Count';
  }
  
  // Time-based charts
  if (title.includes('trend') || title.includes('time')) {
    // Check if category values look like years
    const categoryValue = firstItem[categoryKey];
    if (categoryValue && !isNaN(categoryValue) && categoryValue > 1900 && categoryValue < 2100) {
      xAxisLabel = 'Year';
    } else if (categoryValue && !isNaN(categoryValue) && categoryValue >= 1 && categoryValue <= 12) {
      xAxisLabel = 'Month';
    }
    
    // For trends, Y-axis is usually count/quantity
    if (title.includes('patent')) {
      yAxisLabel = 'Patent Count';
    } else if (title.includes('publication')) {
      yAxisLabel = 'Publications';
    } else if (dataKeys[0] && (dataKeys[0].includes('count') || dataKeys[0].includes('value'))) {
      yAxisLabel = 'Count';
    }
  }
  
  // Country/Geography charts
  if (title.includes('country') || title.includes('geographic')) {
    xAxisLabel = 'Country';
    yAxisLabel = title.includes('patent') ? 'Patent Count' : 'Count';
  }
  
  // SDG charts
  if (title.includes('sdg') || title.includes('sustainable development')) {
    xAxisLabel = 'SDG';
    yAxisLabel = 'Patent Count';
  }
    // Technology/Field charts
  if (title.includes('technology') || title.includes('field') || title.includes('ipc')) {
    xAxisLabel = 'Technology Field';
    yAxisLabel = 'Patent Count';
  }
  
  // Inventor charts
  if (title.includes('inventor')) {
    xAxisLabel = 'Inventors';
    yAxisLabel = 'Patent Count';
  }
  
  // Assignee/Company charts
  if (title.includes('assignee') || title.includes('company') || title.includes('organization')) {
    xAxisLabel = 'Assignees';
    yAxisLabel = 'Patent Count';
  }
  
  // Distribution/Percentage charts
  if (title.includes('distribution') || title.includes('percentage') || title.includes('proportion')) {
    if (dataKeys[0] && dataKeys[0].includes('percentage')) {
      yAxisLabel = 'Percentage (%)';
    }
  }
  
  return {
    xAxisLabel,
    yAxisLabel
  };
};
