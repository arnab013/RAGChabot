import React from 'react';
import BarChart from './BarChart';
import LineChart from './LineChart';
import PieChart from './PieChart';
import AreaChart from './AreaChart';
import StackedBarChart from './StackedBarChart';
import TreemapChart from './TreemapChart';

/**
 * Chart factory component that routes to appropriate chart type
 * This replaces the monolithic PatentChart component
 */
const ChartFactory = ({ chartData }) => {
  console.log('ChartFactory received chartData:', chartData);
  
  if (!chartData || !chartData.type || !chartData.data) {
    console.warn('Invalid chart data received:', chartData);
    return (
      <div className="flex items-center justify-center h-64 text-gray-400 bg-gray-800/50 rounded-xl border-2 border-dashed border-gray-600 backdrop-blur-sm my-6">
        <div className="text-center">
          <svg className="mx-auto h-8 w-8 text-gray-500 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <p className="text-sm text-gray-400">No chart data available</p>
        </div>
      </div>
    );
  }

  const { type } = chartData;

  // Route to appropriate chart component based on type
  switch (type.toLowerCase()) {    case 'bar':
      return <BarChart chartData={chartData} />;
        case 'horizontalbar':
    case 'horizontal-bar':
    case 'horizontal_bar':
      return <BarChart chartData={chartData} />;
      
    case 'line':
      return <LineChart chartData={chartData} />;
      
    case 'pie':
      return <PieChart chartData={chartData} />;
      
    case 'area':
      return <AreaChart chartData={chartData} />;
      
    case 'stacked_bar':
    case 'stacked-bar':
    case 'stackedbar':
      return <StackedBarChart chartData={chartData} />;
      
    case 'treemap':
      return <TreemapChart chartData={chartData} />;
      
    default:
      console.warn('Unsupported chart type:', type);
      return (
        <div className="flex items-center justify-center h-64 text-gray-400 bg-gray-800/50 rounded-xl border border-gray-600 my-6">
          <div className="text-center">
            <svg className="mx-auto h-8 w-8 text-gray-500 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-sm text-gray-400">Unsupported chart type: {type}</p>            <p className="text-xs text-gray-500 mt-1">
              Supported types: bar, horizontal-bar, line, pie, area, stacked_bar, treemap
            </p>
          </div>
        </div>
      );
  }
};

export default ChartFactory;
