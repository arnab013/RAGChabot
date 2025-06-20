import React from 'react';
import { ResponsiveContainer } from 'recharts';

/**
 * Base chart wrapper component with consistent styling
 */
const ChartBase = ({ 
  title, 
  children, 
  height = 600, 
  dataCount = 0, 
  chartType = 'chart',
  className = ''
}) => {
  return (
    <div className={`w-full bg-gradient-to-br from-gray-800/90 to-gray-900/90 rounded-xl shadow-2xl border border-cyan-400/20 p-6 my-6 backdrop-blur-sm ${className}`}>
      {/* Chart Header */}
      {title && (
        <div className="mb-4">
          <h3 className="text-xl font-semibold text-white text-center mb-2">{title}</h3>
          <div className="h-0.5 bg-gradient-to-r from-transparent via-cyan-400 to-transparent mx-auto w-32" />
        </div>
      )}      {/* Chart Container with space for top legend */}
      <div className="w-full" style={{ height: `${height}px` }}>
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ChartBase;
