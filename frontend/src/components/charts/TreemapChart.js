import React from 'react';
import { Treemap, Tooltip } from 'recharts';
import ChartBase from './ChartBase';
import { 
  transformChartData, 
  getChartColors, 
  limitDataPoints,
  toTitleCase
} from './chartUtils';

const TreemapChart = ({ chartData, height = 360, maxDataPoints = 25 }) => {
  // Transform and prepare data
  const transformedData = transformChartData(chartData.data);
  const limitedData = limitDataPoints(transformedData, maxDataPoints);
  const colors = getChartColors(limitedData.length);
  
  // Custom treemap content component
  const TreemapContent = ({ root, depth, x, y, width, height, index, name, value }) => {
    const isLargeEnough = width > 60 && height > 40;
    const fontSize = Math.min(width / 6, height / 4, 12);
    const color = colors[index % colors.length];
    
    return (
      <g>
        {/* Rectangle with gradient and glow effect */}
        <rect
          x={x}
          y={y}
          width={width}
          height={height}
          style={{
            fill: color,
            stroke: '#374151',
            strokeWidth: 2,
            filter: `drop-shadow(0 0 8px ${color}40)`,
            cursor: 'pointer'
          }}
          rx={4}
        />
        
        {/* Text label if rectangle is large enough */}
        {isLargeEnough && (
          <>
            <text
              x={x + width / 2}
              y={y + height / 2 - fontSize / 2}
              textAnchor="middle"
              fill="#ffffff"
              fontSize={fontSize}
              fontWeight="bold"
              style={{ 
                textShadow: '0 1px 3px rgba(0,0,0,0.8)',
                pointerEvents: 'none'
              }}
            >
              {toTitleCase(name)}
            </text>
            <text
              x={x + width / 2}
              y={y + height / 2 + fontSize}
              textAnchor="middle"
              fill="#d1d5db"
              fontSize={fontSize * 0.8}
              style={{ 
                textShadow: '0 1px 3px rgba(0,0,0,0.8)',
                pointerEvents: 'none'
              }}
            >
              {value.toLocaleString()}
            </text>
          </>
        )}
      </g>
    );
  };
  
  // Enhanced tooltip for treemap
  const TreemapTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0];
      const total = limitedData.reduce((sum, item) => sum + (item.value || 0), 0);
      const percentage = ((data.value / total) * 100).toFixed(1);
      
      return (
        <div className="bg-gray-800 p-4 border border-cyan-400/30 rounded-xl shadow-2xl backdrop-blur-md max-w-xs">
          <p className="font-semibold text-white mb-3 text-sm border-b border-gray-700 pb-2">
            {toTitleCase(data.name || data.category)}
          </p>
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-300">Value:</span>
              <span className="font-medium text-white">{data.value.toLocaleString()}</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-300">Percentage:</span>
              <span className="font-medium text-cyan-400">{percentage}%</span>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };
  
  return (
    <ChartBase 
      title={chartData.title} 
      height={height}
      dataCount={limitedData.length}
      chartType="treemap"
    >
      <Treemap
        data={limitedData}
        dataKey="value"
        ratio={4/3}
        stroke="#374151"
        content={<TreemapContent />}
      >
        <Tooltip content={<TreemapTooltip />} />
      </Treemap>
    </ChartBase>
  );
};

export default TreemapChart;
