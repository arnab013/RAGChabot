import React from 'react';
import { PieChart as RechartsPieChart, Pie, Cell, Tooltip, Legend } from 'recharts';
import ChartBase from './ChartBase';
import { 
  transformChartData, 
  getChartColors, 
  limitDataPoints,
  toTitleCase, 
  CHART_THEME
} from './chartUtils';

const PieChart = ({ chartData, height = 480, maxDataPoints = 15 }) => {
  // Transform and prepare data
  const transformedData = transformChartData(chartData.data);
  const limitedData = limitDataPoints(transformedData, maxDataPoints);
  const colors = getChartColors(limitedData.length);
  
  // Custom label function for pie chart
  const renderLabel = ({ name, percent, value }) => {
    if (percent < 0.03) return ''; // Hide labels for slices smaller than 3%
    return `${toTitleCase(name)}: ${(percent * 100).toFixed(1)}%`;
  };
  
  // Enhanced tooltip for pie chart
  const PieTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0];
      const total = limitedData.reduce((sum, item) => sum + (item.value || 0), 0);
      const percentage = ((data.value / total) * 100).toFixed(1);
      
      return (
        <div className="bg-gray-800 p-4 border border-cyan-400/30 rounded-xl shadow-2xl backdrop-blur-md max-w-xs">
          <p className="font-semibold text-white mb-3 text-sm border-b border-gray-700 pb-2">
            {toTitleCase(data.name)}
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
      chartType="pie"
    >
      <RechartsPieChart>
        <Pie
          data={limitedData}
          cx="50%"
          cy="50%"
          outerRadius={Math.min(180, height * 0.35)}
          innerRadius={0}
          dataKey="value"
          nameKey="category"
          label={renderLabel}
          labelLine={false}
          fontSize={11}
          fill="#d1d5db"
        >
          {limitedData.map((entry, index) => (
            <Cell 
              key={`cell-${index}`} 
              fill={colors[index]}
              stroke={colors[index]}
              strokeWidth={2}
              style={{
                filter: `drop-shadow(0 0 8px ${colors[index]}40)`,
                cursor: 'pointer'
              }}
            />
          ))}
        </Pie>
        
        <Tooltip content={<PieTooltip />} />
        
        <Legend 
          {...CHART_THEME.legend}
          verticalAlign="bottom"
          height={36}
          iconType="circle"
          formatter={(value) => toTitleCase(value)}
        />
      </RechartsPieChart>
    </ChartBase>
  );
};

export default PieChart;
