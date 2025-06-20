import React from 'react';
import { BarChart as RechartsBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Label } from 'recharts';
import ChartBase from './ChartBase';
import { 
  transformChartData,
  getDataKeys, 
  getChartColors, 
  limitDataPoints,
  toTitleCase, 
  CHART_THEME,
  getAxisLabels
} from './chartUtils';

const StackedBarChart = ({ chartData, height = 600, maxDataPoints = 25 }) => {
  // Transform and prepare data
  const transformedData = transformChartData(chartData.data);
  const limitedData = limitDataPoints(transformedData, maxDataPoints);
  const dataKeys = getDataKeys(limitedData);
  const colors = getChartColors(dataKeys.length);
  
  // Get dynamic axis labels
  const { xAxisLabel, yAxisLabel } = getAxisLabels(chartData, limitedData);
  
  // Chart configuration - symmetric margins
  const margin = { top: 5, right: 20, left: 20, bottom: 5 };
  
  // Enhanced tooltip for stacked bars
  const StackedTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const total = payload.reduce((sum, item) => sum + (item.value || 0), 0);
      
      return (
        <div className="bg-gray-800 p-4 border border-cyan-400/30 rounded-xl shadow-2xl backdrop-blur-md max-w-xs">
          <p className="font-semibold text-white mb-3 text-sm border-b border-gray-700 pb-2">
            {toTitleCase(String(label))}
          </p>
          <div className="space-y-2">
            {payload.map((entry, index) => (
              <div key={index} className="flex items-center justify-between text-sm">
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
                <span className="font-medium text-white">
                  {entry.value.toLocaleString()}
                </span>
              </div>
            ))}
            <div className="border-t border-gray-700 pt-2 mt-3">
              <div className="flex items-center justify-between text-sm font-semibold">
                <span className="text-gray-300">Total:</span>
                <span className="text-cyan-400">{total.toLocaleString()}</span>
              </div>
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
      chartType="stacked bar"
    >
      <RechartsBarChart 
        data={limitedData} 
        margin={margin}
      >
        <CartesianGrid {...CHART_THEME.grid} />
          <XAxis 
          dataKey="category"
          {...CHART_THEME.axis}
          angle={-45}
          textAnchor="end"
          height={60}
        >
          <Label 
            value={xAxisLabel} 
            offset={0} 
            position="insideBottom"
            {...CHART_THEME.label}
          />
        </XAxis>
        
        <YAxis {...CHART_THEME.axis}>
          <Label 
            value={yAxisLabel} 
            angle={-90} 
            position="insideLeft"
            {...CHART_THEME.label}
          />
        </YAxis>
        
        <Tooltip 
          content={<StackedTooltip />}
        />
        
        <Legend {...CHART_THEME.legend} />
        
        {dataKeys.map((key, index) => (
          <Bar
            key={key}
            dataKey={key}
            stackId="stack"
            fill={colors[index]}
            name={toTitleCase(key)}
            radius={index === dataKeys.length - 1 ? [4, 4, 0, 0] : [0, 0, 0, 0]}
          />
        ))}
      </RechartsBarChart>
    </ChartBase>
  );
};

export default StackedBarChart;
