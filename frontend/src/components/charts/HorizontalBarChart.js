import React from 'react';
import { BarChart as RechartsBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import ChartBase from './ChartBase';
import { 
  transformChartData, 
  getDataKeys, 
  getChartColors, 
  limitDataPoints,
  toTitleCase, 
  CHART_THEME, 
  CustomTooltip
} from './chartUtils';

const HorizontalBarChart = ({ chartData, height = 600, maxDataPoints = 20 }) => {
  console.log('🔍 HorizontalBarChart - Input chartData:', JSON.stringify(chartData, null, 2));
  
  // Transform and prepare data
  const transformedData = transformChartData(chartData.data);
  console.log('🔍 HorizontalBarChart - Transformed data:', JSON.stringify(transformedData.slice(0, 3), null, 2));
  
  const limitedData = limitDataPoints(transformedData, maxDataPoints);
  const dataKeys = getDataKeys(limitedData);
  console.log('🔍 HorizontalBarChart - Data keys:', dataKeys);
  console.log('🔍 HorizontalBarChart - Sample limited data:', JSON.stringify(limitedData.slice(0, 3), null, 2));
  const colors = getChartColors(dataKeys.length);
  
  // Get the max value for proper domain setting
  const maxValue = Math.max(...limitedData.map(item => 
    Math.max(...dataKeys.map(key => Number(item[key]) || 0))
  ));
  console.log('🔍 HorizontalBarChart - Max value:', maxValue);
  
  // Chart configuration for horizontal layout
  const margin = { top: 20, right: 30, left: 100, bottom: 50 };
    return (
    <ChartBase 
      title={chartData.title} 
      height={height}
      dataCount={limitedData.length}
      chartType="horizontal-bar"
    >
      <RechartsBarChart 
        data={limitedData} 
        margin={margin}
        layout="horizontal"
        width={800}
        height={height - 100}
      >
        <CartesianGrid {...CHART_THEME.grid} />
        
        {/* For horizontal charts, X is numeric (values) and Y is categorical (categories) */}
        <XAxis 
          type="number"
          domain={[0, maxValue * 1.1]}
          {...CHART_THEME.axis}
        />
        
        <YAxis 
          type="category"
          dataKey="category"
          {...CHART_THEME.axis}
          width={90}
        />
        
        <Tooltip 
          content={<CustomTooltip />}
        />
        
        <Legend {...CHART_THEME.legend} />
        
        {dataKeys.map((key, index) => (
          <Bar
            key={key}
            dataKey={key}
            fill={colors[index]}
            radius={[0, 6, 6, 0]}
            name={toTitleCase(key)}
          />
        ))}
      </RechartsBarChart>
    </ChartBase>
  );
};

export default HorizontalBarChart;
