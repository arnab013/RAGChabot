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
  CustomTooltip,
  getAxisLabels
} from './chartUtils';

const BarChart = ({ chartData, height = 600, maxDataPoints = 20 }) => {
  // Transform and prepare data
  const transformedData = transformChartData(chartData.data);
  const limitedData = limitDataPoints(transformedData, maxDataPoints);
  const dataKeys = getDataKeys(limitedData);
  const colors = getChartColors(dataKeys.length);
    // Get dynamic axis labels
  const { xAxisLabel, yAxisLabel } = getAxisLabels(chartData, limitedData);
  
  // Chart configuration - symmetric margins
  const margin = { top: 5, right: 20, left: 20, bottom: 5 };
  
  return (
    <ChartBase 
      title={chartData.title} 
      height={height}
      dataCount={limitedData.length}
      chartType="bar"
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
          content={<CustomTooltip />}
        />
        
        <Legend {...CHART_THEME.legend} />
        
        {dataKeys.map((key, index) => (
          <Bar
            key={key}
            dataKey={key}
            fill={colors[index]}
            radius={[6, 6, 0, 0]}
            name={toTitleCase(key)}
          />
        ))}
      </RechartsBarChart>
    </ChartBase>
  );
};

export default BarChart;
