import React from 'react';
import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Label } from 'recharts';
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

const LineChart = ({ chartData, height = 600, maxDataPoints = 50 }) => {
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
      chartType="line"
    >
      <RechartsLineChart 
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
          <Line
            key={key}
            type="monotone"
            dataKey={key}
            stroke={colors[index]}
            strokeWidth={3}
            dot={{ 
              fill: colors[index], 
              strokeWidth: 2, 
              r: 6,
              filter: `drop-shadow(0 0 6px ${colors[index]}50)`
            }}
            activeDot={{ 
              r: 8, 
              stroke: colors[index], 
              strokeWidth: 3, 
              fill: colors[index],
              filter: `drop-shadow(0 0 8px ${colors[index]}80)`
            }}
            name={toTitleCase(key)}
          />
        ))}
      </RechartsLineChart>
    </ChartBase>
  );
};

export default LineChart;
