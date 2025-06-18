# Chart Components Architecture

This directory contains modular chart components that replace the monolithic `PatentChart.js` component. Each chart type now has its own component with specialized formatting and optimizations.

## Component Structure

### Core Components

#### `ChartBase.js`
- Base wrapper component with consistent dark theme styling
- Provides common layout, title, and footer for all charts
- Handles responsive container setup

#### `ChartFactory.js`
- Router component that determines which chart to render based on `chartData.type`
- Replaces the old `PatentChart.js` component
- Supports all chart types with fallback for unknown types

#### `chartUtils.js`
- Shared utilities for data transformation and styling
- Contains color palettes, theme constants, and helper functions
- Provides common tooltip component and data processing functions

### Individual Chart Components

#### `BarChart.js`
- Optimized for categorical data visualization
- Features rounded corners, enhanced tooltips
- Default max data points: 20
- Best for comparing discrete categories

#### `LineChart.js`
- Perfect for time-series and trend data
- Enhanced with glowing dots and smooth animations
- Default max data points: 50
- Features interactive hover effects

#### `PieChart.js`
- Specialized for proportion visualization
- Smart label hiding for small slices (<3%)
- Enhanced tooltip showing percentages
- Default max data points: 15

#### `AreaChart.js`
- Ideal for showing data volume over time
- Semi-transparent fills with gradient effects
- Good for multiple overlapping series
- Default max data points: 50

#### `StackedBarChart.js`
- Shows composition of categories
- Enhanced tooltip displays individual values and totals
- Rounded corners only on top bars
- Default max data points: 20

#### `TreemapChart.js`
- Hierarchical data visualization
- Smart text sizing based on rectangle size
- Glowing effects and interactive hover states
- Default max data points: 25

## Usage

### Basic Usage
```javascript
import { ChartFactory } from './components/charts';

// Replace old PatentChart usage
<ChartFactory chartData={{
  type: 'bar',
  title: 'Patent Distribution',
  data: chartData
}} />
```

### Using Individual Components
```javascript
import { BarChart, LineChart, PieChart } from './components/charts';

// Use specific chart types directly
<BarChart chartData={data} height={400} maxDataPoints={15} />
<LineChart chartData={data} height={500} maxDataPoints={100} />
<PieChart chartData={data} height={400} maxDataPoints={10} />
```

## Benefits of Modular Architecture

### 1. **Maintainability**
- Each chart type is in its own file
- Easier to debug and modify specific chart behaviors
- Clear separation of concerns

### 2. **Performance**
- Only load chart components that are needed
- Smaller bundle size per chart type
- Better tree-shaking support

### 3. **Customization**
- Each chart can have chart-specific optimizations
- Different default settings per chart type
- Specialized tooltips and interactions

### 4. **Extensibility**
- Easy to add new chart types
- Individual components can evolve independently
- Custom props per chart type

### 5. **Testing**
- Each chart component can be tested independently
- Easier to write focused unit tests
- Better code coverage

## Chart-Specific Features

### BarChart
- Rounded top corners
- Optimal spacing for readability
- Color-coded series

### LineChart
- Glowing dot effects
- Smooth line curves
- Enhanced active states

### PieChart
- Intelligent label management
- Percentage-based tooltips
- Optimized for mobile viewing

### AreaChart
- Gradient fills
- Smooth area curves
- Good for overlapping data

### StackedBarChart
- Total calculation in tooltips
- Progressive rounding
- Series composition visualization

### TreemapChart
- Dynamic text sizing
- Hierarchical color coding
- Responsive rectangle sizing

## Data Format

All charts expect data in a consistent format:

```javascript
{
  type: 'bar|line|pie|area|stacked_bar|treemap',
  title: 'Chart Title',
  data: [
    { category: 'Item 1', value: 100 },
    { category: 'Item 2', value: 200 }
    // ... more data points
  ]
}
```

## Theme Integration

All charts use the consistent dark theme with:
- Cyan accent colors (#06b6d4, #22d3ee)
- Dark backgrounds (gray-800/90 to gray-900/90)
- Proper contrast ratios for accessibility
- Glowing effects and shadows for modern appeal

## Future Enhancements

Potential improvements for the chart architecture:
- Add more chart types (scatter, radar, etc.)
- Implement chart animations
- Add export functionality
- Create chart editing interfaces
- Add accessibility features (ARIA labels, keyboard navigation)
