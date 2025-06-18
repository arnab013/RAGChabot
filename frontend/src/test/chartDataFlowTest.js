/**
 * Test file to verify chart data flow from backend to frontend
 * This file tests the data transformation and chart rendering
 */

// Sample backend response data formats that we expect from the API
const sampleBackendResponses = {
  // Bar chart data - from stats.get_sdg_distribution()
  barChart: {
    type: 'bar',
    title: 'SDG Distribution',
    data: {
      labels: ['SDG 7', 'SDG 13', 'SDG 9', 'SDG 6', 'SDG 3'],
      datasets: [{
        label: 'Patents',
        data: [1250, 980, 875, 650, 420],
        backgroundColor: '#3B82F6',
        borderColor: '#3B82F6',
        borderWidth: 1
      }]
    }
  },

  // Line chart data - from stats.get_publication_trends()
  lineChart: {
    type: 'line',
    title: 'Publication Trends Over Time',
    data: {
      labels: ['2020', '2021', '2022', '2023', '2024'],
      datasets: [{
        label: 'Patents',
        data: [150, 180, 220, 280, 350],
        borderColor: '#3B82F6',
        backgroundColor: '#3B82F610',
        fill: false,
        tension: 0.4
      }]
    }
  },

  // Pie chart data - from SDG distribution percentage
  pieChart: {
    type: 'pie',
    title: 'Technology Field Distribution',
    data: {
      labels: ['Clean Energy', 'Water Tech', 'Healthcare', 'Agriculture', 'Transportation'],
      datasets: [{
        data: [35, 25, 20, 12, 8],
        backgroundColor: ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6'],
        borderWidth: 2
      }]
    }
  },

  // Area chart data - cumulative growth
  areaChart: {
    type: 'area',
    title: 'Cumulative Patent Growth',
    data: {
      labels: ['2020', '2021', '2022', '2023', '2024'],
      datasets: [{
        label: 'Cumulative Patents',
        data: [150, 330, 550, 830, 1180],
        borderColor: '#3B82F6',
        backgroundColor: '#3B82F630',
        fill: true,
        tension: 0.4
      }]
    }
  },

  // Stacked bar chart data - multi-series
  stackedBarChart: {
    type: 'stacked_bar',
    title: 'Patents by SDG and Year',
    data: {
      labels: ['2022', '2023', '2024'],
      datasets: [
        {
          label: 'SDG 7 (Clean Energy)',
          data: [45, 52, 63],
          backgroundColor: '#3B82F6'
        },
        {
          label: 'SDG 13 (Climate)',
          data: [38, 44, 51],
          backgroundColor: '#EF4444'
        },
        {
          label: 'SDG 9 (Innovation)',
          data: [32, 39, 47],
          backgroundColor: '#10B981'
        }
      ]
    }
  },

  // Treemap data - hierarchical
  treemap: {
    type: 'treemap',
    title: 'Technology Categories',
    data: [
      { name: 'Solar Energy', value: 450, fill: '#3B82F6' },
      { name: 'Wind Power', value: 380, fill: '#EF4444' },
      { name: 'Energy Storage', value: 320, fill: '#10B981' },
      { name: 'Smart Grid', value: 280, fill: '#F59E0B' },
      { name: 'Hydroelectric', value: 220, fill: '#8B5CF6' },
      { name: 'Geothermal', value: 180, fill: '#06B6D4' }
    ]
  }
};

// Test prompts that should generate specific chart types
const testPrompts = {
  barChart: [
    "Show SDG distribution across patents",
    "Which SDG has the most patents?",
    "Display patent counts by country",
    "What are the top technology fields?"
  ],
  
  lineChart: [
    "Show publication trends over the last 10 years",
    "Display yearly patent growth trends",
    "Show patent filing trends by technology",
    "Plot patent filing trends over time"
  ],
  
  pieChart: [
    "Show percentage distribution of patents by SDG",
    "What's the proportion of patents by technology type?",
    "Display patent distribution by geographic region",
    "Show breakdown of renewable vs non-renewable patents"
  ],
  
  areaChart: [
    "Show cumulative patent growth over time",
    "Display overlapping SDG patent volumes",
    "Show stacked patent trends by technology",
    "Plot cumulative innovation in clean energy"
  ],
  
  stackedBarChart: [
    "Show stacked bar chart of SDG patents by year",
    "Display comparative analysis of patent volumes",
    "Show multi-dimensional patent data visualization",
    "Create stacked view of technology trends"
  ],
  
  treemap: [
    "Create a treemap of patent technology categories",
    "Show hierarchical view of patent classifications",
    "Display complex patent relationship mappings",
    "Create hierarchical technology breakdown"
  ]
};

// Expected API response format
const expectedApiResponse = {
  message: "Here's the statistical analysis you requested...",
  chart: {
    type: 'bar',  // or 'line', 'pie', 'area', 'stacked_bar', 'treemap'
    title: 'Chart Title',
    data: {
      // Chart.js format or direct array format
    }
  },
  error: null
};

// Instructions for testing
console.log(`
=== CHART DATA FLOW TESTING GUIDE ===

1. BACKEND TESTING:
   - Backend should be running on http://localhost:5000
   - Test API endpoint: POST /api/search
   - Use test prompts from testPrompts object above

2. FRONTEND TESTING:
   - Frontend should be running on http://localhost:3000
   - Check browser developer console for chart data logs
   - Verify ChartFactory routes data correctly

3. TEST PROCEDURE:
   a) Open browser developer console
   b) Send chart-specific prompts from the example table
   c) Check console logs for:
      - "ChartFactory received chartData:"
      - "Chart data from message.chartData:"
   d) Verify charts render correctly with proper styling

4. EXPECTED CHART TYPES:
   - Bar Charts: SDG distribution, counts by category
   - Line Charts: Trends over time, growth patterns  
   - Pie Charts: Percentage distributions, proportions
   - Area Charts: Cumulative data, overlapping series
   - Stacked Bar Charts: Multi-series comparisons
   - Treemap Charts: Hierarchical data visualization

5. DATA TRANSFORMATION VERIFICATION:
   - Backend returns Chart.js format (labels + datasets)
   - Frontend transforms to Recharts format (array of objects)
   - Check chartUtils.transformChartData() function

6. VISUAL VERIFICATION:
   - Dark theme consistency
   - Proper color schemes (cyan/blue palette)
   - Chart responsiveness and sizing
   - Tooltip functionality
   - Legend display
`);

export { sampleBackendResponses, testPrompts, expectedApiResponse };
