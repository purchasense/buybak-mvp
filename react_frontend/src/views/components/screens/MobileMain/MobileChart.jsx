import { formatRelative } from 'date-fns';
import React, { useContext } from 'react';
import Chart from 'react-apexcharts';
import {
  Box,
  Typography,
  Card,
  CardContent,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  ShowChart as ChartIcon,
} from '@mui/icons-material';

const formatDate = date => {
  let formattedDate = '';

  if (date) {
    // Convert the date in words relative to the current date
    formattedDate = formatRelative(date, new Date());

    // Uppercase the first letter
    formattedDate =
      formattedDate.charAt(0).toUpperCase() + formattedDate.slice(1);
  }
  
  return formattedDate;
};

let chartData = {
  type: "area",
  height: 200,
  width: '100%',
  options: {
    chart: {
      sparkline: {
        enabled: false,
      },
      background: "transparent",
      toolbar: {
        show: false
      }
    },
    colors: ["#3498db"],
    dataLabels: {
      enabled: false,
    },
    fill: {
        type: "gradient",
        gradient: {
            shadeIntensity: 1,
            opacityFrom: 0.7,
            opacityTo: 0.3,
            stops: [0, 90, 100],
            colorStops: [
                {
                    offset: 0,
                    color: "#3498db",
                    opacity: 0.7
                },
                {
                    offset: 100,
                    color: "#3498db",
                    opacity: 0.1
                }
            ]
        }
    },
    stroke: {
      curve: "smooth",
      width: 3,
      colors: ["#3498db"]
    },
    grid: {
      show: false
    },
    yaxis: {
        show: false
    },
    xaxis: {
      show: false
    },
    tooltip: {
      theme: "dark",
      x: {
        show: false,
      },
      y: {
        title: "Value",
        formatter: function (value) {
          return "$" + value.toLocaleString();
        }
      },
      marker: {
        show: false,
      },
    },
  },
  series: [
    {
      name: "Forecast",
      data: [166.76558, 160.91374, 157.94542, 157.94542, 159.91353, 158.04843]
    },
  ],
};

const MobileChart = (props) => {
    chartData.series[0].data = props.data;
    console.log('CD_DATA---- MobileChart');
    console.log({props});
    console.log(chartData.series[0].data);

    return (
        <Box sx={{ mb: 2 }}>
            <Card sx={{ 
                borderRadius: 3,
                overflow: 'hidden',
                boxShadow: '0 4px 20px rgba(0,0,0,0.1)'
            }}>
                <CardContent sx={{ p: 0 }}>
                    <Box sx={{ 
                        p: 3, 
                        backgroundColor: '#ecf0f1',
                        borderBottom: '1px solid #d5dbdb'
                    }}>
                        <Box sx={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            gap: 1,
                            mb: 1
                        }}>
                            <TrendingUpIcon sx={{ color: '#27ae60', fontSize: 20 }} />
                            <Typography variant="h6" sx={{ 
                                color: '#2c3e50',
                                fontWeight: 600
                            }}>
                                Price Forecast
                            </Typography>
                        </Box>
                        <Typography variant="body2" sx={{ 
                            color: '#7f8c8d'
                        }}>
                            Historical price trends and predictions
                        </Typography>
                    </Box>
                    <Box sx={{ p: 2 }}>
                        <Chart
                            type="area"
                            height={200}
                            width="100%"
                            options={chartData.options}
                            series={chartData.series}
                        />
                    </Box>
                </CardContent>
            </Card>
        </Box>
    );
};

export default MobileChart;
