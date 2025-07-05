import { formatRelative } from 'date-fns';
import { useDispatch, useSelector } from 'react-redux';
import React, { useState, useEffect, useContext } from 'react';
import Chart from 'react-apexcharts';
import {
  Box,
  Typography,
  Avatar,
  Paper,
  Fade,
  Chip,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  ShowChart as ChartIcon,
  AttachMoney as MoneyIcon,
} from '@mui/icons-material';

import {setModalQRCodeStatus, setModalQRCodeLoadingExecutionStatus, setModalQRCodeLoadingStatus, setModalQRCodeScan, setModalQRCodeSell, CustomerRetailFSOP} from 'store/actions';

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


var chartDataOptions = {
  chart: {
    height: 350,
    type: "line",
    stacked: false
  },
  dataLabels: {
    enabled: false
  },
  colors: ["#FF1654", "#247BA0"],
  series: [
    {
      name: "Series A",
      data: [1.4, 2, 2.5, 1.5, 2.5, 2.8, 3.8, 4.6]
    },
    {
      name: "Series B",
      data: [20, 29, 37, 36, 44, 45, 50, 58]
    }
  ],
  stroke: {
    width: [4, 4]
  },
  plotOptions: {
    bar: {
      columnWidth: "20%"
    }
  },
  xaxis: {
    categories: [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
  },
  yaxis: [
    {
      axisTicks: {
        show: true
      },
      axisBorder: {
        show: true,
        color: "#FF1654"
      },
      labels: {
        style: {
          colors: "#FF1654"
        }
      },
      title: {
        text: "Series A",
        style: {
          color: "#FF1654"
        }
      }
    },
    {
      opposite: true,
      axisTicks: {
        show: true
      },
      axisBorder: {
        show: true,
        color: "#247BA0"
      },
      labels: {
        style: {
          colors: "#247BA0"
        }
      },
      title: {
        text: "Series B",
        style: {
          color: "#247BA0"
        }
      }
    }
  ],
  tooltip: {
    shared: false,
    intersect: true,
    x: {
      show: false
    }
  },
  legend: {
    horizontalAlign: "left",
    offsetX: 40
  }
};

let chartData2 = {
  type: "area",
  height: 80,
  width: '100%',
  offsetX: 0,
  options: {
    chart: {
      sparkline: {
        enabled: true,
      },
      background: "#333",
    },
    colors: ["#0F0"],
    dataLabels: {
      enabled: false,
    },
    fill: {
        type: "solid",
        gradient: {
            shadeIntensity: 1,
            opacityFrom: 0.5,
            opacityTo: 0.9,
            stops: [0, 90, 100]
        }
    },
    stroke: {
      curve: "smooth",
      width: 3,
    },
    yaxis: {
        show: "true",
        offsetY: 40
  },
    legend: {
        position: 'bottom',
    },
    xaxis: {
      offsetX: -10,
      categories: [],
      show: "false",
      title: {
        text: "Weekly",
      },
      labels: {
         formatter: function (value) {
            return value;
         }
      },
    },
    tooltip: {
      theme: "dark",
      fixed: {
        enabled: true,
      },
      x: {
        show: false,
      },
      y: {
        title: "FSOP",
        show: "false",
      },
      marker: {
        show: false,
      },
    },
  },
  series: [
  ],
};


let  cseries = [
    {
      name: "Forecast",
      data: [100, 175, 333, 500, 555],
    },
];

export const  MobileMarketData = (props) => {
    const dispatch = useDispatch();
    const [liveMD, setLiveMD] = useState({});

    useEffect(() => {
        if (props.estimuli && props.msg && (props.estimuli === "LiveMarketEvent")) {
            const values = JSON.parse(props.msg);
            console.log('------------- LiveMarketEvent -------------')
            console.log({values})
            setLiveMD(values);
        }
    }, [props.estimuli, props.msg]);

    return (
        <Box sx={{ mb: 2, px: 1 }}>
            <Box sx={{ 
                display: 'flex', 
                alignItems: 'flex-end',
                gap: 1
            }}>
                {/* Avatar */}
                <Box sx={{ 
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 0.5
                }}>
                    <Avatar sx={{ 
                        bgcolor: '#e74c3c',
                        width: 32,
                        height: 32
                    }}>
                        <MoneyIcon sx={{ fontSize: 18 }} />
                    </Avatar>
                    <Typography 
                        variant="caption" 
                        sx={{ 
                            color: '#7f8c8d',
                            fontSize: '0.7rem',
                            mt: 0.5,
                            display: 'block'
                        }}
                    >
                        {formatDate(new Date())}
                    </Typography>
                </Box>

                {/* Market Data Content */}
                <Box sx={{ 
                    display: 'flex',
                    flexDirection: 'column',
                    maxWidth: '75%'
                }}>
                    <Paper
                        elevation={1}
                        sx={{
                            p: 2,
                            backgroundColor: '#fff3cd',
                            color: '#856404',
                            borderRadius: '18px 18px 18px 4px',
                            marginRight: 'auto',
                            marginLeft: '8px',
                            maxWidth: '85%',
                            wordWrap: 'break-word',
                            border: '1px solid #ffeaa7'
                        }}
                    >
                        <Box sx={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            gap: 1, 
                            mb: 1 
                        }}>
                            <TrendingUpIcon sx={{ color: '#e74c3c', fontSize: 20 }} />
                            <Typography variant="subtitle2" sx={{ 
                                fontWeight: 600,
                                color: '#856404'
                            }}>
                                Live Market Data
                            </Typography>
                        </Box>
                        
                        <Box
                            dangerouslySetInnerHTML={{ __html: props.msg }}
                            sx={{
                                fontSize: '0.95rem',
                                lineHeight: 1.5,
                                '& table': {
                                    width: '100%',
                                    borderCollapse: 'collapse',
                                    fontSize: '0.85rem',
                                    backgroundColor: 'white',
                                    borderRadius: '8px',
                                    overflow: 'hidden'
                                },
                                '& th, & td': {
                                    border: '1px solid #ddd',
                                    padding: '8px',
                                    textAlign: 'left'
                                },
                                '& th': {
                                    backgroundColor: '#f8f9fa',
                                    fontWeight: 'bold',
                                    color: '#495057'
                                },
                                '& td': {
                                    color: '#495057'
                                }
                            }}
                        />
                    </Paper>

                    {/* Event Type Badge */}
                    <Chip
                        label={`${props.etype}: ${props.estate} (${props.estimuli})`}
                        size="small"
                        sx={{
                            mt: 0.5,
                            fontSize: '0.7rem',
                            height: '20px',
                            backgroundColor: '#e74c3c',
                            color: 'white',
                            '& .MuiChip-label': {
                                px: 1
                            }
                        }}
                    />
                </Box>
            </Box>
        </Box>
    );
};

