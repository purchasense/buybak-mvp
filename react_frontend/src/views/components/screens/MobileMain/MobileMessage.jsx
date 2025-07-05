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
  Person as PersonIcon,
  SmartToy as BotIcon,
  TrendingUp as TrendingUpIcon,
  WineBar as WineIcon,
} from '@mui/icons-material';

import {setModalQRCodeScan} from 'store/actions';

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

export const  MobileMessage = (props) => {
    const dispatch = useDispatch();
    let [cData, setCData] = useState([]);
    let [liveMD, setLiveMD] = useState({});
    let [prefix, setPrefix] = useState("");

    const currentUser = 'sameer';
    const isUserMessage = props.user === currentUser;
    const isBotMessage = props.user === 'GPT';
    const isSystemMessage = !isUserMessage && !isBotMessage;

    let isChart = false;
    let isLiveMD = false;

    useEffect(() => {
        setPrefix(props.msg);
        if ( props.outline && props.msg && props.outline === "BUY")
        {
            const values = JSON.parse(props.msg);
            console.log('------------- BUY -------------')
            console.log({values})
            dispatch(setModalQRCodeScan(values.wine, values.quantity, values.price));
            const fix = '<html><body><img align="top" style={{position:"relative",right:"1px",top:"-30px"}} width="35px" src="/images/ShoppingCartIcon.png" /> &nbsp;&nbsp; Bought ' + values["wine"] + ' ' + Number(values["quantity"]/100.0) + ' @ $' + Number(values["price"] / 10000.0).toFixed(2) + '</body></html>';
            setPrefix(fix);
            console.log(fix);
        }
        else if ( props.estimuli && props.msg && (props.estimuli === "ForecastEvent"))
        {
            console.log('------------- WineForecast -------------')
            console.log(props.msg);
            try {
                const values = JSON.parse(props.msg);
                console.log({values})
                Object.keys(values).map((key) => {
                    console.log( values[key]);
                    setCData(values[key]);
                    console.log(cData);
                    isChart = true;
                });
            } catch(e ) {
                console.log(e);
            }

        }
        else if ( props.estimuli && props.msg && (props.estimuli === "LiveMarketEvent"))
        {
            const values = JSON.parse(props.msg);
            console.log('------------- LiveMarketEvent -------------')

            console.log({values})
            setLiveMD(values);
            isLiveMD = true;
        }
    }, []);

    if ( cData.length > 0)
    {
        isChart = true;
        console.log( cData);
        cseries[0].data = cData;
    }

    const getMessageStyle = () => {
        if (isUserMessage) {
            return {
                backgroundColor: '#3498db',
                color: 'white',
                borderRadius: '18px 18px 4px 18px',
                marginLeft: 'auto',
                marginRight: '8px',
                maxWidth: '85%',
                wordWrap: 'break-word'
            };
        } else if (isBotMessage) {
            return {
                backgroundColor: '#ecf0f1',
                color: '#2c3e50',
                borderRadius: '18px 18px 18px 4px',
                marginRight: 'auto',
                marginLeft: '8px',
                maxWidth: '85%',
                wordWrap: 'break-word',
                border: '1px solid #d5dbdb'
            };
        } else {
            return {
                backgroundColor: '#f39c12',
                color: 'white',
                borderRadius: '12px',
                margin: '8px auto',
                maxWidth: '90%',
                wordWrap: 'break-word',
                textAlign: 'center'
            };
        }
    };

    const getAvatar = () => {
        if (isUserMessage) {
            return (
                <Avatar sx={{ 
                    bgcolor: '#3498db',
                    width: 32,
                    height: 32,
                    fontSize: '0.9rem'
                }}>
                    <PersonIcon sx={{ fontSize: 18 }} />
                </Avatar>
            );
        } else if (isBotMessage) {
            return (
                <Avatar sx={{ 
                    bgcolor: '#2c3e50',
                    width: 32,
                    height: 32
                }}>
                    <BotIcon sx={{ fontSize: 18 }} />
                </Avatar>
            );
        } else {
            return (
                <Avatar sx={{ 
                    bgcolor: '#f39c12',
                    width: 28,
                    height: 28
                }}>
                    <WineIcon sx={{ fontSize: 16 }} />
                </Avatar>
            );
        }
    };

    const getTimestamp = () => {
        return (
            <Typography 
                variant="caption" 
                sx={{ 
                    color: isUserMessage ? '#bdc3c7' : '#7f8c8d',
                    fontSize: '0.7rem',
                    mt: 0.5,
                    display: 'block'
                }}
            >
                {formatDate(new Date())}
            </Typography>
        );
    };

    return (
        <Box sx={{ mb: 2, px: 1 }}>
            <Box sx={{ 
                display: 'flex', 
                alignItems: 'flex-end',
                gap: 1,
                flexDirection: isUserMessage ? 'row-reverse' : 'row'
            }}>
                {/* Avatar */}
                <Box sx={{ 
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 0.5
                }}>
                    {getAvatar()}
                    {getTimestamp()}
                </Box>

                {/* Message Content */}
                <Box sx={{ 
                    display: 'flex',
                    flexDirection: 'column',
                    maxWidth: '75%'
                }}>
                    <Paper
                        elevation={1}
                        sx={{
                            p: 2,
                            ...getMessageStyle(),
                            '& pre': {
                                backgroundColor: isUserMessage ? 'rgba(255,255,255,0.1)' : '#f8f9fa',
                                borderRadius: '8px',
                                padding: '12px',
                                overflow: 'auto',
                                fontSize: '0.85rem',
                                fontFamily: 'monospace',
                                margin: '8px 0'
                            },
                            '& code': {
                                backgroundColor: isUserMessage ? 'rgba(255,255,255,0.1)' : '#f8f9fa',
                                padding: '2px 4px',
                                borderRadius: '4px',
                                fontSize: '0.85rem',
                                fontFamily: 'monospace'
                            },
                            '& table': {
                                width: '100%',
                                borderCollapse: 'collapse',
                                fontSize: '0.85rem'
                            },
                            '& th, & td': {
                                border: '1px solid #ddd',
                                padding: '8px',
                                textAlign: 'left'
                            },
                            '& th': {
                                backgroundColor: isUserMessage ? 'rgba(255,255,255,0.1)' : '#f8f9fa',
                                fontWeight: 'bold'
                            }
                        }}
                    >
                        <Box
                            dangerouslySetInnerHTML={{ __html: prefix }}
                            sx={{
                                fontSize: '0.95rem',
                                lineHeight: 1.5,
                                '& img': {
                                    maxWidth: '100%',
                                    height: 'auto'
                                }
                            }}
                        />
                    </Paper>

                    {/* Event Type Badge */}
                    {!isUserMessage && (
                        <Chip
                            label={`${props.etype}: ${props.estate} (${props.estimuli})`}
                            size="small"
                            sx={{
                                mt: 0.5,
                                fontSize: '0.7rem',
                                height: '20px',
                                backgroundColor: isBotMessage ? '#34495e' : '#f39c12',
                                color: 'white',
                                '& .MuiChip-label': {
                                    px: 1
                                }
                            }}
                        />
                    )}
                </Box>
            </Box>

            {/* Chart Display */}
            {isChart && (
                <Fade in={true} timeout={500}>
                    <Box sx={{ 
                        mt: 1, 
                        ml: 5,
                        mr: 1,
                        backgroundColor: 'white',
                        borderRadius: '12px',
                        p: 2,
                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                    }}>
                        <Box sx={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            gap: 1, 
                            mb: 1 
                        }}>
                            <TrendingUpIcon sx={{ color: '#27ae60', fontSize: 20 }} />
                            <Typography variant="subtitle2" sx={{ 
                                color: '#2c3e50',
                                fontWeight: 600
                            }}>
                                Price Forecast
                            </Typography>
                        </Box>
                        <Chart
                            type="line"
                            height={120}
                            width="100%"
                            options={chartData2.options}
                            series={cseries}
                        />
                    </Box>
                </Fade>
            )}
        </Box>
    );
};

