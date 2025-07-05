import PropTypes from "prop-types";
import React, { useState, useContext, useEffect } from "react";
import { useSelector } from "react-redux";
import { useDispatch } from "react-redux";
import Chart from 'react-apexcharts';
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableRow,
  Avatar,
  Chip,
  Divider,
  Card,
  CardContent,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  AccountBalance as PortfolioIcon,
  AttachMoney as MoneyIcon,
  ShowChart as ChartIcon,
} from '@mui/icons-material';

import {
  setModalQRCodeStatus,
  setBuybakResetAlertCount,
  setStockQuotes,
} from "store/actions";

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
        title: "Portfolio Value",
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
      name: "Portfolio Value",
      data: [0, 100, 175, 333, 500, 555, 689, 876, 989, 1000, 1103, 1650, 2100, 2255, 2722, 3000, 3501, 4423, 4878, 5689, 6000, 6100, 6655, 7500, 8011, 8333, 8456, 8900, 9119, 9435, 9670],
    },
  ],
};

export const MobilePortfolio = () => {
    const dispatch = useDispatch();

    const list_items = useSelector((state) => { 
        let list = [];
        state.qrcode.map_store_to_fsop.forEach((item) => {
            list.push(item);
        }); 
        return list;
    });
    
    const total_fsop = useSelector((state) => {return state.qrcode.total_fsop.toFixed(2);});
    const alertCount = useSelector((state) => {return state.qrcode.alertCount});

    let cdata = useSelector((state) => {return state.qrcode.cdata});
    chartData.series[0].data = cdata;

    const handleModalSearch = (store_id) => {
        dispatch(setModalQRCodeStatus(true, store_id));
    }

    const handleResetAlertCount = () => {
        dispatch(setBuybakResetAlertCount());
    }

    // Calculate total portfolio value
    const totalPortfolioValue = list_items.reduce((total, item) => {
        return total + (item.fsop * item.stock_price / 1000000.0);
    }, 0);

    return (
        <Box sx={{ 
            p: 2, 
            backgroundColor: '#f8f9fa',
            minHeight: '100vh'
        }}>
            {/* Header Section */}
            <Card sx={{ 
                mb: 3, 
                backgroundColor: '#2c3e50',
                color: 'white',
                borderRadius: 3
            }}>
                <CardContent sx={{ p: 3 }}>
                    <Box sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: 2,
                        mb: 2
                    }}>
                        <PortfolioIcon sx={{ fontSize: 32, color: '#3498db' }} />
                        <Box>
                            <Typography variant="h5" sx={{ 
                                fontWeight: 700,
                                color: 'white'
                            }}>
                                Portfolio Overview
                            </Typography>
                            <Typography variant="body2" sx={{ 
                                color: '#bdc3c7',
                                mt: 0.5
                            }}>
                                Track your wine investment performance
                            </Typography>
                        </Box>
                    </Box>
                    
                    <Box sx={{ 
                        display: 'flex', 
                        justifyContent: 'space-between',
                        alignItems: 'center'
                    }}>
                        <Box>
                            <Typography variant="h6" sx={{ 
                                color: '#3498db',
                                fontWeight: 600
                            }}>
                                Total Portfolio Value
                            </Typography>
                            <Typography variant="h4" sx={{ 
                                color: 'white',
                                fontWeight: 700
                            }}>
                                ${totalPortfolioValue.toFixed(2)}
                            </Typography>
                        </Box>
                        <Chip
                            icon={<ChartIcon />}
                            label={`${list_items.length} Holdings`}
                            sx={{
                                backgroundColor: '#3498db',
                                color: 'white',
                                fontWeight: 600,
                                fontSize: '0.9rem'
                            }}
                        />
                    </Box>
                </CardContent>
            </Card>

            {/* Portfolio Chart */}
            <Card sx={{ 
                mb: 3,
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
                                Portfolio Performance
                            </Typography>
                        </Box>
                        <Typography variant="body2" sx={{ 
                            color: '#7f8c8d'
                        }}>
                            Historical value over time
                        </Typography>
                    </Box>
                    <Box sx={{ p: 2 }}>
                        <Chart 
                            type="area"
                            height={200}
                            options={chartData.options}
                            series={chartData.series} 
                        />
                    </Box>
                </CardContent>
            </Card>

            {/* Holdings List */}
            <Box sx={{ 
                display: 'flex', 
                flexDirection: 'column',
                gap: 2
            }}>
                {list_items !== undefined && list_items.map((row, index) => {
                    let fsop_qty = Number((row.fsop * 1.0 / 100.0)).toFixed(2);
                    let fsop_val = Number((row.fsop * row.stock_price / 1000000.0)).toFixed(2);
                    return (
                        <Card
                            key={row.retailer.store_name}
                            onClick={() => handleModalSearch(row.retailer.store_id)}
                            sx={{
                                cursor: 'pointer',
                                transition: 'all 0.3s ease',
                                border: '1px solid #e0e0e0',
                                borderRadius: 3,
                                '&:hover': {
                                    transform: 'translateY(-2px)',
                                    boxShadow: '0 8px 25px rgba(0,0,0,0.15)',
                                    borderColor: '#3498db'
                                }
                            }}
                        >
                            <CardContent sx={{ p: 3 }}>
                                <Box sx={{ 
                                    display: 'flex', 
                                    alignItems: 'center',
                                    gap: 3
                                }}>
                                    {/* Store Logo */}
                                    <Avatar
                                        src={row.retailer.store_logo}
                                        alt={row.retailer.store_name}
                                        sx={{
                                            width: 70,
                                            height: 70,
                                            border: '3px solid #f8f9fa',
                                            boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                                        }}
                                    />
                                    
                                    {/* Store Info */}
                                    <Box sx={{ flex: 1 }}>
                                        <Typography variant="h6" sx={{ 
                                            fontWeight: 700,
                                            color: '#2c3e50',
                                            mb: 0.5
                                        }}>
                                            {row.retailer.store_name}
                                        </Typography>
                                        <Box sx={{ 
                                            display: 'flex', 
                                            alignItems: 'center',
                                            gap: 1,
                                            mb: 1
                                        }}>
                                            <MoneyIcon sx={{ 
                                                fontSize: 16, 
                                                color: '#27ae60' 
                                            }} />
                                            <Typography variant="body2" sx={{ 
                                                color: '#7f8c8d',
                                                fontWeight: 500
                                            }}>
                                                ${Number((row.stock_price / 10000.0)).toFixed(2)} per share
                                            </Typography>
                                        </Box>
                                    </Box>
                                    
                                    {/* Holdings Value */}
                                    <Box sx={{ 
                                        textAlign: 'right',
                                        minWidth: 120
                                    }}>
                                        <Typography variant="h5" sx={{ 
                                            fontWeight: 700,
                                            color: '#2c3e50',
                                            mb: 0.5
                                        }}>
                                            ${Number(fsop_val).toFixed(2)}
                                        </Typography>
                                        <Chip
                                            label={`${Number(fsop_qty).toFixed(2)} shares`}
                                            size="small"
                                            sx={{
                                                backgroundColor: '#ecf0f1',
                                                color: '#2c3e50',
                                                fontWeight: 600,
                                                fontSize: '0.8rem'
                                            }}
                                        />
                                    </Box>
                                </Box>
                                
                                {/* Divider */}
                                <Divider sx={{ my: 2 }} />
                                
                                {/* Additional Info */}
                                <Box sx={{ 
                                    display: 'flex', 
                                    justifyContent: 'space-between',
                                    alignItems: 'center'
                                }}>
                                    <Typography variant="body2" sx={{ 
                                        color: '#7f8c8d'
                                    }}>
                                        Portfolio Weight: {((Number(fsop_val) / totalPortfolioValue) * 100).toFixed(1)}%
                                    </Typography>
                                    <Box sx={{ 
                                        display: 'flex', 
                                        alignItems: 'center',
                                        gap: 0.5
                                    }}>
                                        <TrendingUpIcon sx={{ 
                                            fontSize: 16, 
                                            color: '#27ae60' 
                                        }} />
                                        <Typography variant="body2" sx={{ 
                                            color: '#27ae60',
                                            fontWeight: 600
                                        }}>
                                            Active
                                        </Typography>
                                    </Box>
                                </Box>
                            </CardContent>
                        </Card>
                    );
                })}
            </Box>

            {/* Empty State */}
            {(!list_items || list_items.length === 0) && (
                <Card sx={{ 
                    mt: 3,
                    backgroundColor: '#ecf0f1',
                    border: '2px dashed #bdc3c7'
                }}>
                    <CardContent sx={{ 
                        p: 4,
                        textAlign: 'center'
                    }}>
                        <PortfolioIcon sx={{ 
                            fontSize: 48, 
                            color: '#95a5a6',
                            mb: 2
                        }} />
                        <Typography variant="h6" sx={{ 
                            color: '#7f8c8d',
                            mb: 1
                        }}>
                            No Portfolio Holdings
                        </Typography>
                        <Typography variant="body2" sx={{ 
                            color: '#95a5a6'
                        }}>
                            Start investing in wine stores to build your portfolio
                        </Typography>
                    </CardContent>
                </Card>
            )}
        </Box>
    );
};
