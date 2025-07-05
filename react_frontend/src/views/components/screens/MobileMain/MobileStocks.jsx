import PropTypes from "prop-types";
import React, { useState, useContext, useEffect } from "react";
import { useSelector } from "react-redux";
import { useDispatch } from "react-redux";
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
  Store as StoreIcon,
  AttachMoney as MoneyIcon,
  ShoppingCart as CartIcon,
} from '@mui/icons-material';

import {
  setModalQRCodeStatus,
  setBuybakResetAlertCount,
  setStockQuotes,
} from "store/actions";

export const MobileStocks = () => {
    const dispatch = useDispatch();

    useEffect(() => {
        setTimeout( function doSomething() {
            // We got the list
            setTimeout(doSomething, 5000); // every 5 seconds
        }, 5000);
    }, []);

    const list_items = useSelector((state) => { 
        let list = [];
        state.qrcode.map_store_to_quotes.forEach((item) => {
            list.push(item);
        }); 
        return list;
    });
    
    const total_fsop = useSelector((state) => {return state.qrcode.total_fsop.toFixed(2);});
    const alertCount = useSelector((state) => {return state.qrcode.alertCount});

    const handleModalSearch = (store_id) => {
        dispatch(setModalQRCodeStatus(true, store_id));
    }

    const handleResetAlertCount = () => {
        dispatch(setBuybakResetAlertCount());
    }

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
                        <StoreIcon sx={{ fontSize: 32, color: '#3498db' }} />
                        <Box>
                            <Typography variant="h5" sx={{ 
                                fontWeight: 700,
                                color: 'white'
                            }}>
                                Wine Stores
                            </Typography>
                            <Typography variant="body2" sx={{ 
                                color: '#bdc3c7',
                                mt: 0.5
                            }}>
                                Browse available wine retailers and their current prices
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
                                ${total_fsop}
                            </Typography>
                        </Box>
                        <Chip
                            icon={<CartIcon />}
                            label={`${list_items.length} Stores`}
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

            {/* Stores List */}
            <Box sx={{ 
                display: 'flex', 
                flexDirection: 'column',
                gap: 2
            }}>
                {list_items !== undefined && list_items.map((row, index) => {
                    let fsop_val = Number((row.fsop * 10000.0 / row.stock_price)).toFixed(2);
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
                                            width: 80,
                                            height: 80,
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
                                                Current Price
                                            </Typography>
                                        </Box>
                                    </Box>
                                    
                                    {/* Price and Quantity */}
                                    <Box sx={{ 
                                        textAlign: 'right',
                                        minWidth: 120
                                    }}>
                                        <Typography variant="h5" sx={{ 
                                            fontWeight: 700,
                                            color: '#2c3e50',
                                            mb: 0.5
                                        }}>
                                            ${Number((row.stock_price / 10000.0)).toFixed(2)}
                                        </Typography>
                                        <Chip
                                            label={`Qty: ${Number(row.fsop / 100.0).toFixed(2)}`}
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
                                        FSOP Value: ${fsop_val}
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
                        <StoreIcon sx={{ 
                            fontSize: 48, 
                            color: '#95a5a6',
                            mb: 2
                        }} />
                        <Typography variant="h6" sx={{ 
                            color: '#7f8c8d',
                            mb: 1
                        }}>
                            No Wine Stores Available
                        </Typography>
                        <Typography variant="body2" sx={{ 
                            color: '#95a5a6'
                        }}>
                            Check back later for available wine retailers
                        </Typography>
                    </CardContent>
                </Card>
            )}
        </Box>
    );
};
