import React, { useRef, useState, useContext, useEffect } from "react";
import { useSelector } from "react-redux";
import { useDispatch } from "react-redux";
import classNames from 'classnames';
import queryIndex, {queryUserInputIndex, queryStreamingIndex, getPredictions, getForecastors, ResponseSources } from '../../../../apis/queryIndex';
import {MobileMessage} from './MobileMessage';
import {MobileMarketData} from './MobileMarketData';
import {MobileChart} from './MobileChart';
import {MobileWineCard} from './MobileWineCard';
import ColorSubCard from "ui-component/cards/ColorSubCard";
import Chart from 'react-apexcharts';

import {
  Card,
  CardContent,
  Grid,
  Button,
  useMediaQuery,
} from '@mui/material';
import {
  Badge,
  Divider,
  InputAdornment,
  OutlinedInput,
  InputLabel,
  IconButton,
  Chip,
  Fab,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableSortLabel,
  TableRow,
  TextField,
  Typography,
  Tabs,
  Tab,
  Paper,
  Container,
  AppBar,
  Toolbar,
  Avatar,
  CircularProgress,
  Fade,
} from '@mui/material';
import {
  Send as SendIcon,
  PlayArrow as PlayIcon,
  WineBar as WineIcon,
  Chat as ChatIcon,
  SmartToy as BotIcon,
} from '@mui/icons-material';

import {
  setBuybakMobileMessage,
  setBuybakPredictions,
  setBuybakForecastors,
  setBuybakRefreshScroll
} from "store/actions";


let chartData = {
  type: "area",
  height: 80,
  width: '100%',
  offsetX: 0,
  options: {
    chart: {
      sparkline: {
        enabled: true,
      },
      background: "#aaa",
    },
    colors: ["#FFF"],
    dataLabels: {
      enabled: false,
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

let  cd_data = [
    {
      name: "Predictions",
      data: [159.91353, 159.91353, 162.44794, 161.24452, 157.94542, 166.76558, 160.91374, 157.94542, 157.94542, 159.91353, 158.04843]
    },
    {
      name: "Forecastors",
      data: [164.27565059931024, 163.64161088130706, 163.51398769833057, 162.67782529379727, 161.94058563595763, 161.17340857674023, 160.40623151752283, 159.82062979168805, 159.20826673198033, 158.1375598295967, 157.74369845011444, 157.06055326456507, 156.47160842118893, 155.8826635778128, 154.96341642072696]
    },
];

const scrollToBottom = (tableContainerRef) => {
     if (tableContainerRef.current) {
       tableContainerRef.current.scrollTo({
         top: tableContainerRef.current.scrollHeight,
         behavior: 'smooth',
       });
     }
};

const wineLabels = [
    "Italian",
    "French",
    "Argentinian",
    "German",
];

const wineIcons = [
    "🇮🇹",
    "🇫🇷", 
    "🇦🇷",
    "🇩🇪",
];

export const MobileIndexQuery = () => {
    const tableContainerRef = useRef(null);
    const inputRef = useRef(null);

    const [isStarted, setStarted] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [userInput, setUserInput] = useState('');
    const [responseText, setResponseText] = useState('```HTML <table> <tr> <th>Departure</th> <th>Total Time</th> <th>Airport Codes</th> <th>Price</th> </tr> <tr> <td>7:20 pm</td> <td>20h 25m</td> <td>ORD-GOX</td> <td>1917.0</td> </tr> <tr> <td>4:10 am</td> <td>21h 50m</td> <td>GOX-ORD</td> <td>1430.0</td> </tr> <tr> <td>7:20 pm</td> <td>20h 25m</td> <td>ORD-GOX</td> <td>1921.0</td> </tr> <tr> <td>4:10 am</td> <td>21h 50m</td> <td>GOX-ORD</td> <td>1919.0</td> </tr> </table> ```');
    let [predictionsText, setPredictionsText] = useState({});
    let [forecastorsText, setForecastorsText] = useState({});
    let [tabsValue, setTabsValue] = useState(0);
    let [listLen, setListLen] = useState(0);

    const dispatch = useDispatch();

    const handleQuery = () => {
          setStarted(true);
          setIsLoading(true);
          const msg = JSON.stringify({"event_type": wineLabels[tabsValue], "event_state": "init", "event_stimuli": "AgenticEvent", "event_content": { "outline": "", "message": "Start"}});
          dispatch(setBuybakMobileMessage(Date.now(), 'sameer', msg));
          let query = '{"region": "' + wineLabels[tabsValue] + '"}';
          queryStreamingIndex(query)
          .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.body.getReader();
          })
          .then(reader => {
            return new ReadableStream({
              async start(controller) {
                try {
                  while (true) {
                    const { done, value } = await reader.read();
                    if (done) {
                      controller.close();
                      setIsLoading(false);
                      break;
                    }
                    const s = String.fromCharCode.apply(null, value);
                    dispatch(setBuybakMobileMessage(Date.now(), 'GPT', s));
                    controller.enqueue(value);
                  }
                } catch (error) {
                  controller.error(error);
                  setIsLoading(false);
                }
              }
            });
          })
            .then(stream => new Response(stream))
            .then(response => response.text())
            .then(result => {
              setIsLoading(false);
            })
          .catch(error => {
            console.error('Error during streaming:', error);
            setIsLoading(false);
          });
    };

    const handleUserInputQuery = (e) => {
        if (e.key === 'Enter' && userInput.trim()) {
          console.log('UserInputQuery: ' + userInput);
          let query = JSON.stringify({"region": wineLabels[tabsValue], "user_input": userInput});
          const msg = JSON.stringify({"event_type": wineLabels[tabsValue], "event_state": "user", "event_stimuli": "SubmitEvent", "event_content": { "outline": "", "message": query}});
          dispatch(setBuybakMobileMessage(Date.now(), 'sameer', msg));
          queryUserInputIndex(query).then((response) => {
            setResponseText(response.text);
            console.log("UserInputQuery: ", response.text);
          });
          setUserInput('');
        }
    };

    const handleSendMessage = () => {
        if (userInput.trim()) {
            handleUserInputQuery({ key: 'Enter' });
        }
    };

    const handleChangeTab = (event, newValue) => {
        setTabsValue(newValue);
    };

    const list_tab_items = useSelector((state) => {
        let list = [];
        state.qrcode.map_store_to_wines.forEach((item) => {
            list.push(item);
        });
        return list;
    });

    const list_messages = useSelector((state) => {
        let region = wineLabels[tabsValue];
        let list = [];
        state.qrcode.list_store_to_mobile_messages.map((message) => {
            if (message.event_type === region)
            {
                list = [...list, message];
            }
        });
        return list;
    });

    const refreshScroll = useSelector((state) => {return state.qrcode.refreshScroll;});

    useEffect(() => {
            scrollToBottom(tableContainerRef);
            const value = false;
            dispatch(setBuybakRefreshScroll(value));
    }, [refreshScroll]);

    useEffect(() => {
        if (inputRef.current) {
            inputRef.current.focus();
        }
    }, []);

  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100vh',
      backgroundColor: '#f5f5f5'
    }}>
      {/* Header */}
      <AppBar position="static" elevation={0} sx={{ 
        backgroundColor: '#2c3e50',
        borderBottom: '1px solid #34495e'
      }}>
        <Toolbar sx={{ minHeight: '60px !important' }}>
          <BotIcon sx={{ mr: 2, color: '#3498db' }} />
          <Typography variant="h6" component="div" sx={{ 
            flexGrow: 1, 
            color: '#ecf0f1',
            fontWeight: 600,
            fontSize: '1.1rem'
          }}>
            {wineIcons[tabsValue]} {wineLabels[tabsValue]} Wine Assistant
          </Typography>
          <Avatar sx={{ 
            bgcolor: '#3498db',
            width: 32,
            height: 32
          }}>
            <WineIcon sx={{ fontSize: 18 }} />
          </Avatar>
        </Toolbar>
      </AppBar>

      {/* Region Tabs */}
      <Paper elevation={1} sx={{ 
        backgroundColor: '#34495e',
        borderRadius: 0
      }}>
        <Tabs
          value={tabsValue}
          onChange={handleChangeTab}
          variant="scrollable"
          scrollButtons="auto"
          sx={{
            '& .MuiTab-root': {
              color: '#bdc3c7',
              fontWeight: 500,
              fontSize: '0.9rem',
              minHeight: 48,
              '&.Mui-selected': {
                color: '#3498db',
                backgroundColor: '#2c3e50'
              }
            },
            '& .MuiTabs-indicator': {
              backgroundColor: '#3498db',
              height: 3
            }
          }}
        >
          {wineLabels.map((label, index) => (
            <Tab
              key={index}
              label={`${wineIcons[index]} ${label}`}
              sx={{ minWidth: 'auto', px: 2 }}
            />
          ))}
        </Tabs>
      </Paper>

      {/* Start Button */}
      <Box sx={{ 
        p: 2, 
        backgroundColor: '#ecf0f1',
        borderBottom: '1px solid #bdc3c7'
      }}>
        <Button
          variant="contained"
          fullWidth
          onClick={handleQuery}
          disabled={isLoading}
          startIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : <PlayIcon />}
          sx={{
            backgroundColor: '#3498db',
            color: 'white',
            fontWeight: 600,
            py: 1.5,
            borderRadius: 2,
            textTransform: 'none',
            fontSize: '1rem',
            '&:hover': {
              backgroundColor: '#2980b9'
            },
            '&:disabled': {
              backgroundColor: '#95a5a6'
            }
          }}
        >
          {isLoading ? 'Starting Agent...' : `Start ${wineLabels[tabsValue]} Wine Agent`}
        </Button>
      </Box>

      {/* Chat Messages Container */}
      <Box
        ref={tableContainerRef}
        sx={{
          flex: 1,
          overflowY: 'auto',
          backgroundColor: '#f8f9fa',
          p: 1,
          '&::-webkit-scrollbar': {
            width: '6px'
          },
          '&::-webkit-scrollbar-track': {
            backgroundColor: '#f1f1f1'
          },
          '&::-webkit-scrollbar-thumb': {
            backgroundColor: '#c1c1c1',
            borderRadius: '3px'
          }
        }}
      >
        <Container maxWidth="md" sx={{ py: 1 }}>
          <MobileWineCard index={tabsValue} />
          
          {list_messages.map(message => {
            if (message.event_stimuli !== "LiveMarketEvent") {
              return (
                <MobileMessage 
                  key={message.id} 
                  user={message.user} 
                  etype={message.event_type} 
                  estate={message.event_state} 
                  estimuli={message.event_stimuli} 
                  outline={message.outline} 
                  msg={message.msg} 
                />
              );
            } else {
              return (
                <MobileMarketData 
                  key={message.id} 
                  user={message.user} 
                  etype={message.event_type} 
                  estate={message.event_state} 
                  estimuli={message.event_stimuli} 
                  outline={message.outline} 
                  msg={message.msg} 
                />
              );
            }
          })}

          {/* Welcome Message */}
          {list_messages.length === 0 && (
            <Box sx={{ 
              textAlign: 'center', 
              py: 4,
              color: '#7f8c8d'
            }}>
              <ChatIcon sx={{ fontSize: 48, mb: 2, color: '#bdc3c7' }} />
              <Typography variant="h6" sx={{ mb: 1, fontWeight: 500 }}>
                Welcome to {wineLabels[tabsValue]} Wine Assistant
              </Typography>
              <Typography variant="body2">
                Click "Start {wineLabels[tabsValue]} Wine Agent" to begin your wine consultation
              </Typography>
            </Box>
          )}
        </Container>
      </Box>

      {/* Input Area */}
      <Paper elevation={3} sx={{ 
        backgroundColor: 'white',
        borderTop: '1px solid #e0e0e0',
        p: 2
      }}>
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center',
          gap: 1
        }}>
          <TextField
            ref={inputRef}
            fullWidth
            variant="outlined"
            placeholder="Ask about wines, prices, recommendations..."
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            onKeyDown={handleUserInputQuery}
            disabled={!isStarted}
            sx={{
              '& .MuiOutlinedInput-root': {
                borderRadius: 3,
                backgroundColor: '#f8f9fa',
                '& fieldset': {
                  borderColor: '#e0e0e0'
                },
                '&:hover fieldset': {
                  borderColor: '#3498db'
                },
                '&.Mui-focused fieldset': {
                  borderColor: '#3498db'
                }
              },
              '& .MuiInputBase-input': {
                fontSize: '0.95rem'
              }
            }}
          />
          <IconButton
            onClick={handleSendMessage}
            disabled={!userInput.trim() || !isStarted}
            sx={{
              backgroundColor: '#3498db',
              color: 'white',
              width: 48,
              height: 48,
              '&:hover': {
                backgroundColor: '#2980b9'
              },
              '&:disabled': {
                backgroundColor: '#bdc3c7'
              }
            }}
          >
            <SendIcon />
          </IconButton>
        </Box>
      </Paper>
    </Box>
  );
};
