import React, { useRef, useState, useContext, useEffect } from "react";
import { useSelector } from "react-redux";
import { useDispatch } from "react-redux";
import classNames from 'classnames';
import queryIndex, {queryUserInputIndex, queryStreamingIndex, getPredictions, getForecastors, ResponseSources } from '../../../../apis/queryIndex';
import {MobileMessage} from './MobileMessage';
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
  Tab
} from '@mui/material';

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
         behavior: 'smooth', // Optional: for smooth scrolling
       });
     }
};

export const MobileIndexQuery = () => {


    const tableContainerRef = useRef(null);

    const [isLoading, setLoading] = useState(false);
    const [responseText, setResponseText] = useState('```HTML <table> <tr> <th>Departure</th> <th>Total Time</th> <th>Airport Codes</th> <th>Price</th> </tr> <tr> <td>7:20 pm</td> <td>20h 25m</td> <td>ORD-GOX</td> <td>1917.0</td> </tr> <tr> <td>4:10 am</td> <td>21h 50m</td> <td>GOX-ORD</td> <td>1430.0</td> </tr> <tr> <td>7:20 pm</td> <td>20h 25m</td> <td>ORD-GOX</td> <td>1921.0</td> </tr> <tr> <td>4:10 am</td> <td>21h 50m</td> <td>GOX-ORD</td> <td>1919.0</td> </tr> </table> ```');
    let [predictionsText, setPredictionsText] = useState({});
    let [forecastorsText, setForecastorsText] = useState({});
    let [tabsValue, setTabsValue] = useState(0);
    let [listLen, setListLen] = useState(0);

    const dispatch = useDispatch();

    // TMD console.log( 'Pred: ' + predictionsText);
    // TMD console.log( 'Fore: ' + forecastorsText);

    const handleQuery = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key == 'Enter') {
          setLoading(true);
          console.log('Query: ' + e.target.value);
          console.log(e);
          const msg = JSON.stringify({"event_type": "user", "event_state": "init", "event_stimuli": "AgenticEvent", "event_content": { "outline": "", "message": e.target.value}});
          dispatch(setBuybakMobileMessage(Date.now(), 'sameer', msg));
          queryStreamingIndex(e.target.value)
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
                      break;
                    }
                    const s = String.fromCharCode.apply(null, value);
                    // TMD console.log(s);
                    dispatch(setBuybakMobileMessage(Date.now(), 'GPT', s));
                    controller.enqueue(value);
                  }
                } catch (error) {
                  controller.error(error);
                }
              }
            });
          })
            .then(stream => new Response(stream))
            .then(response => response.text())
            .then(result => {
              // Process the data
              // TMD console.log(result);
            })
          .catch(error => {
            console.error('Error during streaming:', error);
          });
        }
    };

    const handleUserInputQuery = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key == 'Enter') {
          setLoading(true);
          console.log('UserInputQuery: ' + e.target.value);
          console.log(e);
          const msg = JSON.stringify({"event_type": "submit", "event_state": "user", "event_stimuli": "SubmitEvent", "event_content": { "outline": "", "message": e.target.value}});
          dispatch(setBuybakMobileMessage(Date.now(), 'sameer', msg));
          queryUserInputIndex(e.target.value).then((response) => {
            setLoading(false);
            setResponseText(response.text);
            console.log("UserInputQuery: ", response.text);
            // dispatch(setBuybakMobileMessage(Date.now(), 'GPT', response.text));
          });
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
        // TMD console.log( 'list.length: ' + list.length);
        return list;
    });

    /*********************************************************
    const list_messages = useSelector((state) => {
            let list_msgs = [];
            let mapmsgs = state.qrcode.map_store_to_mobile_messages;
            mapmsgs.map((msg, index) => {
                console.log(msg);
                list_msgs = [...list_msgs, msg];
            })
            return list_msgs;
    });
    ********************************************************/

    const list_messages = useSelector((state) => {
        return state.qrcode.list_store_to_mobile_messages;
    });

    const refreshScroll = useSelector((state) => {return state.qrcode.refreshScroll;});
    console.log( 'RefreshScroll: ' + refreshScroll);

    useEffect(() => {
            scrollToBottom(tableContainerRef);
            console.log( 'scrollToBottom....................');
            const value = false;
            dispatch(setBuybakRefreshScroll(value));

    }, [refreshScroll]);

    /**
    useEffect(() => {
        setTimeout( function doSomething() {
            scrollToBottom(tableContainerRef);
            setTimeout(doSomething, 2000); // every 5 seconds
        }, 2000);
    }, []);
    **/

  return (
    <>
          <Tabs
            value = {tabsValue}
            onChange={handleChangeTab}
            aria-label="ant example"
            sx={{backgroundImage: `url('/images/wallpaper_5.jpg')`,width:'100%'}}
          >
                <Tab
                    sx={{color: 'red'}}
                    label={'Italy'}
                />
                <Tab
                    sx={{color: 'red'}}
                    label={'France'}
                />
                <Tab
                    sx={{color: 'red'}}
                    label={'Argentina'}
                />
                <Tab
                    sx={{backgroundImage: '/images/wallpaper_1.png', color: 'red'}}
                    label={'Germany'}
                />
        </Tabs>

            <TableContainer ref={tableContainerRef} sx={{ backgroundImage: `url('/images/wallpaper_5.jpg')`,width: '100%', height: '750px' }}>
                <MobileWineCard index={tabsValue} />
                {/*
                <MobileChart
                    predictions={predictionsText} 
                    forecastors={forecastorsText}
                />
                */}
                {
                    list_messages.map(message => {
                        // TMD console.log({message});
                        return (
                            <MobileMessage key={message.id} user={message.user} etype={message.event_type} estate={message.event_state} estimuli={message.event_stimuli} outline={message.outline} msg={message.msg} />
                    );
                    })
                }
            </TableContainer>

    <Grid container sx={{backgroundImage: `url("/images/wallpaper_5.jpg")`}} >
      <Grid item background="white" align="left" >
        <TextField
            sx={{ background: 'white', m: 2, width: '90ch' }}
            id="standard-basic" label="Start" variant="standard" 
            onKeyDown={handleQuery}
        ></TextField>
      </Grid>
      <Grid item align="left" >
        <TextField
            sx={{ background: 'white', m: 2, width: '90ch' }}
            id="standard-basic" label="User-Input" variant="standard" 
            onKeyDown={handleUserInputQuery}
        ></TextField>
      </Grid>

    </Grid>
    </>
  );
};
