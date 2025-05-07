import React, { useState, useContext, useEffect } from "react";
import { useSelector } from "react-redux";
import { useDispatch } from "react-redux";
import classNames from 'classnames';
import queryIndex, {getPredictions, getForecastors, ResponseSources } from '../../../../apis/queryIndex';
import {MobileMessage} from './MobileMessage';
import {MobileWineCard} from './MobileWineCard';
import ColorSubCard from "ui-component/cards/ColorSubCard";
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
  getBuybakMobileMessages,
  setBuybakPredictions,
  setBuybakForecastors
} from "store/actions";


export const MobileIndexQuery = () => {
    const [isLoading, setLoading] = useState(false);
    const [responseText, setResponseText] = useState('```HTML <table> <tr> <th>Departure</th> <th>Total Time</th> <th>Airport Codes</th> <th>Price</th> </tr> <tr> <td>7:20 pm</td> <td>20h 25m</td> <td>ORD-GOX</td> <td>1917.0</td> </tr> <tr> <td>4:10 am</td> <td>21h 50m</td> <td>GOX-ORD</td> <td>1430.0</td> </tr> <tr> <td>7:20 pm</td> <td>20h 25m</td> <td>ORD-GOX</td> <td>1921.0</td> </tr> <tr> <td>4:10 am</td> <td>21h 50m</td> <td>GOX-ORD</td> <td>1919.0</td> </tr> </table> ```');
    let [predictionsText, setPredictionsText] = useState({});
    let [forecastorsText, setForecastorsText] = useState({});

    const dispatch = useDispatch();

    console.log( 'Pred: ' + predictionsText);
    console.log( 'Fore: ' + forecastorsText);

    useEffect(() => {
        console.log( 'MobileIndexQuery: fetch Pred/Forecast');
            getPredictions().then((response) => {
                console.log(response.list_items);
                 response.list_items.forEach((item, key) => {
                    console.log( 'PRED Key: ' + key);
                    console.log({item});
                    Object.keys(item).forEach(key => {
                        console.log(key, (item[key].length), item[key]);
                        let values = JSON.parse(item[key]);
                        dispatch(setBuybakPredictions(values));
                        setPredictionsText(values);
                    });
                 })
                // let parsedJson = JSON.parse(response.list_items[0]);
            });
            getForecastors().then((response) => {
                console.log(response.list_items);
                response.list_items.forEach((item, key) => {
                    console.log( 'FORE Key: ' + key);
                    Object.keys(item).forEach(key => {
                        console.log(key, (item[key].length), item[key]);
                        try {
                            let values = JSON.parse(item[key]);
                            dispatch(setBuybakForecastors(values));
                            setForecastorsText(values);
                        } catch (error) {
                            console.log( error);
                        } finally {
                            console.log( 'finally');
                        }
                    });
                })
            });
    }, []);

    const handleQuery = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key == 'Enter') {
          setLoading(true);
          console.log('Query: ' + e.target.value);
          console.log(e);
          dispatch(setBuybakMobileMessage(Date.now(), 'sameer', e.target.value));
          queryIndex(e.target.value).then((response) => {
            setLoading(false);
            setResponseText(response.text);
            dispatch(setBuybakMobileMessage(Date.now(), 'GPT', response.text));
          });
        }
    };

    const [tabsValue, setTabsValue] = React.useState(0);

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
            let list_msgs = [];
            let mapmsgs = state.qrcode.map_store_to_mobile_messages;
            mapmsgs.map((msg, index) => {
                console.log(msg);
                list_msgs = [...list_msgs, msg];
            })
            return list_msgs;
    });
    console.log(list_messages);

  return (
    <>
          <Tabs
            value = {tabsValue}
            onChange={handleChangeTab}
            aria-label="ant example"
            sx={{width:'100%'}}
          >
                <Tab
                    sx={{background: 'cornsilk', color: 'red'}}
                    label={'Italy'}
                />
                <Tab
                    sx={{background: 'cornsilk', color: 'red'}}
                    label={'France'}
                />
                <Tab
                    sx={{background: 'cornsilk', color: 'red'}}
                    label={'Argentina'}
                />
                <Tab
                    sx={{background: 'cornsilk', color: 'red'}}
                    label={'Germany'}
                />
        </Tabs>
    <Grid container >
      <Grid item align="left" >
        <TextField
            sx={{ m: 2, width: '90ch' }}
            id="standard-basic" label="Query" variant="standard" 
            onKeyDown={handleQuery}
        ></TextField>
      </Grid>

         <TableContainer sx={{ width: '100%', height: '450px' }}>
            <MobileWineCard index={tabsValue} />
            {
                list_messages.map(message => (
                    <MobileMessage key={message.id} user={message.user} value={message.msg} />
                ))
            }
        </TableContainer>

    </Grid>
    </>
  );
};
