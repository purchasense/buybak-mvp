import { formatRelative } from 'date-fns';
import { useDispatch, useSelector } from 'react-redux';
import React, { useEffect, useContext } from 'react';
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
} from '@mui/material';

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


export const  MobileMessage = (props) => {

    const dispatch = useDispatch();

    const currentUser = 'sameer';
    const bgc = ((props.estimuli === "LiveMarketEvent") || (props.estimuli === "CompareMarketEvent")) ? "white" : "cornsilk";
    const nbgc = ((props.estimuli === "GetUserEvent") || (props.estimuli === "BuyOrSellEvent")) ? "#FFDDDD" : bgc;
    const fbgc = (props.estimuli === "ForecastEvent") ? "#CCF" : nbgc;

    useEffect(() => {
        if ( props.outline && props.msg && props.outline === "BUY")
        {
            const values = JSON.parse(props.msg);
            console.log('------------- BUY -------------')
            console.log({values})
            dispatch(setModalQRCodeScan(values.wine, values.quantity, values.price));
        }
    }, []);

    return (
        <div className="mx-4">
            <Grid container spacing={1} padding={1} >
            {
                // If the user who sent the message is the currentUser
                props.user === currentUser ? (
                    <>
                    <Grid item xs="3" />
                    <Grid item xs="9">
                    <ColorSubCard
                      padding={1}
                      spacing={0}
                      border={'red'}
                      background={'blue'}
                      align-items="right"
                      md={8}
                      aria-label="main mailbox folders"
                      sx={{ boxShadow: '0px 0px 0px #000', border: '2px solid', borderRadius: '30px', background: "lightblue" }}
                    >
                                    <span 
                                        className="px-2 py-2 rounded-lg inline-block max-w-sm break-all float-right rounded-br-none bg-blue-600 text-white "
                                        style={{color: 'black'}}
                                        dangerouslySetInnerHTML={{ __html: props.msg}}
                                        > 
                                    </span> <br/>
                                        {/* props.user || "Guest User" } - {formatDate(new Date())*/}
                                    <Typography style={{ color: "gray", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "0.9rem" }}><small style={{color: 'blue'}}>{ props.user || "Guest User"  }</small>&nbsp;{props.etype}{': '}&nbsp;{props.estate}{'( '}{props.estimuli}&nbsp;{')'}</Typography>
                    </ColorSubCard>
                    </Grid>
                    </>
                ) : (
                    <>
                    <Grid item xs="9">
                        <ColorSubCard
                          padding={1}
                          spacing={0}
                          border={'red'}
                          background={'blue'}
                          align-items="left"
                          md={8}
                          aria-label="main mailbox folders"
                          sx={{ boxShadow: '0px 0px 0px #000', border: '1px solid', borderRadius: '30px', background: fbgc}}
                        >
                                    <span 
                                        className="px-4 py-2 rounded-lg inline-block max-w-sm break-all rounded-bl-none bg-gray-800 text-gray-100"
                                        style={{color: 'black', fontSize: '1.1rem'}}
                                        dangerouslySetInnerHTML={{ __html: props.msg}}
                                    >
                                    </span> <br/>
                                    <small style={{color: 'blue'}}>{ props.user || "Guest User"  }</small>
                                    <small style={{color: 'gray'}}>&nbsp;{props.etype}{': '}</small>
                                    <small style={{color: 'black'}}>&nbsp;{props.estate}</small>
                                    <small style={{color: 'red'}}>{'( '}{props.estimuli}&nbsp;{')'}</small>
                    </ColorSubCard>
                    </Grid>
                    <Grid item xs="3" />
                    </>
                )
            }
            </Grid>
        </div>
    )
};

