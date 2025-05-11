import { formatRelative } from 'date-fns';
import React, { useContext } from 'react';
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
    const currentUser = 'sameer';

    return (
        <div className="mx-4">
            <Grid container spacing={2} padding={2} >
            {
                // If the user who sent the message is the currentUser
                props.user === currentUser ? (
                    <>
                    <Grid item xs="3" />
                    <Grid item xs="9">
                    <ColorSubCard
                      padding={1}
                      spacing={1}
                      border={'red'}
                      background={'blue'}
                      align-items="right"
                      md={8}
                      aria-label="main mailbox folders"
                      sx={{ boxShadow: '0px 0px 0px #000', border: '3px solid', borderRadius: '30px', background: "lightblue" }}
                    >
                        <div className="flex items-end justify-end">
                            <div className="flex flex-col space-y-2 text-xs max-w-xs mx-2 order- items-end">
                                <div>
                                    <span 
                                        className="px-4 py-2 rounded-lg inline-block max-w-sm break-all float-right rounded-br-none bg-blue-600 text-white "
                                        dangerouslySetInnerHTML={{ __html: props.value}}
                                        > 
                                    </span> <br/>
                                    <Typography style={{ color: "black", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "0.9rem" }}>
                                    <small style={{color:'blue'}}>
                                        { props.user || "Guest User" } - {formatDate(new Date())}
                                    </small>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{props.esender}</Typography>
                                </div>
                            </div>
                        </div>
                    </ColorSubCard>
                    </Grid>
                    </>
                ) : (
                    <>
                    <Grid item xs="9">
                    <ColorSubCard
                      padding={1}
                      spacing={1}
                      border={'red'}
                      background={'blue'}
                      align-items="left"
                      md={8}
                      aria-label="main mailbox folders"
                      sx={{ boxShadow: '0px 0px 0px #000', border: '3px solid', borderRadius: '30px', background: "lightyellow" }}
                    >
                        <div className="flex items-end">
                            <div className="flex flex-col space-y-2 text-xs max-w-xs mx-2 order-2 items-start">
                                <div>
                                    <span 
                                        className="px-4 py-2 rounded-lg inline-block max-w-sm break-all rounded-bl-none bg-gray-800 text-gray-100"
                                        dangerouslySetInnerHTML={{ __html: props.value}}
                                    >
                                    </span> <br/>
                                    <Typography style={{ color: "black", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "0.9rem" }}>
                                    <small style={{color:'blue'}}>
                                        { props.user || "Guest User"  } - {formatDate(new Date())}
                                    </small>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{props.esender}</Typography>
                                </div>
                            </div>
                        </div>
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

