import { formatRelative } from 'date-fns';
import { useDispatch, useSelector } from 'react-redux';
import React, { useState, useEffect, useContext } from 'react';
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
    const bgc = ((props.estimuli === "LiveMarketEvent") || (props.estimuli === "CompareMarketEvent")) ? "white" : "cornsilk";
    const nbgc = ((props.estimuli === "GetUserEvent") || (props.estimuli === "BuyOrSellEvent")) ? "#DDFFDD" : bgc;
    const fbgc = (props.estimuli === "ForecastEvent") ? "#FFC" : nbgc;

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
            const values = JSON.parse(props.msg);
            console.log('------------- WineForecast -------------')
            console.log({values})
            Object.keys(values).map((key: string) => {
                console.log( values[key]);
                setCData(values[key]);
                console.log(cData);
                isChart = true;
            });
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
    // console.log('isChart: ' + isChart);
    console.log( props.user + ' ' + currentUser);
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
                      border={'#FDD'}
                      align-items="right"
                      md={8}
                      aria-label="main mailbox folders"
                      sx={{ boxShadow: '0px 0px 0px #000', border: '2px solid', borderRadius: '15px', background: "#FDD" }}
                    >
                                    <span 
                                        className="px-2 py-2 rounded-lg inline-block max-w-sm break-all float-right rounded-br-none bg-blue-600 text-white "
                                        style={{ fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'bold', fontSize: "1.2rem", color: 'black'}}
                                        dangerouslySetInnerHTML={{ __html: prefix}}
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
                          border={fbgc}
                          background={fbgc}
                          align-items="left"
                          md={8}
                          aria-label="main mailbox folders"
                          sx={{ boxShadow: '0px 0px 0px #000', border: '1px solid', borderRadius: '15px', background: fbgc}}
                        >
                                    <span 
                                        className="px-4 py-2 rounded-lg inline-block max-w-sm break-all rounded-bl-none bg-gray-800 text-gray-100"
                                        style={{color: 'black', fontSize: '1.1rem'}}
                                        dangerouslySetInnerHTML={{ __html: prefix}}
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
            <Grid xs="12">
             {isChart === true ? (
                <>
                    <Grid item spacing={2} padding={2} xs="9" >
                   <Chart
                        type="line"
                        padding={2}
                        spacing={2}
                        height={100}
                        width={'100%'}
                        options={chartData2.options}
                        series={cseries}
                    />                
                    </Grid>
                    <Grid item xs="3" />
                </>
             ) : ('')}
             </Grid>
            </Grid>
        </div>
    )
};

