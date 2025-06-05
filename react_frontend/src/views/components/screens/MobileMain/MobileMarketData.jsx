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

export const  MobileMarketData = (props) => {

    const dispatch = useDispatch();
    let [liveMD, setLiveMD] = useState({});

    const fbgc = 'white';

    let isLiveMD = false;

    useEffect(() => {
        if ( props.estimuli && props.msg && (props.estimuli === "LiveMarketEvent"))
        {
            const values = JSON.parse(props.msg);
            console.log('------------- LiveMarketEvent -------------')

            console.log({values})
            setLiveMD(values);
            isLiveMD = true;
        }
    }, []);

    // TMD console.log('isLiveMD: ' + isLiveMD);
    let message = liveMD["wine"] + ": " + liveMD["quantity"] + " @ $" + Number(liveMD["price"] / 10000.0).toFixed(2);
    return (
        <div className="mx-4">
            <Grid container spacing={1} padding={1} >
            {
            <>
                <Grid item xs="9">
                    <ColorSubCard
                      padding={1}
                      spacing={0}
                      border={'#000'}
                      align-items="right"
                      md={8}
                      aria-label="main mailbox folders"
                      sx={{ boxShadow: '0px 0px 0px #000', border: '2px solid', borderRadius: '15px', background: "#d7e3ef" }}
                    >
                        <span 
                            className="px-4 py-2 rounded-lg inline-block max-w-sm break-all rounded-bl-none bg-gray-800 text-gray-100"
                            style={{color: 'black', fontSize: '1.1rem'}}
                            dangerouslySetInnerHTML={{ __html: message}}
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
            }
            </Grid>
        </div>
    )
};

