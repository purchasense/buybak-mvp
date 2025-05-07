import { formatRelative } from 'date-fns';
import React, { useContext } from 'react';
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


let chartData = {
  type: "area",
  height: '100%',
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
    fill: {
        type: "gradient",
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

let cd_data = [
    {
      name: "Predictions",
      data: [159.91353, 159.91353, 162.44794, 161.24452, 157.94542, 166.76558, 160.91374, 157.94542, 157.94542, 159.91353, 158.04843]
    },
    {
      name: "Forecastors",
      data: [164.27565059931024, 163.64161088130706, 163.51398769833057, 162.67782529379727, 161.94058563595763, 161.17340857674023, 160.40623151752283, 159.82062979168805, 159.20826673198033, 158.1375598295967, 157.74369845011444, 157.06055326456507, 156.47160842118893, 155.8826635778128, 154.96341642072696]
    },
];

export const  MobileChart = (props) => {
    // cd_data[0].data = props.predictions;
    // cd_data[1].data = props.forecastors;
    console.log( props.predictions);
    console.log( props.forecastors);

    return (
        <div className="mx-4">
            <Grid container spacing={2} padding={2} >
            {
                (
                    <>
                    <Grid item xs="9">
                    <ColorSubCard
                      padding={0}
                      spacing={0}
                      border={'red'}
                      background={'blue'}
                      align-items="left"
                      aria-label="main mailbox folders"
                      sx={{ boxShadow: '0px 0px 0px #000', border: '3px solid', borderRadius: '30px', background: "lightyellow" }}
                    >
                       <Chart
                            type="area"
                            width="100%"
                            height="100%"
                            sx={{boxShadow: '0px 0px 0px #000', border: '3px solid', borderRadius: '30px', background: "lightyellow"}}
                            options={chartData.options}
                            series={cd_data}
                        />
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

