import PropTypes from "prop-types";
import React, { useState, useContext, useEffect } from "react";
import { useFormik } from "formik";
import * as yup from "yup";
import { useSelector } from "react-redux";
import { useDispatch } from "react-redux";
import buybak_logo from "../../../../assets/images/logo.png";

import {
  setModalQRCodeStatus,
  setBuybakResetAlertCount,
  setStockQuotes,
} from "store/actions";


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
// import MUIFormControl from "@material-ui/core/FormControl";
// import Transitions from "ui-component/extended/Transitions";

// material-ui
import { withStyles, makeStyles, useTheme} from '@mui/styles';
import {
  Card,
  CardContent,
  Grid,
  Button,
  useMediaQuery,
} from '@mui/material';
import MainCard from "ui-component/cards/MainCard";
import ColorSubCard from "ui-component/cards/ColorSubCard";
import AnimateButton from "ui-component/extended/AnimateButton";

// style constant
const useStyles = makeStyles((theme) => ({
  root: {
    width: "100%",
    height: 670,
    backgroundColor: theme.palette.background.paper,
  },
  avatarSuccess: {
    width: "16px",
    height: "16px",
    borderRadius: "5px",
    backgroundColor: theme.palette.success.light,
    color: theme.palette.success.dark,
    marginRight: "5px",
  },
  avatarError: {
    width: "16px",
    height: "16px",
    borderRadius: "5px",
    backgroundColor: theme.palette.orange.light,
    color: theme.palette.orange.dark,
    marginRight: "5px",
  },
  ScrollHeight: {
        height: '400px',
        overflowX: 'hidden',
        minHeight: '400px',
        [theme.breakpoints.down('md')]: {
            height: 'calc(100vh - 190px)',
            minHeight: 0
        }
  },
  chart: {
    position: 'absolute inherit',
    width: '100%'
  }
}));

const CustomizedBadge = withStyles((theme) => ({
  badge: {
    right: -5,
    top: 0,
    border: `2px solid ${theme.palette.background.paper}`,
    padding: "0 4px",
  },
}))(Badge);

// de3a3a: RED
// e37627: ORANGE

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
// =============================|| REVENUE CARD ||============================= //

let  cseries = [
    {
      name: "Portfolio",
      data: [0, 100, 175, 333, 500, 555, 689, 876, 989, 1000, 1103, 1650, 2100, 2255, 2722, 3000, 3501, 4423, 4878, 5689, 6000, 6100, 6655, 7500, 8011, 8333, 8456, 8900, 9119, 9435, 9670],
    },
  ];

const fetchQuotesFromSolana = async () => {
}

export const MobileStocks = () => {
    const dispatch = useDispatch();
    const theme = useTheme();

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
    // console.log( 'TotalFSOP: ' + total_fsop);

    let cdata = useSelector((state) => {return state.qrcode.cdata});

    cseries[0].data = cdata;
    // chartData.series.current.setData(cdata);

    // console.log({list_items});
    // TMD console.log( {cdata});

    const handleModalSearch  = (store_id) => {
        dispatch(setModalQRCodeStatus(true, store_id));
    }

    const handleResetAlertCount  = () => {
        dispatch(setBuybakResetAlertCount());
    }

    const alertCount = useSelector((state) => {return state.qrcode.alertCount});

  return (
    <>
    <ColorSubCard
      padding={0}
      spacing={0}
      border={'red'}
      background={'white'}
      align-items="middle"
      md={12}
      aria-label="main mailbox folders"
      style={{position:'relative'}}
    >

        <Grid container >
          <div style={{position:"absolute",right:'1px',top:'1px'}}>
          </div>

          <Grid item>
            <TableContainer >
              <Table sx={{ width: "100%" }} size="small" aria-label="LiqPortfolio">
                <TableBody>
                  {
                    list_items !== undefined && list_items.map(
                      (row, index) => {

                        let fsop_val = Number((row.fsop * 10000.0 / row.stock_price)).toFixed(2);
                        return (
                          <TableRow
                            hover
                            role="checkbox"
                            key={row.retailer.store_name}
                            onClick={(evt) => handleModalSearch(row.retailer.store_id)}
                          >
                                <TableCell>
                                    <img src={row.retailer.store_logo}
                                         alt={row.retailer.store_name}
                                         width="60px"
                                    />
                                </TableCell>
                                <TableCell>
                                    <Typography variant="subtitle1" sx={{ fontFamily: 'Abhaya Libre ExtraBold', fontSize: "1.1rem", textAlign: "left", color: "black", }} >
                                        {row.retailer.store_name}
                                    </Typography>
                                </TableCell>
                                <TableCell>
                                    <Typography variant="subtitle1" sx={{ fontFamily: 'Abhaya Libre ExtraBold', fontSize: "1.4rem", fontWeight: 'bold', textAlign: "right", color: "black", }} >
                                        {'$'}{Number((row.stock_price / 10000.0)).toFixed(2)}
                                    </Typography>
                                    <Typography variant="subtitle1" sx={{ fontFamily: 'Abhaya Libre ', fontSize: "0.9rem", textAlign: "right", color: "#444", }} >
                                        {'qty '}{Number(row.fsop / 100.0).toFixed(2)}
                                    </Typography>
                                </TableCell>
                            </TableRow>
                        )})
                    }
                </TableBody>
              </Table>
            </TableContainer>
          </Grid>
        </Grid>

    </ColorSubCard>
    </>
  );
};
