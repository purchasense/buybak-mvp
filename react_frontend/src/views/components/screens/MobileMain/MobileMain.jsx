import PropTypes from "prop-types";
import React, { useState, useContext, useEffect } from "react";
import { useFormik } from "formik";
import * as yup from "yup";
import { useSelector } from "react-redux";
import { useDispatch } from "react-redux";
import Chart from 'react-apexcharts';
import buybak_logo from "../../../../assets/images/logo.png";
import {MobileStocks} from './MobileStocks';
import {MobilePortfolio} from './MobilePortfolio';
import {MobileIndexQuery} from './MobileIndexQuery';

// {'Up by '}{'7%'}{' from last month'}

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
  Tab,
  Tabs,
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

// =============================|| REVENUE CARD ||============================= //

export const MobileMain = () => {
    const dispatch = useDispatch();
    const theme = useTheme();


    const [tabsValue, setTabsValue] = React.useState(0);
    const handleChangeTab = (event, newValue) => {
        setTabsValue(newValue);
    };

    const handleResetAlertCount  = () => {
        dispatch(setBuybakResetAlertCount());
    }

    const total_fsop = useSelector((state) => {return state.qrcode.total_fsop.toFixed(2);});
    const total_curr = useSelector((state) => {return state.qrcode.total_curr.toFixed(2);});
    const pnl = (total_curr - total_fsop);
    const alertCount = useSelector((state) => {return state.qrcode.alertCount});

  return (
    <>

    <ColorSubCard
      spacing="0"
      padding="0"
      border={'#888800'}
      background={'#888800'}
      align-items="middle"
      md={8}
      style={{height: '100px',position:'relative'}}
      sx={{ border: '0 0 0 0 solid'}}
    >

                <Grid container >
                    <Grid item xs="8.5" align="right">
                    <svg xmlns="http://www.w3.org/2000/svg" className="icon icon-tabler icon-tabler-settings" width="24" height="24" viewBox="0 0 24 24" strokeWidth="1.5" stroke="#fff" fill="none" strokeLinecap="round" strokeLinejoin="round">
  <path stroke="none" d="M0 0h24v24H0z" fill="none"/>
  <path d="M10.325 4.317c.426 -1.756 2.924 -1.756 3.35 0a1.724 1.724 0 0 0 2.573 1.066c1.543 -.94 3.31 .826 2.37 2.37a1.724 1.724 0 0 0 1.065 2.572c1.756 .426 1.756 2.924 0 3.35a1.724 1.724 0 0 0 -1.066 2.573c.94 1.543 -.826 3.31 -2.37 2.37a1.724 1.724 0 0 0 -2.572 1.065c-.426 1.756 -2.924 1.756 -3.35 0a1.724 1.724 0 0 0 -2.573 -1.066c-1.543 .94 -3.31 -.826 -2.37 -2.37a1.724 1.724 0 0 0 -1.065 -2.572c-1.756 -.426 -1.756 -2.924 0 -3.35a1.724 1.724 0 0 0 1.066 -2.573c-.94 -1.543 .826 -3.31 2.37 -2.37c1 .608 2.296 .07 2.572 -1.065z" />
  <path d="M9 12a3 3 0 1 0 6 0a3 3 0 0 0 -6 0" />
</svg>
                    </Grid>
                    <Grid item xs="1.5" sx={{marginLeft: "0px"}} align="right">
                    <svg xmlns="http://www.w3.org/2000/svg" className="icon icon-tabler icon-tabler-user-circle" width="24" height="24" viewBox="0 0 24 24" strokeWidth="1.5" stroke="#fff" fill="none" strokeLinecap="round" strokeLinejoin="round">
  <path stroke="none" d="M0 0h24v24H0z" fill="none"/>
  <path d="M12 12m-9 0a9 9 0 1 0 18 0a9 9 0 1 0 -18 0" />
  <path d="M12 10m-3 0a3 3 0 1 0 6 0a3 3 0 1 0 -6 0" />
  <path d="M6.168 18.849a4 4 0 0 1 3.832 -2.849h4a4 4 0 0 1 3.834 2.855" />
</svg>
                    </Grid>
                    <Grid item xs="1.5" sx={{marginTop: "-8px", marginLeft: "0px" }} align="right">
                        <IconButton onClick={handleResetAlertCount} aria-label="cart">
                            <CustomizedBadge badgeContent={alertCount} color="error">
                            <svg xmlns="http://www.w3.org/2000/svg" className="icon icon-tabler icon-tabler-bell" width="24" height="24" viewBox="0 0 24 24" strokeWidth="1.5" stroke="#fff" fill="none" strokeLinecap="round" strokeLinejoin="round">
  <path stroke="none" d="M0 0h24v24H0z" fill="none"/>
  <path d="M10 5a2 2 0 1 1 4 0a7 7 0 0 1 4 6v3a4 4 0 0 0 2 3h-16a4 4 0 0 0 2 -3v-3a7 7 0 0 1 4 -6" />
  <path d="M9 17v1a3 3 0 0 0 6 0v-1" />
</svg>
                            </CustomizedBadge>
                        </IconButton>
                    </Grid>
                </Grid>
        <Grid container>
            <Grid item xs="2">
                <img align="top" style={{position:"relative",right:'1px',top:'-30px'}} width='80px' alt={'BuyBak.xyz'}
                    src={buybak_logo}
                />
            </Grid>
            <Grid item xs="2">
                <Typography variant="subtitle1" sx={{ position:'relative', top:'-30px', fontFamily: 'Abhaya Libre ExtraBold', fontSize: "2.3rem", fontWeight: 'bold', textAlign: "center", color: "white", }} >
                    {'BuyBak'}
                </Typography>
            </Grid>
            <Grid item align="right" xs="3" />
            <Grid item align="right" xs="5" >
                <Tabs
                    value = {tabsValue}
                    onChange={handleChangeTab}
                    aria-label="ant example"
                    sx={{width:'400px'}}
                  >
                        <Tab
                            sx={{color: 'white', fontWeight: 'bold', background: '#888800'}}
                            label={'Wines'}
                        />
                        <Tab
                            sx={{color: 'white', fontWeight: 'bold', background: '#888800'}}
                            label={'Portfolio'}
                        />
                        <Tab
                            sx={{color: 'white', fontWeight: 'bold', background: '#888800'}}
                            label={'Query'}
                        />
                </Tabs>
            </Grid>
        </Grid>
    </ColorSubCard>

        {tabsValue === 0 && <MobileStocks />}
        {tabsValue === 1 && <MobilePortfolio />}
        {tabsValue === 2 && <MobileIndexQuery />}
    </>
  );
};
