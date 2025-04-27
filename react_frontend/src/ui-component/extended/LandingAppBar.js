import React, { useState, useEffect } from 'react';
import { Link as RouterLink } from 'react-router-dom';

// material-ui
import { styled, useTheme } from '@mui/material/styles';
import {
  AppBar as MuiAppBar,
  Box,
  Button,
  Container,
  Drawer,
  Grid,
  IconButton,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Stack,
  Toolbar,
  Typography,
  useScrollTrigger,
  FormControlLabel,
  FormLabel,
  Radio,
  RadioGroup,
  SpeedDial,
  SpeedDialAction,
  Switch,
} from '@mui/material';
import { useDispatch } from 'react-redux';
import { useSelector } from 'react-redux';
import { MENU_TYPE } from 'store/actions'; // THEME_RTL

// project imports
// assets
import {
  IconBook,
  IconCreditCard,
  IconDashboard,
  IconHome2,
} from "@tabler/icons";
import BuyBakLogo from "assets/images/logo.png";

const AuthWrapper1 = styled('div')(({ theme }) => ({
    // backgroundColor: theme.palette.mode === 'dark' ? "#00070f" : theme.palette.primary.light,
    // backgroundColor:   theme.palette.primary.dark,
    backgroundColor:  '#FFF',
    minHeight: '100vh'
}));


// elevation scroll
function ElevationScroll(props) {
  const { children, window } = props;
  const theme = useTheme();
  const borderColor =
    theme.palette.mode === "dark"
      ? theme.palette.dark.dark
      : theme.palette.grey[200];

  const trigger = useScrollTrigger({
    disableHysteresis: true,
    threshold: 0,
    target: window ? window() : undefined,
  });

  return React.cloneElement(children, {
    elevation: trigger ? 2 : 0,
    style: {
      // backgroundColor: theme.palette.grey[900],
      backgroundColor: "#FFF",
      borderBottom: trigger ? "none" : "1px solid",
      borderColor: trigger ? "" : borderColor,
      color: theme.palette.text.dark,
      width: '100%',
    },
  });
}

// ===========================|| MINIMAL LAYOUT APP BAR ||=========================== //

// de3a3a: RED
// e37627: ORANGE

const LandingAppBar = ({ ...others }) => {

  const dispatch = useDispatch();
  const themeColor = "light";
  const [width, setWidth] = useState(window.innerWidth);
  const isMobile = width <= 768;

  return (
    <>
    {!isMobile && (
    <MuiAppBar>
        <Grid container alignItems="left" align="left" sx={{ background: 'white'}}>
            <Grid item xs={.5}>
              <img alt='BuyBak' src={BuyBakLogo} height={55} />
            </Grid>
            <Grid item align="left" xs={4} padding={3} spacing={3}>
                    <font style={{ fontFamily: 'Abhaya Libre ExtraBold', fontWeight: 'bold', fontSize: '2.1rem', color: '#de3a3a'}} >Buy</font>
                    <font style={{ fontFamily: 'Abhaya Libre ExtraBold', fontWeight: 'bold', fontSize: '2.1rem', color: '#e37627'}} >Bak</font>
            </Grid> 
            <Grid item align="right" xs={7.5} padding={3} spacing={3}>
                <Typography component={RouterLink} to="/login" variant="subtitle2" color="error" style={{ fontFamily: 'Abhaya Libre ExtraBold', fontWeight: 'bold', fontSize: '1.1rem', color: '#de3a3a'}} >
                    FSOP
                </Typography>
            </Grid> 
        </Grid>
    </MuiAppBar>
    )}
    </>
  );
};

export default LandingAppBar;
