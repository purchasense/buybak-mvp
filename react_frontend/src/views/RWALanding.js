import SolanaLogo from "assets/images/SolanaLogo.png";
import { useNavigate } from "react-router-dom";
import './components/screens/ModalQRCodeDialog/style.css';
import LandingAppBar from '../ui-component/extended/LandingAppBar';
// assets
import Logo from 'assets/images/logo.png';
import PropTypes from "prop-types";
import React, { useState, useContext, useEffect } from "react";
import { useFormik } from "formik";
import * as yup from "yup";
import { useSelector } from "react-redux";
import { useDispatch } from "react-redux";
import Chart from 'react-apexcharts';

// {'Up by '}{'7%'}{' from last month'}

import {
  setModalQRCodeStatus,
  setBuybakResetAlertCount,
  setStockQuotes,
  setBuybakStatistics,
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
// import Transitions from "ui-component/extended/Transitions";

// material-ui
import { styled, withStyles, makeStyles, useTheme} from '@mui/styles';
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
// f5d7ba: LIGHT ORANGE


const AuthWrapper1 = styled('div')(({ theme }) => ({
    backgroundColor:  '#FFF',
}));


const useStyles = makeStyles({
  authPurpleCard: {
    "&:after": {
      content: '""',
      position: "absolute",
      top: "12%",
      left: "-3%",
      width: "755px",
      backgroundSize: "751px",
      height: "622px",
      // backgroundImage: `url(${mac})`,
      backgroundRepeat: "no-repeat",
      backgroundPosition: "left",
      animation: "15s wings ease-in-out infinite",
    },
  },
  blueButton: {
    backgroundColor: " #6f9eff",
    color: "white",
  },
  redButton: {
    backgroundColor: " #ff6f9e",
    color: "white",
  },
  link: {
    color: "#ec407a",
  },
  signDivider: {
    borderColor: "#22242a",
  },
  signText: {
    color: "red",
  },
});

const PageOne = () => {

    const map_stock_quotes_count = useSelector((state) => {
        let count = 0;
        state.qrcode.map_store_to_quotes.forEach((item, index) => {
            // console.log({item});
            count += item.fsop;
        });
        return count;
    });
    // TMD console.log(map_stock_quotes_count);
    const total_stats = useSelector((state) => state.qrcode.total_stats);
    const total_trans = useSelector((state) => state.qrcode.total_trans);

    return(
        <>
            <Grid item align="center" xs={12}>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "black", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.9rem" }}>
                   Worlds First <font style={{fontWeight: 'bold', color: '#de3a3a'}}>RWA</font> as Fractional Stocks Ownership Plan<font style={{color: '#e37627'}}>(FSOP)</font> on
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >
                   <img alt='solana' style={{border: '0px'}} src={SolanaLogo} height={55}/>
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
            </Grid>
            <Grid item padding={3} xs={1.5} />
            <Grid item padding={3} xs={3}>
              <ColorSubCard border="#f5d7ba" sx={{ boxShadow: '4px 4px 4px #000', height: '300px', border: '5px solid', borderRadius: '30px', background: "#f5d7ba" }}>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "#000", fontFamily: 'Trykker, sans-serif', fontWeight: 'normal', fontSize: "4.0rem" }}>
                  {'$'}{(map_stock_quotes_count/1000).toFixed(2)}K+
                </Typography>
                <Typography >{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "#000", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "2.0rem" }}>
                   TVL from Inventory
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
              </ColorSubCard>
            </Grid>

            <Grid item padding={3} xs={3}>
              <ColorSubCard border="#f5d7ba" sx={{ boxShadow: '4px 4px 4px #000', height: '300px', border: '5px solid', borderRadius: '30px', background: "#f5d7ba" }}>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "#000", fontFamily: 'Trykker, sans-serif', fontWeight: 'normal', fontSize: "4.0rem" }}>
                  {total_trans}+
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "#000", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "2.0rem" }}>
                    # of Transactions
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
              </ColorSubCard>
            </Grid>

            <Grid item padding={3} xs={3}>
              <ColorSubCard border="#f5d7ba" sx={{ boxShadow: '4px 4px 4px #000', height: '300px', border: '5px solid', borderRadius: '30px', background: "#f5d7ba" }}>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "#000", fontFamily: 'Trykker, sans-serif', fontWeight: 'normal', fontSize: "4.0rem" }}>
                    {'$'}{(total_stats/1000000000.0).toFixed(2)}K+
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >{' '}</Typography>
                <Typography style={{ color: "#000", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "2.0rem" }}>
                    TVL from Users
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
              </ColorSubCard>
            </Grid>
            <Grid item padding={3} xs={1.5} />

            <Grid item padding={3} xs={12} >
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
            </Grid>
    </>
    );
}

const PageTwo = () => {


    const list_items = useSelector((state) => {
        let list = [];
        state.qrcode.map_store_to_quotes.forEach((item) => {
            list.push(item);
        });
        return list;
    });

    // TMD console.log( {list_items});

    return(
        <>
            <Grid item xs={3} />
            <Grid item xs={6}>
            <ColorSubCard border="#f5d7ba" sx={{ boxShadow: '4px 4px 4px #000', height: '700px', border: '0px solid', borderRadius: '30px', background: "#fff" }}>
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
                                        <Typography variant="subtitle1" sx={{ fontFamily: 'Abhaya Libre ExtraBold', fontSize: "1.2rem", fontWeight: 'bold', textAlign: "right", color: "#000", }} >
                                            {Number(row.fsop / 100.0).toFixed(2)}
                                        </Typography>
                                    </TableCell>
                                    <TableCell>
                                        <Typography variant="subtitle1" sx={{ fontFamily: 'Abhaya Libre ExtraBold', fontSize: "1.4rem", fontWeight: 'bold', textAlign: "right", color: "black", }} >
                                            {'$'}{Number((row.stock_price / 10000.0)).toFixed(2)}
                                        </Typography>
                                    </TableCell>
                                </TableRow>
                            )})
                        }
                    </TableBody>
                  </Table>
                </TableContainer>
            </ColorSubCard>
            </Grid>
            <Grid item xs={3} />

        </>
    );
}

const PageThree = () => {

    return(
        <>
            <Grid item padding={3} xs={12} />
            <Grid item padding={3} xs={12} />
            <Grid item padding={3} xs={12} />
            <Grid item padding={3} align="center" xs={12} >
                <Typography style={{ color: "black", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "2.6rem" }}>
                   What Are you Waiting For?
                </Typography>
            </Grid>
            <Grid item padding={3} xs={4.5} />
            <Grid item padding={3} align="center" xs={3}>
              <ColorSubCard border="#e37627" sx={{ boxShadow: '2px 2px 2px #000', marginTop: '-20px', width: '200px', height: '50px', border: '5px solid', borderRadius: '100px', background: "#e37627" }}>
                <Typography style={{ marginTop: '-10px', color: "white", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.2rem" }}>
                Join the Waitlist
                </Typography>
              </ColorSubCard>
            </Grid>

            <Grid item padding={3} xs={4.5} />
            <Grid item padding={3} xs={12} />
            <Grid item padding={3} xs={12} />
            <Grid item padding={3} xs={12} />

        </>
    );
}

const PageFour = () => {

    return (
        <>
            <Grid item padding={3} xs={2.0} />
            <Grid item padding={3} xs={2.0} >
                <Typography style={{ color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'bold', fontSize: "1.0rem" }}>
                About
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.0rem" }}>
                Company
                </Typography>
                <Typography> &nbsp;{' '} </Typography>
                <Typography style={{ color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.0rem" }}>
                Careers
                </Typography>
                <Typography> &nbsp;{' '} </Typography>
                <Typography style={{ color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.0rem" }}>
                Media
                </Typography>
                <Typography> &nbsp;{' '} </Typography>
                <Typography style={{ color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.0rem" }}>
                Affiliates
                </Typography>
            </Grid>
            <Grid item padding={3} xs={2.0} >
                <Typography style={{ color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'bold', fontSize: "1.0rem" }}>
                Legal
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.0rem" }}>
                Terms of use
                </Typography>
                <Typography> &nbsp;{' '} </Typography>
                <Typography style={{ color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.0rem" }}>
                Privacy policy
                </Typography>
                <Typography> &nbsp;{' '} </Typography>
                <Typography style={{ color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.0rem" }}>
                Accessibility
                </Typography>
                <Typography> &nbsp;{' '} </Typography>
                <Typography> &nbsp;{' '} </Typography>
            </Grid>
            <Grid item padding={3} xs={2.0} >
                <Typography style={{ color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'bold', fontSize: "1.0rem" }}>
                Products
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.0rem" }}>
                Investing
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.0rem" }}>
                Direct deposit
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.0rem" }}>
                Lending 
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.0rem" }}>
                Borrowing
                </Typography>
            </Grid>
            <Grid item padding={3} xs={2.0} >
                <Typography style={{ color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'bold', fontSize: "1.0rem" }}>
                Social
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.0rem" }}>
                Discord
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.0rem" }}>
                Telegram
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.0rem" }}>
                X (Twitter)
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.0rem" }}>
                LinkedIn
                </Typography>
            </Grid>
            <Grid item padding={3} xs={2.0} />
    </>
    );
}
// ================================|| AUTH3 - LOGIN ||================================ //

// de3a3a: RED
// e37627: ORANGE

const fetchQuotesFromSolana = async () => {
}

const fetchStatsFromSolana = async () => {
}

export const RWALanding = () => {
    const dispatch = useDispatch();
    const theme = useTheme();
    const classes = useStyles();
    const [width, setWidth] = useState(window.innerWidth);

    const isMobile = width <= 768;

    useEffect(() => {
        setTimeout( function doSomething() {
            console.log("Hello World in RWALanding");
            setTimeout(doSomething, 5000); // every 5 seconds
        }, 5000);
    }, []);

  const handleSubmit = (e) => {
  }


  return (
    <AuthWrapper1>
        <div id="home" className={classes.header}>
            <LandingAppBar />
        </div>

    {!isMobile && (
    <>
        <Grid container justifyContent="space-between" alignItems="center"
          sx={{ minHeight: "820px", backgroundColor: "#FFF", marginTop: "20px", backgroundSize: "100%", }}
        >
            <PageOne />
        </Grid>

        <Grid container justifyContent="space-between" alignItems="center"
          sx={{ minHeight: "820px", backgroundColor: "#f5d7ba", marginTop: "20px", backgroundSize: "100%", }}
        >
            <PageTwo />
        </Grid>

        <Grid container justifyContent="space-between" alignItems="center"
          sx={{ minHeight: "320px", backgroundColor: "#fff", marginTop: "20px", backgroundSize: "100%", }}
        >
            <PageFour />
        </Grid>

        <Grid
          container
          justifyContent="space-between"
          alignItems="center"
          sx={{
            backgroundColor: "f5d7ba",
            marginTop: "20px",
            backgroundSize: "100%",
          }}
        >
                <Grid item xs={3}/>
                <Grid item xs={9}>
                    <Typography sx={{color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.0rem" }}>
                        © 2024, BuyBak Inc. All Rights Reserved.  By using this website, you accept our Terms of Use and Privacy Policy.
                    </Typography>
                </Grid>
                <Grid item xs={12}>
                    <Typography >&nbsp;{' '}</Typography>
                </Grid>
                        
          </Grid>
    </>
    )}
    </AuthWrapper1>
  );
};

export default RWALanding;
