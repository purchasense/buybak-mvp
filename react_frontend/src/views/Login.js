import { Link as RouterLink} from 'react-router-dom';
import { useState, useEffect } from 'react';
import { useFormik } from "formik";
import * as yup from "yup";
import { useNavigate } from "react-router-dom";
import './components/screens/ModalQRCodeDialog/style.css';

// material-ui
import { styled, makeStyles, useTheme } from '@mui/styles';
import { Divider, Grid, Stack, Typography, useMediaQuery } from '@mui/material';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from "@mui/material";

import {
  FormControl,
  FormControlLabel,
  Radio,
  RadioGroup,
  Checkbox,
} from "@mui/material";

// project imports
import Wrapper from './components/Wrapper';
import CardWrapper from './components/CardWrapper';
import { Box, Button } from '@mui/material';
import AnimateButton from 'ui-component/extended/AnimateButton';
import LandingAppBar from "ui-component/extended/LandingAppBar";
import ColorSubCard from "ui-component/cards/ColorSubCard";

// assets
import Logo from 'assets/images/logo.png';

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

// de3a3a: RED
// e37627: ORANGE
// f5d7ba: LITE ORANGE

const PageOne = () => {

    return(
        <>
            <Grid item padding={3} spacing={3} align="center" xs={1}/>
            <Grid item padding={3} spacing={3} align="left" xs={5}>
              <ColorSubCard border="#fff" sx={{ boxShadow: '2px 4px 4px #000', height: '400px', border: '5px solid', borderRadius: '30px', background: "#fff" }}>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "black", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "2.0rem" }}>
                   Earn fractional stocks as <font style={{color: '#e37627'}}>Loyalty</font>, when you spend money at your favorite <font style={{color: '#e37627'}}>Retailer</font>
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
              </ColorSubCard>
            </Grid>

            <Grid item padding={3} spacing={3} align="left" xs={6}>
                <Typography style={{ color: "black", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'bold', fontSize: "2.0rem" }}>
                   Fractional Stocks Ownership Plan<font style={{color: '#e37627'}}>(FSOP)</font>
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
            </Grid>

        {/*
            <Grid item padding={3} spacing={3} xs={1.5} />

            <Grid item padding={3} spacing={3} align="center" xs={9}>
                <Typography style={{ color: "black", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.5rem" }}>
                   Fractional Stocks Ownership Plan<font style={{color: '#e37627'}}>(FSOP)</font>
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "black", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "2.6rem" }}>
                   Earn Fractional Stocks As <font style={{color: '#e37627'}}>Loyalty</font>
                </Typography>
                <Typography style={{ color: "black", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "2.6rem" }}>
                   When U Spend Money at your Favorite <font style={{color: '#e37627'}}>Retailer</font>
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
            </Grid>

            <Grid item padding={3} spacing={3} xs={1.5} />
        */}
        </>

    );
}

const PageTwo = () => {
    return(
        <>
            <Grid item padding={3} spacing={3} align="left" xs={1}/>
            <Grid item padding={3} spacing={3} align="left" xs={5}>
                <Typography style={{ color: "black", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'bold', fontSize: "2.0rem" }}>
                    The landscape of loyalty is changing…
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
            </Grid>

            <Grid item padding={3} spacing={3} align="left" xs={5}>
              <ColorSubCard border="#f5d7ba" sx={{ boxShadow: '2px 4px 4px #000', height: '400px', border: '5px solid', borderRadius: '30px', background: "#f5d7ba" }}>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "#000", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "2.0rem" }}>
                    <font style={{color: '#de3a3a'}}>Buy</font><font style={{color: '#e37627'}}>Bak</font> is the first platform that enables consumers to build wealth while they spend at their favorite retailers
                    Consumers expect brands to build meaningful relationships and provide real value
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
              </ColorSubCard>
            </Grid>
            <Grid item padding={3} spacing={3} align="left" xs={1}/>
        </>
    );
}

const PageThree = () => {

    return(
        <>
            <Grid item padding={3} xs={6}>
        <Typography style={{ textAlign: 'center', color: "#000", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.8rem" }}>
            Enhanced Rewards
        </Typography>
                <Typography >&nbsp;{' '}</Typography>
        <ul>
        <Typography style={{ color: "#000", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.4rem" }}>
        <li>Today&lsquo;s consumers seek loyalty programs that offer meaningful rewards beyond just discounts.</li>
        <li>A fractional stock ownership plan turns consumers into investors.</li>
        <li>Consumers overwhelmingly spend more at companies where they have a financial investment.</li>
        <li>Equity ownership provides meaningful customer recognition and appreciation, strengthening the customer relationship and brand affinity</li>
        </Typography>
        </ul>
            </Grid>

            <Grid item padding={3} xs={6}>
        <Typography style={{ textAlign: 'center', color: "#000", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.8rem" }}>
Competitive Differentiation 
        </Typography>
                <Typography >&nbsp;{' '}</Typography>
        <ul>
        <Typography style={{ color: "#000", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.4rem" }}>
        <li>In a crowded marketplace, brands need to stand out.</li>
        <li>The Buybak platform provides unique and valuable rewards that brands can differentiate themselves from competitors and attract and retain customers more effectively.</li>
        <li>Transforming loyalty spend from discounting to equity rewards enables continual support for share price</li>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
        </Typography>
        </ul>
            </Grid>

            <Grid item padding={3} xs={12}>
        <Typography style={{ textAlign: 'center', color: "#000", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.8rem" }}>
Long-Term Engagement
        </Typography>
                <Typography >&nbsp;{' '}</Typography>
        <Typography style={{ textAlign: 'center', color: "#000", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.4rem" }}>
        The BuyBak enables brands to continuously engage customers with compelling incentives, encouraging repeat purchases.
        </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
            </Grid>

        </>
    );
}

const PageFour = () => {

    return(
        <>
            <Grid item padding={3} spacing={3} align="left" xs={1}/>
            <Grid item padding={3} spacing={3} align="left" xs={5}>
                <Typography style={{ color: "black", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'bold', fontSize: "2.0rem" }}>
                    How it Works...
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
            </Grid>

            <Grid item padding={3} spacing={3} align="left" xs={5}>
              <ColorSubCard border="#f5d7ba" sx={{ boxShadow: '2px 4px 4px #000', height: '400px', border: '5px solid', borderRadius: '30px', background: "#f5d7ba" }}>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <ul>
                <Typography style={{ color: "#000", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.6rem" }}>
                    <li>Consumers use <font style={{color: '#de3a3a'}}>Buy</font><font style={{color: '#e37627'}}>Bak</font> app to scan receipts</li>
                    <li><font style={{color: '#de3a3a'}}>Buy</font><font style={{color: '#e37627'}}>Bak</font> (OMS) analyzes receipts, and allocates a % to customer FSOP account </li>
                    <li><font style={{color: '#de3a3a'}}>Buy</font><font style={{color: '#e37627'}}>Bak</font> manages the stock transactions and portfolio in real-time and updates for account holders</li>
                </Typography>
                </ul>
                    
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
              </ColorSubCard>
            </Grid>
            <Grid item padding={3} spacing={3} align="left" xs={1}/>
        </>
    );
}

const PageNine = () => {

    return(
        <>
            <Grid item padding={3} spacing={3} xs={1.5} />
            <Grid item padding={3} spacing={3} xs={3}>
              <ColorSubCard border="#FFF" sx={{ boxShadow: '2px 4px 4px #000', height: '300px', border: '5px solid', borderRadius: '30px', background: "white" }}>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "black", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "5.0rem" }}>
                  5+
                </Typography>
                <Typography >{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "2.0rem" }}>
                   Users Logged In
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
              </ColorSubCard>
            </Grid>

            <Grid item padding={3} spacing={3} xs={3}>
              <ColorSubCard border="#FFF" sx={{ boxShadow: '4px 4px 4px #000', height: '300px', border: '5px solid', borderRadius: '30px', background: "white" }}>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "black", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "5.0rem" }}>
                  20+
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "2.0rem" }}>
                    # of Transactions
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
              </ColorSubCard>
            </Grid>

            <Grid item padding={3} spacing={3} xs={3}>
              <ColorSubCard border="#FFF" sx={{ boxShadow: '4px 4px 4px #000', height: '300px', border: '5px solid', borderRadius: '30px', background: "white" }}>
                <Typography >&nbsp;{' '}</Typography>
                <Typography style={{ color: "black", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "5.0rem" }}>
                    100+
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >&nbsp;{' '}</Typography>
                <Typography >{' '}</Typography>
                <Typography style={{ color: "#de3a3a", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "2.0rem" }}>
                    Total TVL Locked
                </Typography>
                <Typography >&nbsp;{' '}</Typography>
              </ColorSubCard>
            </Grid>
            <Grid item padding={3} spacing={3} xs={1.5} />
        </>
    );
}

const PageFive = () => {

    return(
        <>
            <Grid item padding={3} spacing={3} xs={12} />
            <Grid item padding={3} spacing={3} xs={12} />
            <Grid item padding={3} spacing={3} xs={12} />
            <Grid item padding={3} align="center" spacing={3} xs={12} >
                <Typography style={{ color: "black", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "2.6rem" }}>
                   What Are you Waiting For?
                </Typography>
            </Grid>
            <Grid item padding={3} spacing={3} xs={4.5} />
            <Grid item padding={3} align="center" spacing={3} xs={3}>
              <ColorSubCard border="#e37627" sx={{ boxShadow: '2px 2px 2px #000', marginTop: '-20px', width: '200px', height: '50px', border: '5px solid', borderRadius: '100px', background: "#e37627" }}>
                <Typography style={{ marginTop: '-10px', color: "white", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.2rem" }}>
                Join the Waitlist
                </Typography>
              </ColorSubCard>
            </Grid>

            <Grid item padding={3} spacing={3} xs={4.5} />
            <Grid item padding={3} spacing={3} xs={12} />
            <Grid item padding={3} spacing={3} xs={12} />
            <Grid item padding={3} spacing={3} xs={12} />
        </>
    );
}

const PageSix = () => {

    return(
        <>
            <Grid item padding={3} spacing={3} xs={2.0} />
            <Grid item padding={3} spacing={3} xs={2.0} >
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
            <Grid item padding={3} spacing={3} xs={2.0} >
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
            <Grid item padding={3} spacing={3} xs={2.0} >
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
            <Grid item padding={3} spacing={3} xs={2.0} >
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
            <Grid item padding={3} spacing={3} xs={2.0} />
        </>
    );
}

// ================================|| AUTH3 - LOGIN ||================================ //

// de3a3a: RED
// e37627: ORANGE

const Login = () => {

  const classes = useStyles();
  const theme = useTheme();
  const matchDownSM = useMediaQuery(theme.breakpoints.down('md'));
  const [width, setWidth] = useState(window.innerWidth);

  const isMobile = width <= 768;

  const handleSubmit = (e) => {
  }

  return (
    <AuthWrapper1>
        <div id="home" className={classes.header}>
            <LandingAppBar />
        </div>

    {!isMobile && (
    <>
        <Grid
          container
          justifyContent="space-between"
          alignItems="center"
          sx={{
            minHeight: "620px",
            backgroundColor: "#f5d7ba",
            marginTop: "50px",
            backgroundSize: "100%",
          }}
        >
            <PageOne />
        </Grid>

        <Grid
          container
          justifyContent="space-between"
          alignItems="center"
          sx={{
            minHeight: "620px",
            backgroundColor: "white",
            backgroundSize: "100%",
          }}
        >

            <PageTwo />

        </Grid>

        <Grid
          container
          justifyContent="space-between"
          alignItems="center"
          sx={{
            minHeight: "620px",
            backgroundColor: "#f5d7ba",
            marginTop: "50px",
            backgroundSize: "100%",
          }}
        >

            <PageThree />

        </Grid>

        <Grid
          container
          justifyContent="space-between"
          alignItems="center"
          sx={{
            minHeight: "620px",
            backgroundColor: "white",
            marginTop: "50px",
            backgroundSize: "100%",
          }}
        >
            <PageFour />
        </Grid>

        <Grid
          container
          justifyContent="space-between"
          alignItems="center"
          sx={{
            minHeight: "620px",
            backgroundColor: "#f5d7ba",
            marginTop: "50px",
            backgroundSize: "100%",
          }}
        >
            <PageFive />
        </Grid>

        <Grid
          container
          justifyContent="space-between"
          alignItems="center"
          sx={{
            minHeight: "320px",
            backgroundColor: "white",
            marginTop: "50px",
            backgroundSize: "100%",
          }}
        >
            <PageSix />
        </Grid>

        <Grid
          container
          justifyContent="space-between"
          alignItems="center"
          sx={{
            backgroundColor: "#f5d7ba",
            marginTop: "20px",
            backgroundSize: "100%",
          }}
        >
                <Grid item xs={3}/>
                <Grid item xs={9}>
                    <Typography sx={{color: "#000", fontFamily: 'tiempos-headline,Lucida,Georgia,serif', fontWeight: 'normal', fontSize: "1.0rem" }}>
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

export default Login;
