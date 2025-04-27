import React, {useContext, useState} from 'react';
// material-ui
import { Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, Grid, Typography } from '@mui/material';
import { useDispatch, useSelector } from "react-redux";
import {setModalQRCodeStatus, setModalQRCodeLoadingExecutionStatus, setModalQRCodeLoadingStatus, setModalQRCodeScan, setModalQRCodeSell, CustomerRetailFSOP} from 'store/actions';
import './style.css';
import ColorSubCard from "ui-component/cards/ColorSubCard";

import { Tab, Tabs } from '@mui/material';
import { makeStyles, useTheme } from "@mui/styles";

const useStyles = makeStyles((theme) => ({
  revenueCard: {
    position: "relative",
    color: "#fff",
  },
  revenueIcon: {
    position: "absolute",
    right: "13px",
    top: "14px",
    color: "#fff",
    "&> svg": {
      width: "100px",
      height: "100px",
      opacity: "0.5",
    },
    [theme.breakpoints.down("xs")]: {
      top: "13px",
      "&> svg": {
        width: "80px",
        height: "80px",
      },
    },
  },
}));

// ===============================|| UI DIALOG - SCROLLABLE ||=============================== //

const runBuyFSOPOnSolana = async (username, fsop, price, stock, dispatch) => {

}

const runSellFSOPOnSolana = async (username, fsop, price, stock, dispatch) => {

}

export const  ModalQRCodeDialog = () => {
    //
    const classes = useStyles();
    const dispatch = useDispatch();
    const isOpen = useSelector((state) => { return state.qrcode.isOpen});
    const storeId = useSelector((state) => { return state.qrcode.last_store_id});
    const login_username = useSelector((state) => { return state.qrcode.login_username});

    console.log( 'username: ' + login_username + ', storeId: ' + storeId);

    const [fsopInput, setFsopInput] = useState(0);
    const [fsopFraction, setFsopFraction] = useState(0);

    const [fsopQty, setFsopQty] = useState(0);
    const [fsopVal, setFsopVal] = useState(0);

    let  lfsop: CustomerRetailFSOP = useSelector((state) => {return state.qrcode.map_store_to_fsop.get(storeId);});
    // console.log({lfsop});
    let  stock_quotes: CustomerRetailFSOP = useSelector((state) => {return state.qrcode.map_store_to_quotes.get(storeId);});
    // console.log({stock_quotes});

    let fQty = 0;
    let fVal = 0;

    if (lfsop !== undefined)
    {
        fQty = Number((lfsop.fsop * 1.0 / 100.0)).toFixed(2);
        fVal = Number((lfsop.fsop * lfsop.stock_price / 1000000.0)).toFixed(2);

        // USER's FSOP Qty/Val
        // setFsopQty(fQty);
        // setFsopVal(fVal);
    }

    const handleScan = (stock_price, quantity, side) => {
        dispatch(setModalQRCodeLoadingStatus(true, '0'));
        if ( side === 'BUY')
        {
            dispatch(setModalQRCodeLoadingExecutionStatus("Buying..."));
            runBuyFSOPOnSolana(login_username, quantity, stock_price, storeId, dispatch).then(() => {
                dispatch(setModalQRCodeScan(storeId, quantity, stock_price));
            })
            .catch(error => console.log(error))
            .finally(() => console.log('runBuyFSOPOnSolana not executed'));
        }
        else if ( side === 'SELL')
        {
            dispatch(setModalQRCodeLoadingExecutionStatus("Selling..."));
            runSellFSOPOnSolana(login_username, quantity, stock_price, storeId, dispatch).then(() => {
                dispatch(setModalQRCodeSell(storeId, quantity, stock_price));
            })
            .catch(error => console.log(error))
            .finally(() => console.log('runSellFSOPOnSolana not executed'));
        }

        dispatch(setModalQRCodeStatus(false, '0'));
    };
    
    const handleBuy = async (e) => {
        e.preventDefault();
        // We have Dollar value, lets convert this to fractions.
        if ( fsopInput > 0.0)
        {
            let fsopQty = Math.ceil((fsopInput * 1000000 / stock_quotes.stock_price ));
            console.log( 'handleSumbit: fsop = ' + fsopQty);
            handleScan( stock_quotes.stock_price, fsopQty, 'BUY');
        }
    }

    const handleSell = async (e) => {
        e.preventDefault();
        // We have Dollar value, lets convert this to fractions.
        if ( fsopInput > 0.0)
        {
            let fsopQty = Math.ceil((fsopInput * 1000000 / stock_quotes.stock_price ));
            console.log( 'handleSumbit: fsop = ' + fsopQty);
            handleScan( stock_quotes.stock_price, fsopQty, 'SELL');
        }
    }

    const handleClose = () => {
        console.log( 'handleClose');
        dispatch(setModalQRCodeStatus(false, '0'));
    };
    

    const descriptionElementRef = React.useRef(null);
    React.useEffect(() => {
        if (isOpen) {
            const { current: descriptionElement } = descriptionElementRef;
            if (descriptionElement !== null) {
                descriptionElement.focus();
            }
        }
    }, [isOpen]);


    const [tabValue, setTabValue] = React.useState(0);

    const handleTabChange = (event, newValue) => {
        setTabValue(newValue);
    };
    
    const Buy = () => {
        return <font style={{ color: 'green' }}>Buy</font>;
    };

    const Sell = () => {
        return <font style={{ color: 'red' }}>Sell</font>;
    };

    const card_color = (tabValue === 0) ? 'green' : 'red';

    return (
            <Dialog
                open={isOpen}
                onClose={handleClose}
                overflow={'hidden'}
                sx={{ overflow: 'hidden' }}
            >
                    {stock_quotes !== undefined && (
                    <>
                    <DialogTitle >
                        <Grid container >
                            <Grid item xs="3">
                                <img src={stock_quotes.retailer.store_logo}
                                    alt={stock_quotes.retailer.store_name}
                                    width="60px"
                                    align="left"
                                />
                            </Grid>
                            <Grid item xs="9">
                                <Typography variant="subtitle1" sx={{ fontFamily: 'Abhaya Libre ExtraBold', fontSize: "1.3rem", marginLeft: "25px", marginTop: "10px", fontWeight: 'bold', textAlign: "left", color: "black", }} >
                                    {stock_quotes.retailer.store_name}
                                </Typography>
                            </Grid>
                        </Grid>
                    </DialogTitle>
                    <DialogContent
                        sx={{ overflow: 'hidden' }}
                    >
                <ColorSubCard
                    border={card_color}
                    alignItems="middle"
                    md={8}
                    aria-label="main mailbox folders"
                    style={{position:'relative'}}
                >
                                <Grid container alignItems="center" flexDirection="column">
                                    <Grid item>
                                    <Tabs
                                      value={tabValue}
                                      onChange={handleTabChange}
                                      aria-label="ant example"
                                      classes={classes}
                                    >
                                      <Tab label={<font style={{ color: 'green' }}>Buy</font>} />
                                      <Tab label={<font style={{ color: 'red' }}>Sell</font>} />
                                    </Tabs>
                                  </Grid>
                                  <Grid item container spacing={2} justifyContent="center"></Grid>
                                  <Grid item xs={12} textAlign="center" alignItems="center">
                        <Grid container spacing={0} padding={0} >
                            {tabValue === 0 && (
                            <>
                                <Grid item xs={5} spacing={1} padding={1}>
                                    <Typography variant="subtitle1" sx={{ fontFamily: 'Abhaya Libre ExtraBold', fontSize: "1.8rem", fontWeight: 'normal', textAlign: "center", color: "#888", }} >
                                        {'Stock'}
                                    </Typography>
                                </Grid>
                                <Grid item xs={7} spacing={1} padding={1}>
                                    <Typography variant="subtitle1" sx={{ fontFamily: 'Abhaya Libre ExtraBold', fontSize: "1.8rem", fontWeight: 'bold', textAlign: "center", color: "black", }} >
                                        {'$'}{Number((stock_quotes.stock_price / 10000.0)).toFixed(2)}
                                    </Typography>
                                </Grid>
                                <Grid item xs={5} spacing={1} padding={1}>
                                    <Typography variant="subtitle1" sx={{ fontFamily: 'Abhaya Libre ExtraBold', fontSize: "1.8rem", fontWeight: 'normal', textAlign: "center", color: "#888", }} >
                                        {'Fraction'}
                                    </Typography>
                                </Grid>
                                <Grid item xs={7} spacing={1} padding={1}>
                                    <Typography variant="subtitle1" sx={{ fontFamily: 'Abhaya Libre ExtraBold', fontSize: "1.8rem", fontWeight: 'bold', textAlign: "center", color: "black", }} >
                                        {Number(fsopFraction/100.0).toFixed(2)}
                                    </Typography>
                                </Grid>
                            </>
                            )}
                            {tabValue === 1 && (
                            <>
                                <Grid item xs={5} spacing={1} padding={1}>
                                    <Typography variant="subtitle1" sx={{ fontFamily: 'Abhaya Libre ExtraBold', fontSize: "1.8rem", fontWeight: 'normal', textAlign: "center", color: "#888", }} >
                                        {'Invested'}
                                    </Typography>
                                </Grid>
                                <Grid item xs={7} spacing={1} padding={1}>
                                    <Typography variant="subtitle1" sx={{ fontFamily: 'Abhaya Libre ExtraBold', fontSize: "1.8rem", fontWeight: 'bold', textAlign: "center", color: "black", }} >
                                        {'$'}{Number(fVal).toFixed(2)}
                                    </Typography>
                                </Grid>
                                <Grid item xs={5} spacing={1} padding={1}>
                                    <Typography variant="subtitle1" sx={{ fontFamily: 'Abhaya Libre ExtraBold', fontSize: "1.8rem", fontWeight: 'normal', textAlign: "center", color: "#888", }} >
                                        {'FSOP'}
                                    </Typography>
                                </Grid>
                                <Grid item xs={7} spacing={1} padding={1}>
                                    <Typography variant="subtitle1" sx={{ fontFamily: 'Abhaya Libre ExtraBold', fontSize: "1.8rem", fontWeight: 'bold', textAlign: "center", color: "black", }} >
                                        {Number(fQty).toFixed(2)}
                                    </Typography>
                                </Grid>
                            </>
                            )}
                        </Grid>
                        <Grid container alignItems="left" >
                        </Grid>
                            <form >
                                <Grid item xs="12">
                                      <font style={{ fontFamily: 'Abhaya Libre ExtraBold', fontSize: "1.5rem", fontWeight: 'bold', textAlign: "center", color: "black"}} >$</font>
                                      <input
                                        id="fsop"
                                        type="text"
                                        value={fsopInput}
                                        onFocus={(e) => {
                                            setFsopInput(e.target.value);
                                            setFsopFraction( Math.ceil(e.target.value * 1000000 / stock_quotes.stock_price));
                                        }}
                                        onChange={(e) => {
                                            setFsopInput(e.target.value);
                                            setFsopFraction( Math.ceil(e.target.value * 1000000 / stock_quotes.stock_price));
                                        }}
                                      />
                                </Grid>
                            </form>
                                  </Grid>
                                  <Grid item xs={12} padding={2} spacing={2} textAlign="center" alignItems="center">
                                        {tabValue === 0 && (
                                          <Button
                                            variant="contained"
                                            align="center"
                                            size="small"
                                            label="BUY"
                                            color="success"
                                            onClick={handleBuy}
                                            width="80%"
                                          >
                                            BUY
                                          </Button>
                                        )}

                                        {tabValue === 1 && (
                                          <Button
                                            type="submit"
                                            variant="contained"
                                            label="SELL"
                                            size="small"
                                            color="error"
                                            onClick={handleSell}
                                            width="80%"
                                          >
                                            SELL
                                          </Button>
                                        )}
                                  </Grid>
                                </Grid>

              </ColorSubCard>
                    </DialogContent>
                    <DialogActions sx={{ pr: 2.5, pt: 2.5 }}>
                        <Button onClick={handleClose} color="error"> Cancel </Button>
                    </DialogActions>
                  </>
                )}
            </Dialog>
    );
};
