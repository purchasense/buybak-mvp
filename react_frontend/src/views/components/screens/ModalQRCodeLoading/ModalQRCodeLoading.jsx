import React, {useContext, useState} from 'react';
// material-ui
import { Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, Grid, Typography } from '@mui/material';
import './style.css';
import ColorSubCard from "ui-component/cards/ColorSubCard";
import BuyBakLogo from "assets/images/logo.png";
import { useSelector } from 'react-redux';

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

export const  ModalQRCodeLoading = () => {
    //
    const classes = useStyles();
    const isLoadingOpen = useSelector((state) => { return state.qrcode.isLoadingOpen});
    const [executionStatus, setExecutionStatus] = useState("Starting");

    const execStatus = useSelector((state) => state.qrcode.executionStatus);


    const descriptionElementRef = React.useRef(null);
    React.useEffect(() => {
        if (isLoadingOpen) {
            const { current: descriptionElement } = descriptionElementRef;
            if (descriptionElement !== null) {
                descriptionElement.focus();
            }
        }
        setExecutionStatus(execStatus);
    }, [isLoadingOpen, execStatus]);


    return (
            <Dialog
                open={isLoadingOpen}
                overflow={'hidden'}
                sx={{ overflow: 'hidden' }}
            >
                    <DialogTitle >
                        <Grid container >
                            <Grid item xs="12">
                                <img alt='BuyBak' src={BuyBakLogo} height={55} />
                                    <Typography variant="subtitle1" sx={{ fontFamily: 'Abhaya Libre ExtraBold', fontSize: "1.2rem", fontWeight: 'bold', textAlign: "center", color: "black", }} >
                                        {'Executing...'}
                                    </Typography>
                            </Grid>
                        </Grid>
                    </DialogTitle>
                    <DialogContent
                        sx={{ overflow: 'hidden' }}
                    >
                <ColorSubCard
                    border={'#e37627'}
                    alignItems="middle"
                    md={8}
                    aria-label="main mailbox folders"
                    style={{position:'relative'}}
                >
                                <Grid item xs={12} padding={5}>
                                    <Typography variant="subtitle1" sx={{ fontFamily: 'Abhaya Libre ExtraBold', fontSize: "1.5rem", fontWeight: 'bold', textAlign: "center", color: "black", }} >
                                        {executionStatus}
                                    </Typography>
                                </Grid>

              </ColorSubCard>
                    </DialogContent>
            </Dialog>
    );
};
