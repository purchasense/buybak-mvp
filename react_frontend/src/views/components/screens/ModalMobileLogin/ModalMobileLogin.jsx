import React,{useContext, useState} from 'react';
// material-ui
import { Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, Grid, Typography } from '@mui/material';
import { useDispatch, useSelector } from "react-redux";
import {setModalMobileLoginName, setModalQRCodeLoadingStatus, setModalQRCodeLoadingExecutionStatus, setModalMobileLoginStatus, setModalQRCodeScan} from 'store/actions';
import './style.css';
import ColorSubCard from "ui-component/cards/ColorSubCard";



// ===============================|| UI DIALOG - SCROLLABLE ||=============================== //

const runInitUserProfile = async (username, name, email, phone, address, dispatch) => {

    // const dispatch = useDispatch();
    dispatch(setModalQRCodeLoadingStatus(true, '0'));

    
}

export const  ModalMobileLogin = () => {
    //
    const [username, setUsername] = React.useState("");

    const dispatch = useDispatch();
    const isOpen = useSelector((state) => { return state.qrcode.is_login_open});

    const handleLogin = (e) => {
        console.log( 'handleLogin ' + username);
        console.log({e});

        runInitUserProfile(username, "", "", "", "", dispatch).then(() => {
            console.log( username + ' successfully logged into SOLANA-buybak-FSOP');
        }).catch(error => console.log(error));

        dispatch(setModalMobileLoginName(username));
    };

    const handleClose = () => {
        console.log( 'handleClose');
        dispatch(setModalMobileLoginStatus(false));
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


    return (
            <Dialog
                open={isOpen}
                onClose={handleClose}
                overflow={'hidden'}
                sx={{ overflow: 'hidden' }}
            >
                <DialogTitle >
                    <Grid container >
                        <Grid item xs="3">
                        </Grid>
                        <Grid item xs="12">
                            <Typography variant="subtitle1" sx={{ fontFamily: 'Abhaya Libre ExtraBold', fontSize: "1.1rem", marginLeft: "25px", marginTop: "15px", fontWeight: 'bold', textAlign: "left", color: "black", }} >
                                {'Username'}
                            </Typography>
                        </Grid>
                        <Grid item xs="12">
                            <form >
                                      <input
                                        id="username"
                                        type="text"
                                        value={username}
                                        onChange={(e) => setUsername(e.target.value)}
                                      />
                            </form>
                        </Grid>
                    </Grid>
                </DialogTitle>
                <DialogActions sx={{ size: '1.25rem', }}>
                    <Button onClick={handleLogin} color="success">
                        Login
                    </Button>
                    <Button onClick={handleClose} color="error">
                        Cancel
                    </Button>
                </DialogActions>
            </Dialog>
    );
};
