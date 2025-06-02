import { formatRelative } from 'date-fns';
import React, { useContext } from 'react';
import { useSelector } from "react-redux";
import ColorSubCard from "ui-component/cards/ColorSubCard";
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


export const  MobileWineCard = (props) => {
    const currentUser = 'sameer';

    const wineSelection = useSelector((state) => {
        // TMD console.log(state.qrcode.map_store_to_wines);
        // TMD console.log(state.qrcode.map_store_to_wines.get(props.index));
        if ( props.index === 0)
        {
            return state.qrcode.map_store_to_wines.get("0");
        }
        else if ( props.index === 1)
        {
            return state.qrcode.map_store_to_wines.get("1");
        }
        else if ( props.index === 2)
        {
            return state.qrcode.map_store_to_wines.get("2");
        }
        else if ( props.index === 3)
        {
            return state.qrcode.map_store_to_wines.get("3");
        }
        return undefined;
    });

    // TMD console.log(props);
    return (
            <Grid container spacing={2} padding={2} >
                <Grid item xs="12">
                    <ColorSubCard
                      padding={1}
                      spacing={1}
                      border={'red'}
                      background={'white'}
                      align-items="center"
                      md={8}
                      aria-label="main mailbox folders"
                      sx={{ boxShadow: '0px 0px 0px #000', border: '3px solid', borderRadius: '30px', background: "white" }}
                    >
                    {wineSelection !== undefined && (
                        <Grid container>
                            <Grid item xs="3" align="left">
                                <img src={wineSelection.image}
                                     alt={wineSelection.name}
                                     width="120px"
                                />
                            </Grid>
                            <Grid item xs="9" align="left">
                                <Typography variant="subtitle2" sx={{ fontFamily: 'Abhaya Libre ExtraBold', fontSize: "1.4rem", textAlign: "le    ft", color: "black", }} >
                                    {wineSelection.title}
                                    {wineSelection.place}
                                    <img src={"/images/clubdvin_location_icon.png"} alt={wineSelection.name} width="25px" />
                                </Typography>
                                <Typography variant="subtitle2" sx={{ fontFamily: 'Abhaya Libre ExtraBold', fontSize: "1.1rem", textAlign: "le    ft", color: "gray", }} >
                                    {wineSelection.notes}
                                </Typography>
                            </Grid>
                            <Grid item xs="12" align="left">
                                <Typography variant="subtitle1" sx={{ fontFamily: 'Abhaya Libre ExtraBold', fontSize: "1.1rem", textAlign: "le    ft", color: "black", }} >
                                    {wineSelection.name}
                                </Typography>
                            </Grid>
                        </Grid>
                    )}
                    </ColorSubCard>
                </Grid>
            </Grid>
    )
};

