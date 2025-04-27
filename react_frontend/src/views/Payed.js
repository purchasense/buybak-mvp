import { useEffect, useState } from 'react';

// material-ui
import { Grid } from '@mui/material';
import { Main } from './components/screens/Main';
import { MobileMain } from './components/screens/MobileMain';
import { ModalQRCodeDialog } from './components/screens/ModalQRCodeDialog';
import { ModalQRCodeLoading } from './components/screens/ModalQRCodeLoading';
import { ModalMobileLogin } from './components/screens/ModalMobileLogin';

// project imports


// ==============================|| DEFAULT DASHBOARD ||============================== //

const Payed = () => {
  const [isLoading, setLoading] = useState(true);
  useEffect(() => {
    setLoading(false);
  }, []);

  return (
    <Grid container padding={0} spacing={0}>
        <Grid item sm={12} xs={12} md={12} lg={12}>
            <MobileMain />
        </Grid>
    </Grid>
  );
};

export default Payed;
