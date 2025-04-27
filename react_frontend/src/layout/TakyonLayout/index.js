import { Outlet } from 'react-router-dom';

// material-ui
import { styled, useTheme } from '@mui/material/styles';
import { Box, CssBaseline } from '@mui/material';

// styles
const Main = styled('main', { shouldForwardProp: (prop) => prop !== 'open' })(({ theme }) => ({
  ...theme.typography.mainContent,
  borderBottomLeftRadius: 0,
  borderBottomRightRadius: 0,
  padding: 0,
}));

// ==============================|| MAIN LAYOUT ||============================== //

const TakyonLayout = () => {
  const theme = useTheme();

  return (
    <Box padding={0} sx={{ display: 'flex' }}>
      <CssBaseline />
      <Main sx={{ px: 0}} padding={0} theme={theme}>
        <Outlet />
      </Main>
    </Box>
  );
};

export default TakyonLayout;
