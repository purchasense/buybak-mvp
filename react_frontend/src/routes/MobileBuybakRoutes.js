import { lazy } from 'react';

// project imports
import TakyonLayout from 'layout/TakyonLayout';
import Loadable from 'ui-component/Loadable';

// dashboard routing
const Payed = Loadable(lazy(() => import('views/Payed')));
const Login = Loadable(lazy(() => import('views/Login')));
const QRCode = Loadable(lazy(() => import('views/components/screens/QRCode/index.js')));

// ==============================|| MAIN ROUTING ||============================== //

const MobileBuybakRoutes = {
  path: '/',
  element: <TakyonLayout />,
  children: [
    {
      path: '/',
      element: <Payed />
    },
    {
      path: '/payed',
      element: <Login />
    }
  ]
};

export default MobileBuybakRoutes;
