import { lazy } from 'react';

// project imports
import TakyonLayout from 'layout/TakyonLayout';
import Loadable from 'ui-component/Loadable';

// dashboard routing
const Login = Loadable(lazy(() => import('views/Login')));
const RWALanding = Loadable(lazy(() => import('views/RWALanding')));

// ==============================|| MAIN ROUTING ||============================== //

const TakyonRoutes = {
  path: '/',
  element: <TakyonLayout />,
  children: [
    {
      path: '/',
      element: <RWALanding />
    },
    {
      path: '/login',
      element: <Login />
    }
  ]
};

export default TakyonRoutes;
