import React, { lazy, useEffect, useState } from 'react';
import { useRoutes } from 'react-router-dom';

// routes
import TakyonRoutes from './TakyonRoutes';
import MobileBuybakRoutes from './MobileBuybakRoutes';

// ==============================|| ROUTING RENDER ||============================== //

export default function ThemeRoutes() {

const [width, setWidth] = useState(window.innerWidth);

  function handleWindowSizeChange() {
    setWidth(window.innerWidth);
  }
  useEffect(() => {
    window.addEventListener("resize", handleWindowSizeChange);
    return () => {
      window.removeEventListener("resize", handleWindowSizeChange);
    };
  }, []);

  const isMobile = width <= 768;

  const routes = [MobileBuybakRoutes]

  return useRoutes(routes);
}
