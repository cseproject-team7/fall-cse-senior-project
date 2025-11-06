// import React from 'react';
// import ReactDOM from 'react-dom/client';
// import './index.css';
// import LogDashboard from './LogDashboard';

// const root = ReactDOM.createRoot(document.getElementById('root'));
// root.render(
//   <React.StrictMode>
//     <LogDashboard />
//   </React.StrictMode>
// );

import React from 'react';
import ReactDOM from 'react-dom/client';
import {
  createBrowserRouter,
  RouterProvider,
} from "react-router-dom";

import './index.css';
import App from './App';                 // Your ML prediction page
import LogDashboard from './LogDashboard'; // Your new analytics page
import Layout from './Layout';           // <-- IMPORT OUR NEW LAYOUT

// This is the new router structure
const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />, // 1. The Layout is the parent for all routes
    children: [
      {
        index: true,     // 2. index:true means this is the default child ("/")
        element: <App />, 
      },
      {
        path: "dashboard", // 3. This renders at "/dashboard"
        element: <LogDashboard />,
      },
    ]
  },
]);

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
