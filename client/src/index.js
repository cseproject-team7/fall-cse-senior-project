/*
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

*/

import React from 'react';
import ReactDOM from 'react-dom/client';
import {
  createBrowserRouter,
  RouterProvider,
} from "react-router-dom";

import './index.css';
import App from './App';                 // Your ML prediction page
import LogDashboard from './LogDashboard'; // Your analytics page
import Layout from './Layout';           // Your main layout with sidebar
import LoginPage from './pages/LoginPage'; // Our new login page
import ProtectedRoute from './components/ProtectedRoute'; // Our new gatekeeper
import { AuthProvider } from './context/AuthContext'; // Our new auth provider

// This is the new router structure
const router = createBrowserRouter([
  {
    path: "/login", // The login page is a public, standalone route
    element: <LoginPage />,
  },
  {
    path: "/",
    element: <ProtectedRoute />, // This protects all child routes
    children: [
      {
        path: "/",
        element: <Layout />, // The Layout (with sidebar) is now protected
        children: [
          {
            index: true,     // index:true means this is the default child ("/")
            element: <App />, 
          },
          {
            path: "dashboard", // This renders at "/dashboard"
            element: <LogDashboard />,
          },
        ]
      }
    ]
  },
]);

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    {/* Wrap the *entire app* in the AuthProvider */}
    <AuthProvider>
      <RouterProvider router={router} />
    </AuthProvider>
  </React.StrictMode>
);
