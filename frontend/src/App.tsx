import {
  createBrowserRouter,
  RouterProvider,
} from "react-router-dom";

import AppLayout from "./components/layout/AppLayout";
import BacktestPage from "./pages/BacktestPage";
import DashboardPage from "./pages/DashboardPage";
import DataStatusPage from "./pages/DataStatusPage";
import RiskAnalysisPage from "./pages/RiskAnalysisPage";

const router = createBrowserRouter([
  {
    path: "/",
    element: <AppLayout />,
    children: [
      {
        index: true,
        element: <DashboardPage />,
      },
      {
        path: "risk",
        element: <RiskAnalysisPage />,
      },
      {
        path: "backtest",
        element: <BacktestPage />,
      },
      {
        path: "status",
        element: <DataStatusPage />,
      },
    ],
  },
]);

function App() {
  return <RouterProvider router={router} />;
}

export default App;