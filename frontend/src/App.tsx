import {
  NavLink,
  Route,
  Routes,
} from "react-router-dom";

import BacktestPage from "./pages/BacktestPage";
import DashboardPage from "./pages/DashboardPage";
import DataStatusPage from "./pages/DataStatusPage";
import RiskAnalysisPage from "./pages/RiskAnalysisPage";

import "./App.css";


const navigationItems = [
  {
    path: "/",
    label: "Dashboard",
    icon: "◫",
    end: true,
  },
  {
    path: "/risk",
    label: "Risk Analysis",
    icon: "◒",
    end: false,
  },
  {
    path: "/backtest",
    label: "Backtest",
    icon: "↗",
    end: false,
  },
  {
    path: "/data-status",
    label: "Data Status",
    icon: "●",
    end: false,
  },
];


function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header__inner">
          <NavLink
            to="/"
            className="app-brand"
            aria-label="前往 Dashboard"
          >
            <span className="app-brand__logo">
              ₿
            </span>

            <span className="app-brand__text">
              <small>
                CRYPTO INTELLIGENCE
              </small>

              <strong>
                Crypto Risk Dashboard
              </strong>
            </span>
          </NavLink>


          <nav
            className="app-navigation"
            aria-label="主要導覽列"
          >
            {navigationItems.map(
              (item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  end={item.end}
                  className={({
                    isActive,
                  }) => {
                    return isActive
                      ? "app-navigation__link active"
                      : "app-navigation__link";
                  }}
                >
                  <span
                    className={
                      "app-navigation__icon"
                    }
                    aria-hidden="true"
                  >
                    {item.icon}
                  </span>

                  <span>{item.label}</span>
                </NavLink>
              ),
            )}
          </nav>
        </div>
      </header>


      <main className="app-main">
        <Routes>
          <Route
            path="/"
            element={<DashboardPage />}
          />

          <Route
            path="/risk"
            element={<RiskAnalysisPage />}
          />

          <Route
            path="/backtest"
            element={<BacktestPage />}
          />

          <Route
            path="/data-status"
            element={<DataStatusPage />}
          />
        </Routes>
      </main>


      <footer className="app-footer">
        <div className="app-footer__inner">
          <span>
            Crypto Risk Dashboard
          </span>

          <span>
            FastAPI · React · PostgreSQL
          </span>
        </div>
      </footer>
    </div>
  );
}


export default App;