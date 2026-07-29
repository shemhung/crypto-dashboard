import { NavLink, Outlet } from "react-router-dom";

function AppLayout() {
  return (
    <div>
      <header>
        <h1>Crypto Risk Dashboard</h1>

        <nav>
          <NavLink to="/">Dashboard</NavLink>
          {" | "}
          <NavLink to="/risk">Risk Analysis</NavLink>
          {" | "}
          <NavLink to="/backtest">Backtest</NavLink>
          {" | "}
          <NavLink to="/status">Data Status</NavLink>
        </nav>
      </header>

      <hr />

      <main>
        <Outlet />
      </main>
    </div>
  );
}

export default AppLayout;