import { useEffect, useState } from "react";
import { getHealth, type HealthResponse } from "./api/health";

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    async function loadHealth() {
      try {
        const result = await getHealth();
        setHealth(result);
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Unknown error";

        setError(message);
      }
    }

    void loadHealth();
  }, []);

  return (
    <main>
      <h1>Crypto Dashboard</h1>

      {error && (
        <p>後端連線失敗：{error}</p>
      )}

      {!error && !health && (
        <p>正在連線後端...</p>
      )}

      {health && (
        <section>
          <h2>Backend status</h2>
          <p>Status: {health.status}</p>
          <p>Service: {health.service}</p>
        </section>
      )}
    </main>
  );
}

export default App;