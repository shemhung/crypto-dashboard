export interface HealthResponse {
  status: string;
  service: string;
}

const API_BASE_URL = "http://127.0.0.1:8000";

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);

  if (!response.ok) {
    throw new Error(`Health API failed: ${response.status}`);
  }

  return response.json() as Promise<HealthResponse>;
}