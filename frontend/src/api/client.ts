const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  "http://127.0.0.1:8000";

interface ApiErrorResponse {
  detail?: string;
}

export async function apiGet<T>(
  path: string,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(
    `${API_BASE_URL}${path}`,
    {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
      signal,
    },
  );

  if (!response.ok) {
    let errorMessage = `API request failed: HTTP ${response.status}`;

    try {
      const body =
        (await response.json()) as ApiErrorResponse;

      if (body.detail) {
        errorMessage = body.detail;
      }
    } catch {
      // 回應內容不一定是 JSON，保留原本 HTTP 錯誤訊息。
    }

    throw new Error(errorMessage);
  }

  return (await response.json()) as T;
}