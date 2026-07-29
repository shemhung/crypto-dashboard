const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  "http://127.0.0.1:8000";


interface ApiErrorResponse {
  detail?: unknown;
}


function getErrorMessage(
  detail: unknown,
  status: number,
): string {
  if (typeof detail === "string") {
    return detail;
  }

  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (
          typeof item === "object" &&
          item !== null &&
          "msg" in item &&
          typeof item.msg === "string"
        ) {
          return item.msg;
        }

        return JSON.stringify(item);
      })
      .join("；");
  }

  return `API request failed: HTTP ${status}`;
}


async function parseError(
  response: Response,
): Promise<Error> {
  try {
    const body =
      (await response.json()) as ApiErrorResponse;

    return new Error(
      getErrorMessage(
        body.detail,
        response.status,
      ),
    );
  } catch {
    return new Error(
      `API request failed: HTTP ${response.status}`,
    );
  }
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
    throw await parseError(response);
  }

  return (await response.json()) as T;
}


export async function apiPost<
  TResponse,
  TRequest,
>(
  path: string,
  body: TRequest,
  signal?: AbortSignal,
): Promise<TResponse> {
  const response = await fetch(
    `${API_BASE_URL}${path}`,
    {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      signal,
    },
  );

  if (!response.ok) {
    throw await parseError(response);
  }

  return (await response.json()) as TResponse;
}