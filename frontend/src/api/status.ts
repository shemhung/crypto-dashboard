import { apiGet } from "./client";

import type {
  DataStatusResponse,
} from "../types/status";


export function getDataStatus(
  signal?: AbortSignal,
): Promise<DataStatusResponse> {
  return apiGet<DataStatusResponse>(
    "/api/v1/data-status",
    signal,
  );
}