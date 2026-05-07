// Единая точка управления baseline-обучением — polling + mutation. Возвращает
// презентационный объект, который потребляют BaselineCard и StickyActionBar.
//
// Sprint 6, Phase 2: state поднимается из BaselineCard в страницу AnalysisResult,
// чтобы sticky-bar и карточка делили один статус и один error.
import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { analysesApi } from "../api/analyses";
import { useBaselinePolling } from "./useBaselinePolling";
import type {
  BaselineResponse,
  BaselineResult,
  BaselineStatus,
} from "../types/analysis";

export interface BaselineActions {
  status: BaselineStatus;
  result: BaselineResult | null;
  pollingError: string | null;
  startError: string | null;
  isStarting: boolean;
  start: () => void;
}

export function useBaselineActions(
  analysisId: string | undefined,
): BaselineActions {
  const queryClient = useQueryClient();
  const polling = useBaselinePolling(analysisId);
  const [startError, setStartError] = useState<string | null>(null);

  const startMutation = useMutation({
    mutationFn: () => analysesApi.startBaseline(analysisId!),
    onMutate: () => setStartError(null),
    onSuccess: (resp) => {
      queryClient.setQueryData<BaselineResponse>(
        ["baseline", analysisId],
        {
          baseline_status: resp.baseline_status,
          baseline: null,
          baseline_error: null,
        },
      );
      queryClient.invalidateQueries({ queryKey: ["baseline", analysisId] });
    },
    onError: (err: unknown) => {
      const detail =
        (err as { response?: { data?: { detail?: string } } })?.response?.data
          ?.detail ??
        (err instanceof Error ? err.message : "Не удалось запустить обучение");
      setStartError(detail);
    },
  });

  return {
    status: polling.data?.baseline_status ?? "not_started",
    result: polling.data?.baseline ?? null,
    pollingError: polling.data?.baseline_error ?? null,
    startError,
    isStarting: startMutation.isPending,
    start: () => {
      if (analysisId) startMutation.mutate();
    },
  };
}
