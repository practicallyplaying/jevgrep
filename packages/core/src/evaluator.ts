import { APICallError, experimental_evaluate as evaluate } from "ai";
import { createTypeSafeAi } from "@ai-sdk/typesafe-ai";
import { endpointFor, type CredentialProvider } from "./providers";
import { type createEvaluationCache, type CacheInput } from "./cache";
import { setTimeout as delay } from "node:timers/promises";
import { stripVTControlCharacters } from "node:util";
import { createRateBudget, estimatedInputTokens } from "./rate-budget";

export type EvaluationRequest = {
  state: Parameters<typeof evaluate>[0]["state"];
  questions: Record<string, { type: "boolean"; instructions: string }>;
};

export class EvaluationFailure extends Error {
  constructor(
    public readonly kind:
      | "authentication"
      | "request-limit"
      | "provider"
      | "cancelled"
      | "source-invalid",
    public readonly splitEligible = false,
    public readonly diagnostic?: { statusCode: number; message?: string },
    message?: string,
  ) {
    super(message ?? `Jev evaluation failed: ${kind}`);
    this.name = "EvaluationFailure";
  }
}

export function createEvaluator(options: {
  provider: CredentialProvider;
  apiKey: string;
  baseURL?: string;
  model?: string;
  cache?: ReturnType<typeof createEvaluationCache>;
  policyVersion?: string;
  fetch?: typeof fetch;
  signal: AbortSignal;
  requestLimit?: number;
  timeoutMs?: number;
  concurrency?: number;
}) {
  const endpoint = endpointFor(options);
  const concurrency = options.concurrency ?? 32;
  const timeoutMs =
    options.timeoutMs ?? (options.provider === "typesafe" || options.provider === "laya" ? 60_000 : 15_000);
  if (!Number.isSafeInteger(concurrency) || concurrency < 1)
    throw new Error("Concurrency must be a positive integer");
  let requests = 0;
  let cacheHits = 0;
  let cooldownUntil = 0;
  const rateBudget =
    options.provider === "typesafe"
      ? createRateBudget({ tokensPerSecond: 100_000, requestsPerSecond: 40 })
      : undefined;
  const authenticationFailure = new AbortController();
  function assertActive() {
    if (options.signal.aborted) throw new EvaluationFailure("cancelled");
    if (authenticationFailure.signal.aborted) throw new EvaluationFailure("authentication");
  }
  let active = 0;
  const waiting: Array<{ resolve: () => void; reject: (error: Error) => void }> = [];
  const stopped = AbortSignal.any([options.signal, authenticationFailure.signal]);
  stopped.addEventListener(
    "abort",
    () => {
      const kind = options.signal.aborted ? "cancelled" : "authentication";
      for (const request of waiting.splice(0)) request.reject(new EvaluationFailure(kind));
    },
    { once: true },
  );
  async function acquire() {
    assertActive();
    if (active < concurrency) active++;
    else await new Promise<void>((resolve, reject) => waiting.push({ resolve, reject }));
    return () => {
      const next = waiting.shift();
      if (next) next.resolve();
      else active--;
    };
  }
  const provider = createTypeSafeAi({
    apiKey: options.apiKey,
    baseURL: endpoint.baseURL,
    fetch: async (input, init) => {
      assertActive();
      if (requests >= (options.requestLimit ?? 50_000))
        throw new EvaluationFailure("request-limit");
      requests++;
      const response = await (options.fetch ?? fetch)(input, init);
      if (response.status === 429) {
        const raw = response.headers.get("retry-after");
        const seconds = raw === null ? NaN : Number(raw);
        const date = raw === null ? NaN : Date.parse(raw);
        const wait =
          Number.isFinite(seconds) && seconds >= 0
            ? seconds * 1000
            : Number.isFinite(date)
              ? Math.max(0, date - Date.now())
              : 1000;
        cooldownUntil = Math.max(cooldownUntil, Date.now() + wait);
      }
      return response;
    },
  });
  async function evaluateWithLaya(request: EvaluationRequest): Promise<Record<string, number>> {
    assertActive();
    if (requests >= (options.requestLimit ?? 50_000)) throw new EvaluationFailure("request-limit");
    requests++;
    const state = truncateState(request.state);
    const questions = Object.fromEntries(
      Object.entries(request.questions).map(([id, question]) => [
        id,
        {
          type: "choice",
          prompt: question.instructions,
          instructions: question.instructions,
          criteria: {
            yes: `Yes: ${question.instructions}`,
            no: `No: ${question.instructions}`,
          },
        },
      ]),
    );
    const response = await (options.fetch ?? fetch)(`${endpoint.baseURL}/judge`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state, questions }),
      signal: AbortSignal.any([
        options.signal,
        authenticationFailure.signal,
        AbortSignal.timeout(timeoutMs),
      ]),
    });
    if (!response.ok) {
      const message = stripVTControlCharacters((await response.text()).slice(0, 300));
      throw new EvaluationFailure("provider", false, { statusCode: response.status, message });
    }
    const result = (await response.json()) as {
      answers?: Record<
        string,
        | { type?: string; probability?: unknown; noul?: unknown; probabilities?: Record<string, unknown> }
        | string
      >;
      usage?: { input_tokens?: unknown; inputTokens?: unknown };
    };
    const answers = result.answers;
    if (!answers || typeof answers !== "object") throw new Error("Invalid Laya response");
    return Object.fromEntries(
      Object.keys(request.questions).map((id) => {
        const probability = layaProbability(answers[id]);
        if (!Number.isFinite(probability) || probability < 0 || probability > 1)
          throw new Error("Invalid Laya answer");
        return [id, probability];
      }),
    );
  }
  return {
    get cacheHits() {
      return cacheHits;
    },
    get cacheIssues() {
      return options.cache?.stats().issues ?? [];
    },
    get requests() {
      return requests;
    },
    async evaluate(
      request: EvaluationRequest,
      policy?: { navigation?: boolean; beforeAttempt?: () => Promise<void> },
    ): Promise<Record<string, number>> {
      assertActive();
      const cacheInput: CacheInput = {
        request,
        namespace: {
          model: endpoint.model,
          provider: options.provider,
          endpoint: endpoint.baseURL,
          protocol: "typesafe-ai-3.0.8",
          policyVersion: options.policyVersion ?? "1",
          parserVersion: "tree-sitter-0.27.0-python-0.25.0-go-0.25.0-rust-0.24.0-ts-5.9.3",
          promptVersion: "unit-locators-1",
        },
      };
      const cached = await options.cache?.get(cacheInput);
      assertActive();
      if (
        cached &&
        Object.keys(cached).length === Object.keys(request.questions).length &&
        Object.keys(request.questions).every(
          (id) => typeof cached[id] === "number" && cached[id]! >= 0 && cached[id]! <= 1,
        )
      ) {
        await policy?.beforeAttempt?.();
        assertActive();
        cacheHits++;
        return cached;
      }
      const navigation = policy?.navigation === true;
      if (options.provider === "laya") {
        await policy?.beforeAttempt?.();
        assertActive();
        const scores = await evaluateWithLaya(request);
        await options.cache?.put(cacheInput, scores);
        return scores;
      }
      const reservedTokens = rateBudget ? estimatedInputTokens(request) : 0;
      const multiple = Object.keys(request.questions).length > 1;
      let attemptLimit = navigation && multiple ? 1 : 2;
      for (let attempt = 0; attempt < attemptLimit; attempt++) {
        assertActive();
        if (requests >= (options.requestLimit ?? 50_000))
          throw new EvaluationFailure("request-limit");
        const release = await acquire();
        try {
          assertActive();
          // Wait before validating source and before starting the network timeout.
          // Recheck after validation: another worker may have consumed the budget.
          for (;;) {
            const wait = Math.max(
              cooldownUntil - Date.now(),
              rateBudget?.waitMs(reservedTokens) ?? 0,
            );
            if (wait > 0) {
              try {
                await delay(Math.min(60_000, wait), undefined, { signal: stopped });
              } catch {
                assertActive();
                throw new EvaluationFailure("cancelled");
              }
              continue;
            }
            await policy?.beforeAttempt?.();
            assertActive();
            if (cooldownUntil > Date.now() || (rateBudget?.waitMs(reservedTokens) ?? 0) > 0)
              continue;
            break;
          }
          const reservation = rateBudget?.reserve(reservedTokens);
          try {
            const result = await evaluate({
              model: provider.evaluationModel(endpoint.model),
              ...request,
              maxRetries: 0,
              abortSignal: AbortSignal.any([
                options.signal,
                authenticationFailure.signal,
                AbortSignal.timeout(timeoutMs),
              ]),
            });
            reservation?.reconcile(result.usage.inputTokens);
            const scores = Object.fromEntries(
              Object.keys(request.questions).map((id) => {
                const answer = result.answers[id];
                if (
                  !answer ||
                  answer.type !== "boolean" ||
                  !Number.isFinite(answer.probability) ||
                  answer.probability < 0 ||
                  answer.probability > 1
                )
                  throw new Error("Invalid answer");
                return [id, answer.probability];
              }),
            );
            await options.cache?.put(cacheInput, scores);
            return scores;
          } catch (error) {
            assertActive();
            const status =
              error && typeof error === "object" && "statusCode" in error
                ? error.statusCode
                : undefined;
            if (status === 401 || status === 403) {
              authenticationFailure.abort();
              throw new EvaluationFailure(
                "authentication",
                false,
                providerDiagnostic(error, options.apiKey),
              );
            }
            if (requests >= (options.requestLimit ?? 50_000))
              throw new EvaluationFailure("request-limit");
            const name = error instanceof Error ? error.name : "unknown";
            const transient =
              status === 408 ||
              status === 429 ||
              (typeof status === "number" && status >= 500 && status <= 599) ||
              name === "TimeoutError" ||
              (APICallError.isInstance(error) &&
                error.statusCode === undefined &&
                error.isRetryable);
            if (navigation && status === 429) attemptLimit = Math.max(attemptLimit, 2);
            if ((navigation && !transient) || attempt + 1 === attemptLimit) {
              const diagnostic = providerDiagnostic(error, options.apiKey);
              const description = diagnostic
                ? `HTTP ${diagnostic.statusCode}${diagnostic.message ? `: ${diagnostic.message}` : ""}`
                : name === "TimeoutError"
                  ? `Request timed out after ${timeoutMs} ms`
                  : APICallError.isInstance(error) && error.statusCode === undefined
                    ? "Network request failed (connection unavailable or reset)"
                    : "Invalid or incomplete provider response";
              throw new EvaluationFailure(
                "provider",
                navigation && multiple && transient && status !== 429,
                diagnostic,
                `${description} (max concurrent requests: ${concurrency})`,
              );
            }
          }
        } finally {
          release();
        }
      }
      throw new EvaluationFailure("provider");
    },
  };
}

function truncateState(state: unknown): string {
  const text = typeof state === "string" ? state : JSON.stringify(state);
  const maxChars = 1800;
  if (text.length <= maxChars) return text;
  return `${text.slice(0, maxChars)}\n[truncated for Laya 512-token context]`;
}

function layaProbability(answer: unknown): number {
  if (typeof answer === "string") return /^yes$/i.test(answer) ? 1 : /^no$/i.test(answer) ? 0 : NaN;
  if (!answer || typeof answer !== "object") return NaN;
  const direct = "probability" in answer ? answer.probability : "noul" in answer ? answer.noul : undefined;
  if (typeof direct === "number") return direct;
  const probabilities =
    "probabilities" in answer && answer.probabilities && typeof answer.probabilities === "object"
      ? (answer.probabilities as Record<string, unknown>)
      : undefined;
  const yes = probabilities?.yes ?? probabilities?.true;
  if (typeof yes === "number") return yes;
  const value = "value" in answer ? answer.value : "choice" in answer ? answer.choice : undefined;
  return typeof value === "string" ? layaProbability(value) : NaN;
}

// Only explicit text fields from a parsed provider error are suitable for user-facing diagnostics.
// SDK messages can instead serialize arbitrary detail objects, requests, or response bodies.
function providerDiagnostic(error: unknown, apiKey: string): EvaluationFailure["diagnostic"] {
  if (!APICallError.isInstance(error) || error.statusCode === undefined) return undefined;
  const data: unknown = error.data;
  let message: unknown;
  if (data && typeof data === "object") {
    const nested = "error" in data ? data.error : undefined;
    message =
      ("message" in data && typeof data.message === "string" ? data.message : undefined) ??
      (typeof nested === "string"
        ? nested
        : nested && typeof nested === "object" && "message" in nested
          ? nested.message
          : undefined) ??
      ("detail" in data && typeof data.detail === "string" ? data.detail : undefined);
  }
  if (typeof message !== "string") return { statusCode: error.statusCode };
  const safe = stripVTControlCharacters(message)
    .replace(/[\p{Cc}\p{Cf}]+/gu, " ")
    .trim();
  const redacted = apiKey ? safe.replaceAll(apiKey, "[redacted]") : safe;
  // A provider may split a credential with whitespace or invisible separators.
  if (apiKey && redacted.replace(/\s/gu, "").includes(apiKey))
    return { statusCode: error.statusCode };
  return { statusCode: error.statusCode, message: redacted.slice(0, 500) || undefined };
}
