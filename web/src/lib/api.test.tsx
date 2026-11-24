import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { usePoll } from "@/lib/api";

function setVisibility(v: "visible" | "hidden") {
  Object.defineProperty(document, "visibilityState", { value: v, configurable: true });
}

describe("usePoll", () => {
  let fetchMock: ReturnType<typeof vi.fn>;
  beforeEach(() => {
    vi.useFakeTimers();
    setVisibility("visible");
    fetchMock = vi.fn(async () => ({ ok: true, json: async () => ({ n: 1 }) }));
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    setVisibility("visible");
  });

  it("fetches at once and then every interval", async () => {
    const { result } = renderHook(() => usePoll<{ n: number }>("/api/x", 1000));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(result.current.data).toEqual({ n: 1 });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("stops on unmount", async () => {
    const { unmount } = renderHook(() => usePoll("/api/x", 1000));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000);
    });
    const n = fetchMock.mock.calls.length;
    unmount();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });
    expect(fetchMock).toHaveBeenCalledTimes(n);
  });

  it("pauses while the tab is hidden", async () => {
    renderHook(() => usePoll("/api/x", 1000));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    setVisibility("hidden");
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("does nothing when disabled or path is null", async () => {
    renderHook(() => usePoll("/api/x", 1000, { enabled: false }));
    renderHook(() => usePoll(null, 1000));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
