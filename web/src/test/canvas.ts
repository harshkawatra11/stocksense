// jsdom has no canvas: a recording-free 2d context so the imperative islands can run in tests.
export function installFakeCanvas(): void {
  const store: Record<string, unknown> = {};
  const ctx = new Proxy(store, {
    get(t, k: string) {
      if (k === "measureText") return () => ({ width: 10 });
      if (k === "createLinearGradient") return () => ({ addColorStop() {} });
      return k in t ? t[k] : () => {};
    },
    set(t, k: string, v) {
      t[k] = v;
      return true;
    },
  });
  HTMLCanvasElement.prototype.getContext = (() => ctx) as unknown as typeof HTMLCanvasElement.prototype.getContext;
}
