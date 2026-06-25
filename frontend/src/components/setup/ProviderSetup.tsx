import { useCallback, useEffect, useState } from 'react'
import { Check, X, Loader2, Cpu, Terminal } from 'lucide-react'
import type { ProviderStatus, ProviderOptions } from '../../types'

const api = {
  get: (p: string) => fetch(`/api${p}`).then(r => r.json()),
  post: (p: string, body: unknown) =>
    fetch(`/api${p}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      .then(r => r.json()),
}

interface VerifyState { ok: boolean; detail: string }

export default function ProviderSetup({ onConfigured }: { onConfigured: () => void }) {
  const [opts, setOpts] = useState<ProviderOptions | null>(null)
  const [status, setStatus] = useState<ProviderStatus | null>(null)
  const [macroMode, setMacroMode] = useState<string>('local')
  const [backend, setBackend] = useState<string>('claude')
  const [verify, setVerify] = useState<{ synthesis?: VerifyState; macro?: VerifyState }>({})
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([api.get('/providers/options'), api.get('/providers/status')])
      .then(([o, s]: [ProviderOptions, ProviderStatus]) => {
        setOpts(o); setStatus(s)
        setBackend(s.active?.synthesis?.backend || 'claude')
        setMacroMode(s.active?.macro?.mode || 'local')
      })
      .catch(() => setError('Could not load setup options'))
  }, [])

  const runVerify = useCallback(async () => {
    setBusy('verify'); setError(null)
    try {
      const r = await api.post('/providers/verify', { synthesis_backend: backend, macro_mode: macroMode })
      setVerify(r)
    } catch { setError('Verify failed') }
    finally { setBusy(null) }
  }, [backend, macroMode])

  const save = useCallback(async () => {
    setBusy('save'); setError(null)
    try {
      const r = await api.post('/providers/config', { synthesis_backend: backend, macro_mode: macroMode })
      if (r.ok) onConfigured()
      else setError(r.error || 'Save failed')
    } catch { setError('Save failed') }
    finally { setBusy(null) }
  }, [backend, macroMode, onConfigured])

  if (!opts || !status) {
    return (
      <div className="h-full flex items-center justify-center bg-bg-primary text-text-secondary">
        <Loader2 className="animate-spin" size={20} />
      </div>
    )
  }

  const synthInstalled = (b: string) => status.synthesis_clis[b]?.installed

  return (
    <div className="h-full overflow-y-auto bg-bg-primary flex justify-center">
      <div className="w-full max-w-2xl px-6 py-10">
        <h1 className="text-xl font-semibold text-white">Connect your engines</h1>
        <p className="text-sm text-text-secondary mt-1.5">
          StockSense runs on your own LLMs. Pick the synthesis CLI you have logged in, and where the
          macro/news layer should run. Nothing here uses an API key directly.
        </p>

        {/* Synthesis CLI */}
        <section className="mt-7">
          <div className="flex items-center gap-2 text-sm font-medium text-white">
            <Terminal size={15} className="text-accent" /> Synthesis CLI
            <span className="text-text-secondary font-normal">— validates the final calls</span>
          </div>
          <div className="mt-3 space-y-2">
            {opts.synthesis.map(s => {
              const installed = synthInstalled(s.backend)
              const sel = backend === s.backend
              return (
                <button key={s.backend} onClick={() => setBackend(s.backend)}
                  className={`w-full text-left p-3 rounded-lg border transition-colors ${
                    sel ? 'border-accent bg-accent/10' : 'border-border bg-bg-card hover:border-text-secondary'}`}>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-white font-medium">{s.label}</span>
                    {installed
                      ? <span className="flex items-center gap-1 text-[10px] text-green"><Check size={11} /> installed</span>
                      : <span className="flex items-center gap-1 text-[10px] text-yellow">not found</span>}
                  </div>
                  {!installed && (
                    <div className="mt-1.5 text-[11px] text-text-secondary font-mono">
                      install: {s.install}<br />login: {s.login}
                    </div>
                  )}
                </button>
              )
            })}
          </div>
        </section>

        {/* Macro layer */}
        <section className="mt-7">
          <div className="flex items-center gap-2 text-sm font-medium text-white">
            <Cpu size={15} className="text-accent" /> Macro / news layer
            <span className="text-text-secondary font-normal">— sector sentiment</span>
          </div>
          <div className="mt-3 space-y-2">
            {opts.macro.map(m => {
              const sel = macroMode === m.mode
              return (
                <button key={m.mode} onClick={() => setMacroMode(m.mode)}
                  className={`w-full text-left p-3 rounded-lg border transition-colors ${
                    sel ? 'border-accent bg-accent/10' : 'border-border bg-bg-card hover:border-text-secondary'}`}>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-white font-medium">{m.label}</span>
                    {m.mode === 'cloud' && status.macro.has_cloud_key &&
                      <span className="text-[10px] text-green flex items-center gap-1"><Check size={11} /> key set</span>}
                  </div>
                  <div className="mt-1 text-[11px] text-text-secondary">{m.note}</div>
                  <div className="mt-0.5 text-[11px] text-text-secondary font-mono">{m.install}</div>
                </button>
              )
            })}
          </div>
        </section>

        {/* Verify results */}
        {(verify.synthesis || verify.macro) && (
          <div className="mt-5 space-y-1.5 text-xs">
            {verify.synthesis && <VerifyRow label="Synthesis CLI" v={verify.synthesis} />}
            {verify.macro && <VerifyRow label="Macro endpoint" v={verify.macro} />}
          </div>
        )}

        {error && <div className="mt-4 text-xs text-red">{error}</div>}

        <div className="mt-6 flex items-center gap-3">
          <button onClick={runVerify} disabled={!!busy}
            className="flex items-center gap-2 text-sm px-4 py-2 rounded bg-bg-hover text-text-primary hover:text-white disabled:opacity-40">
            {busy === 'verify' ? <Loader2 size={14} className="animate-spin" /> : null} Verify
          </button>
          <button onClick={save} disabled={!!busy}
            className="flex items-center gap-2 text-sm px-4 py-2 rounded bg-accent text-white disabled:opacity-40">
            {busy === 'save' ? <Loader2 size={14} className="animate-spin" /> : null} Save and continue
          </button>
          <span className="text-[11px] text-text-secondary">You can change this later from the header.</span>
        </div>
      </div>
    </div>
  )
}

function VerifyRow({ label, v }: { label: string; v: VerifyState }) {
  return (
    <div className={`flex items-center gap-2 ${v.ok ? 'text-green' : 'text-red'}`}>
      {v.ok ? <Check size={13} /> : <X size={13} />}
      <span className="text-text-primary">{label}:</span>
      <span>{v.ok ? 'working' : 'not ready'}</span>
      <span className="text-text-secondary">— {v.detail}</span>
    </div>
  )
}
