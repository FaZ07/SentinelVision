import { useEffect, useRef, useState } from 'react'
import { api } from './api'

/* ─────────────────────────  shared bits  ───────────────────────── */
function Stat({ k, v }) {
  return (
    <div>
      <div className="stat-k">{k}</div>
      <div className="stat-v">{v}</div>
    </div>
  )
}

function Spinner({ label }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm text-slate-400">
      <span className="w-4 h-4 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
      {label}
    </span>
  )
}

/* ─────────────────────────  header  ───────────────────────── */
function Header({ health }) {
  return (
    <header className="flex items-center justify-between flex-wrap gap-4 mb-10">
      <div>
        <h1 className="text-3xl font-black text-white tracking-tight flex items-center gap-2">
          🔭 SentinelVision
        </h1>
        <p className="text-slate-400 text-sm mt-1">
          Zero-shot industrial anomaly detection · CLIP + FAISS memory bank · no labels, no training, no API
        </p>
      </div>
      {health && (
        <div className="card px-5 py-3 flex gap-6 text-sm">
          <div>
            <div className="stat-k">Model</div>
            <div className="font-mono text-slate-300">{health.model.split(' / ')[0]}</div>
          </div>
          <div>
            <div className="stat-k">Device</div>
            <div className="font-mono text-brand-400 uppercase">{health.device}</div>
          </div>
          <div>
            <div className="stat-k">Gallery</div>
            <div className={`font-mono ${health.gallery_ready ? 'text-emerald-400' : 'text-amber-400'}`}>
              {health.gallery_ready ? `${health.gallery_tiles} tiles` : 'not built'}
            </div>
          </div>
        </div>
      )}
    </header>
  )
}

/* ─────────────────────────  step 1 · gallery  ───────────────────────── */
function GalleryBuilder({ onBuilt }) {
  const [files, setFiles] = useState([])
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [stats, setStats] = useState(null)

  const build = async () => {
    setBusy(true); setErr('')
    try {
      const s = await api.buildGallery(files)
      setStats(s); onBuilt()
    } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }

  return (
    <section className="card p-6">
      <div className="flex items-center gap-3 mb-1">
        <span className="w-7 h-7 rounded-lg bg-brand-500 text-ink font-black grid place-items-center text-sm">1</span>
        <h2 className="text-lg font-bold text-white">Teach “normal”</h2>
      </div>
      <p className="text-slate-400 text-sm mb-4 ml-10">
        Upload ≥3 known-good parts. CLIP embeds every image tile into a FAISS memory bank and
        auto-calibrates the decision threshold. No defect samples needed.
      </p>

      <label className="block border-2 border-dashed border-edge rounded-xl p-8 text-center cursor-pointer hover:border-brand-500 transition">
        <input type="file" accept="image/*" multiple className="hidden"
               onChange={(e) => setFiles(e.target.files)} />
        <div className="text-slate-300 font-semibold">
          {files.length ? `${files.length} image(s) selected` : 'Drop / select known-good images'}
        </div>
        <div className="text-slate-500 text-xs mt-1">PNG · JPG · run scripts/make_sample_data.py for a demo set</div>
      </label>

      <div className="flex items-center gap-3 mt-4">
        <button className="btn-primary" disabled={busy || files.length < 3} onClick={build}>
          {busy ? <Spinner label="Building bank…" /> : 'Build memory bank'}
        </button>
        {err && <span className="text-rose-400 text-sm">{err}</span>}
      </div>

      {stats && (
        <div className="grid grid-cols-3 gap-4 mt-6 pt-5 border-t border-edge">
          <Stat k="Images" v={stats.images} />
          <Stat k="Tiles banked" v={stats.tiles} />
          <Stat k="Threshold" v={stats.threshold.toFixed(3)} />
        </div>
      )}
    </section>
  )
}

/* ─────────────────────────  step 2 · inspect  ───────────────────────── */
function VerdictBadge({ verdict, confidence }) {
  const ok = verdict === 'PASS'
  return (
    <div className={`rounded-xl px-5 py-4 border ${ok
      ? 'bg-emerald-500/10 border-emerald-500/40'
      : 'bg-rose-500/10 border-rose-500/40'}`}>
      <div className={`text-3xl font-black ${ok ? 'text-emerald-400' : 'text-rose-400'}`}>
        {ok ? '✓ PASS' : '✕ DEFECT'}
      </div>
      <div className="text-slate-400 text-xs mt-1">
        confidence {(confidence * 100).toFixed(0)}%
      </div>
    </div>
  )
}

function TileGrid({ grid, threshold }) {
  if (!grid?.length) return null
  const flat = grid.flat()
  const max = Math.max(...flat, threshold * 1.5)
  return (
    <div>
      <div className="stat-k mb-2">Per-tile anomaly map</div>
      <div className="inline-grid gap-1 p-2 bg-ink rounded-lg"
           style={{ gridTemplateColumns: `repeat(${grid.length}, 1fr)` }}>
        {flat.map((s, i) => {
          const t = Math.min(1, s / max)
          const hot = s > threshold
          return (
            <div key={i} title={s.toFixed(3)}
              className="w-7 h-7 rounded grid place-items-center text-[9px] font-mono"
              style={{
                background: `rgba(${Math.round(255 * t)},${Math.round(120 * (1 - t))},60,${0.25 + 0.6 * t})`,
                outline: hot ? '1px solid #fb7185' : 'none',
              }}>
              {s.toFixed(2).slice(1)}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function Inspector({ ready }) {
  const [preview, setPreview] = useState(null)
  const [res, setRes] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const fileRef = useRef(null)

  const run = async (file) => {
    setBusy(true); setErr(''); setRes(null)
    setPreview(URL.createObjectURL(file))
    try { setRes(await api.inspect(file)) }
    catch (e) { setErr(e.message) }
    finally { setBusy(false) }
  }

  return (
    <section className="card p-6">
      <div className="flex items-center gap-3 mb-1">
        <span className="w-7 h-7 rounded-lg bg-brand-500 text-ink font-black grid place-items-center text-sm">2</span>
        <h2 className="text-lg font-bold text-white">Inspect a part</h2>
      </div>
      <p className="text-slate-400 text-sm mb-4 ml-10">
        Score = blended global + worst-region cosine distance to the normal manifold.
        Heat-map shows <em>where</em> the model sees the defect.
      </p>

      <button className="btn-ghost" disabled={!ready} onClick={() => fileRef.current?.click()}>
        {ready ? 'Choose image to inspect' : 'Build the gallery first'}
      </button>
      <input ref={fileRef} type="file" accept="image/*" className="hidden"
             onChange={(e) => e.target.files[0] && run(e.target.files[0])} />

      {busy && <div className="mt-4"><Spinner label="Running CLIP + memory-bank search…" /></div>}
      {err && <p className="text-rose-400 text-sm mt-4">{err}</p>}

      {res && (
        <div className="mt-6 grid lg:grid-cols-2 gap-6">
          <div className="space-y-3">
            <div className="stat-k">Input</div>
            <img src={preview} alt="input" className="rounded-xl border border-edge w-full" />
          </div>
          <div className="space-y-3">
            <div className="stat-k">Anomaly heat-map</div>
            <img src={res.overlay} alt="overlay" className="rounded-xl border border-edge w-full" />
          </div>
          <div className="lg:col-span-2 grid sm:grid-cols-2 gap-6 items-center pt-5 border-t border-edge">
            <VerdictBadge verdict={res.verdict} confidence={res.confidence} />
            <div className="grid grid-cols-2 gap-4">
              <Stat k="Anomaly score" v={res.score.toFixed(3)} />
              <Stat k="Threshold" v={res.threshold.toFixed(3)} />
            </div>
            <div className="sm:col-span-2">
              <TileGrid grid={res.tile_grid} threshold={res.threshold} />
            </div>
          </div>
        </div>
      )}
    </section>
  )
}

/* ─────────────────────────  step 3 · benchmark + onnx  ───────────────────────── */
function Cell({ label, v, good }) {
  return (
    <div className="bg-ink rounded-lg p-3 text-center">
      <div className="stat-k">{label}</div>
      <div className={`text-xl font-extrabold font-mono ${good ? 'text-emerald-400' : 'text-slate-200'}`}>{v}</div>
    </div>
  )
}

function BenchAndOnnx({ ready }) {
  const [bench, setBench] = useState(null)
  const [onnx, setOnnx] = useState(null)
  const [b1, setB1] = useState(false)
  const [b2, setB2] = useState(false)
  const [err, setErr] = useState('')

  const runBench = async () => {
    setB1(true); setErr('')
    try { setBench(await api.benchmark()) } catch (e) { setErr(e.message) } finally { setB1(false) }
  }
  const runOnnx = async () => {
    setB2(true); setErr('')
    try { setOnnx(await api.exportOnnx()) } catch (e) { setErr(e.message) } finally { setB2(false) }
  }

  return (
    <section className="card p-6">
      <div className="flex items-center gap-3 mb-1">
        <span className="w-7 h-7 rounded-lg bg-brand-500 text-ink font-black grid place-items-center text-sm">3</span>
        <h2 className="text-lg font-bold text-white">Validate &amp; ship</h2>
      </div>
      <p className="text-slate-400 text-sm mb-4 ml-10">
        Benchmark on a held-out test set (AUROC / confusion), then export the exact
        encoder to ONNX with verified PyTorch parity for edge deployment.
      </p>

      <div className="flex gap-3 flex-wrap">
        <button className="btn-ghost" disabled={!ready || b1} onClick={runBench}>
          {b1 ? <Spinner label="Scoring test set…" /> : 'Run benchmark'}
        </button>
        <button className="btn-ghost" disabled={b2} onClick={runOnnx}>
          {b2 ? <Spinner label="Exporting ONNX…" /> : 'Export ONNX'}
        </button>
      </div>
      {err && <p className="text-rose-400 text-sm mt-3">{err}</p>}

      {bench && (
        <div className="grid grid-cols-3 sm:grid-cols-6 gap-3 mt-6">
          <Cell label="AUROC" v={bench.auroc} good={bench.auroc > 0.85} />
          <Cell label="Accuracy" v={bench.accuracy} good={bench.accuracy > 0.85} />
          <Cell label="Precision" v={bench.precision} />
          <Cell label="Recall" v={bench.recall} />
          <Cell label="F1" v={bench.f1} good={bench.f1 > 0.85} />
          <Cell label="N" v={bench.n} />
          <div className="col-span-3 sm:col-span-6 grid grid-cols-4 gap-3">
            <Cell label="TP" v={bench.confusion.tp} />
            <Cell label="TN" v={bench.confusion.tn} />
            <Cell label="FP" v={bench.confusion.fp} />
            <Cell label="FN" v={bench.confusion.fn} />
          </div>
        </div>
      )}

      {onnx && (
        <div className="mt-6 grid grid-cols-3 gap-4 pt-5 border-t border-edge">
          <Stat k="ONNX size" v={`${onnx.size_mb} MB`} />
          <Stat k="Max |Δ| vs Torch" v={onnx.max_abs_diff.toExponential(1)} />
          <div>
            <div className="stat-k">Parity</div>
            <div className={`text-2xl font-extrabold ${onnx.parity_ok ? 'text-emerald-400' : 'text-rose-400'}`}>
              {onnx.parity_ok ? '✓ verified' : '✕ failed'}
            </div>
          </div>
        </div>
      )}
    </section>
  )
}

/* ─────────────────────────  app  ───────────────────────── */
export default function App() {
  const [health, setHealth] = useState(null)
  const refresh = () => api.health().then(setHealth).catch(() => {})
  useEffect(() => { refresh() }, [])

  return (
    <div className="max-w-5xl mx-auto px-6 py-10">
      <Header health={health} />
      <div className="space-y-6">
        <GalleryBuilder onBuilt={refresh} />
        <Inspector ready={health?.gallery_ready} />
        <BenchAndOnnx ready={health?.gallery_ready} />
      </div>
      <footer className="text-center text-slate-600 text-xs mt-12">
        SentinelVision · PatchCore-style memory bank on a frozen CLIP backbone ·
        runs fully offline · Mohamed Fazil
      </footer>
    </div>
  )
}
