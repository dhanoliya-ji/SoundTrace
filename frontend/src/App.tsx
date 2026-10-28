import { useEffect, useRef, useState } from 'react';

const API = ''; // same origin: FastAPI serves this app; Vite proxies /api in dev

type Status = { available: boolean; message: string; checkpoint?: string; variant?: string; val_iou?: number;
  manifest_valid?: boolean; manifest_video_count?: number; manifest_records?: number; dataset?: string };
type Job = { id: string; status: string; error?: string; progress?: number; frame_count?: number; variant?: string };
type Frame = { frame: number; timestamp: number; mask: number[][]; audio_rms?: number; gate_mean?: number };
type Summary = Record<string, { mean_iou: number; examples: number }>;
type Report = { evaluation_id: string; kind: string; dataset?: string; variant?: string; seed?: number; seeds?: number[];
  test_mean_iou?: number; mean_iou?: number; mean_test_iou_delta?: number; std_test_iou_delta?: number;
  pairs?: { seed: number; baseline_test_iou: number; gated_test_iou: number }[]; condition_summary?: Summary };

const pct = (v?: number) => (v === undefined || v === null ? '—' : `${(v * 100).toFixed(1)}%`);
const get = async <T,>(path: string): Promise<T> => {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};

export default function App() {
  const [model, setModel] = useState<Status>();
  const [dataset, setDataset] = useState<Status>();
  const [reports, setReports] = useState<Report[]>([]);
  const [error, setError] = useState('');
  const [file, setFile] = useState<File>();
  const [src, setSrc] = useState('');
  const [job, setJob] = useState<Job>();
  const [frames, setFrames] = useState<Frame[]>([]);
  const [time, setTime] = useState(0);
  const [opacity, setOpacity] = useState(60);
  const [aspect, setAspect] = useState(16 / 9);
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);

  const refresh = () =>
    Promise.all([get<Status>('/api/v1/model/status'), get<Status>('/api/v1/dataset/status'), get<Report[]>('/api/v1/evaluations')])
      .then(([m, d, r]) => { setModel(m); setDataset(d); setReports(r); setError(''); })
      .catch(() => setError('Backend is offline. Start the API to load live status.'));
  useEffect(() => { void refresh(); }, []);

  useEffect(() => {
    if (!file) return setSrc('');
    const url = URL.createObjectURL(file);
    setSrc(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const current = frames.reduce<Frame | undefined>((best, f) => (f.timestamp <= time + 1e-3 ? f : best), frames[0]);
  useEffect(() => {
    const el = canvas.current, ctx = el?.getContext('2d');
    if (!el || !ctx) return;
    if (!current) return ctx.clearRect(0, 0, el.width, el.height);
    el.height = current.mask.length; el.width = current.mask[0].length;
    const img = ctx.createImageData(el.width, el.height);
    current.mask.flat().forEach((v, i) => img.data.set([255, 196, 0, v ? opacity * 2.55 : 0], i * 4));
    ctx.putImageData(img, 0, 0);
  }, [current, opacity]);

  async function run() {
    if (!file) return;
    setError(''); setFrames([]);
    try {
      const form = new FormData();
      form.append('file', file);
      const r = await fetch(`${API}/api/v1/predictions`, { method: 'POST', body: form });
      if (!r.ok) throw new Error((await r.json()).detail || 'Upload failed');
      let j: Job = await r.json();
      setJob(j);
      while (j.status === 'queued' || j.status === 'processing') {
        await new Promise((ok) => setTimeout(ok, 1000));
        j = await get<Job>(`/api/v1/predictions/${j.id}`);
        setJob(j);
      }
      if (j.status === 'completed') setFrames((await get<{ frames: Frame[] }>(`/api/v1/predictions/${j.id}/masks`)).frames);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed');
      setJob(undefined);
    }
  }

  const step = (dir: number) => {
    if (!frames.length || !video.current) return;
    const i = Math.min(frames.length - 1, Math.max(0, frames.indexOf(current!) + dir));
    video.current.pause();
    video.current.currentTime = frames[i].timestamp;
  };

  const busy = job?.status === 'queued' || job?.status === 'processing';
  const study = reports.find((r) => r.kind === 'paired_study');
  const robust = ['baseline', 'gated'].map((v) => reports.find((r) => r.kind === 'robustness' && r.variant === v));

  return (
    <main>
      <header>
        <div><h1>SoundTrace</h1><p>Which pixels are making the sound? Audio-visual source segmentation under modality corruption.</p></div>
        <span className="tag">RESEARCH PROTOTYPE</span>
      </header>
      {error && <div className="notice">{error}</div>}

      <section className="cards">
        <Card title="Model" value={model ? (model.available ? `${model.variant ?? 'trained'} checkpoint` : 'Not configured') : 'Checking…'}
          foot={model?.available ? `validation IoU ${pct(model.val_iou)} · ${model.checkpoint}` : model?.message} ok={model?.available} />
        <Card title="Dataset" value={dataset ? (dataset.manifest_valid ? `${dataset.dataset ?? 'Custom'} manifest` : 'Not ready') : 'Checking…'}
          foot={dataset?.manifest_valid ? `${dataset.manifest_video_count} videos · ${dataset.manifest_records} annotated frames` : dataset?.message} ok={dataset?.manifest_valid} />
        <Card title="Gated − baseline (test IoU)" value={study ? `${((study.mean_test_iou_delta ?? 0) * 100).toFixed(2)} pp` : 'No study yet'}
          foot={study ? `${study.dataset} · ${study.seeds?.length} paired seeds · sd ${((study.std_test_iou_delta ?? 0) * 100).toFixed(2)} pp` : 'Run: soundtrace.cli study'} ok={!!study} />
      </section>

      <h2>Video analysis</h2>
      <section className="panel analysis">
        <div className="controls">
          <label className="drop">
            <input type="file" accept="video/*" onChange={(e) => { setFile(e.target.files?.[0]); setJob(undefined); setFrames([]); }} />
            <b>{file ? file.name : 'Choose a video with sound'}</b>
            <span>{file ? `${(file.size / 1048576).toFixed(1)} MB` : 'MP4 / MOV / MKV / WEBM · up to 200 MB · deleted after processing'}</span>
          </label>
          <button className="primary" disabled={!file || busy || !model?.available} onClick={() => void run()}>
            {busy ? `Processing${job?.progress ? ` ${Math.round(job.progress * 100)}%` : '…'}` : 'Run segmentation'}
          </button>
          {job?.status === 'failed' && <div className="notice">{job.error}</div>}
          {job?.status === 'completed' && <p className="ok">{job.frame_count} masks predicted ({job.variant} model) · no ground truth for uploads</p>}
          <label className="row">Overlay <input type="range" min={0} max={100} value={opacity} onChange={(e) => setOpacity(+e.target.value)} /> {opacity}%</label>
          <div className="row">
            <button onClick={() => step(-1)} disabled={!frames.length}>◀ frame</button>
            <button onClick={() => step(1)} disabled={!frames.length}>frame ▶</button>
            <span className="muted">{current ? `t = ${current.timestamp.toFixed(2)} s · gate ${current.gate_mean?.toFixed(2)}` : ''}</span>
          </div>
        </div>
        <div>
          <div className="screen" style={{ aspectRatio: aspect }}>
            {src ? <video ref={video} src={src} controls onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
              onLoadedMetadata={(e) => setAspect(e.currentTarget.videoWidth / e.currentTarget.videoHeight || 16 / 9)} />
              : <span className="muted">Video preview with predicted mask overlay</span>}
            <canvas ref={canvas} />
          </div>
          {frames.length > 0 && <Timeline frames={frames} time={time} onSeek={(t) => { if (video.current) video.current.currentTime = t; }} />}
        </div>
      </section>

      <h2>Experiments</h2>
      <section className="grid2">
        <div className="panel">
          <h3>Paired study: baseline vs gated</h3>
          {study?.pairs ? <table><thead><tr><th>Seed</th><th>Baseline</th><th>Gated</th><th>Δ</th></tr></thead><tbody>
            {study.pairs.map((p) => <tr key={p.seed}><td>{p.seed}</td><td>{pct(p.baseline_test_iou)}</td><td>{pct(p.gated_test_iou)}</td>
              <td className={p.gated_test_iou >= p.baseline_test_iou ? 'pos' : 'neg'}>{((p.gated_test_iou - p.baseline_test_iou) * 100).toFixed(2)}</td></tr>)}
          </tbody></table> : <p className="muted">No paired study report saved.</p>}
          <p className="muted small">Held-out test IoU on a custom source-video split. Not an official benchmark score.</p>
        </div>
        <div className="panel">
          <h3>Robustness (test IoU per condition)</h3>
          {robust[0] || robust[1] ? <Robustness base={robust[0]?.condition_summary} gated={robust[1]?.condition_summary} />
            : <p className="muted">No robustness report saved.</p>}
        </div>
      </section>

      <section className="panel">
        <h3>Saved reports</h3>
        {reports.length ? <table><thead><tr><th>ID</th><th>Kind</th><th>Variant</th><th>Seed</th><th>IoU</th></tr></thead><tbody>
          {reports.slice(0, 12).map((r) => <tr key={r.evaluation_id}><td><a href={`${API}/api/v1/evaluations/${r.evaluation_id}`} target="_blank" rel="noreferrer">{r.evaluation_id}</a></td>
            <td>{r.kind.replace(/_/g, ' ')}</td><td>{r.variant ?? '—'}</td><td>{r.seed ?? r.seeds?.join(',')}</td>
            <td>{r.mean_test_iou_delta !== undefined ? `Δ ${(r.mean_test_iou_delta * 100).toFixed(2)} pp` : pct(r.test_mean_iou ?? r.mean_iou)}</td></tr>)}
        </tbody></table> : <p className="muted">No reports yet.</p>}
      </section>

      <footer>Data: AVATAR (CC BY 4.0, Choi et al., ICCV 2025). Results use a custom split and are exploratory, not official AVATAR or AVSBench scores. Gate values are diagnostics, not probabilities.</footer>
    </main>
  );
}

function Card({ title, value, foot, ok }: { title: string; value: string; foot?: string; ok?: boolean }) {
  return <div className="panel card"><span className="label">{title}</span><b>{value}</b><span className={ok ? 'ok' : 'muted'}>{foot}</span></div>;
}

function Timeline({ frames, time, onSeek }: { frames: Frame[]; time: number; onSeek: (t: number) => void }) {
  const end = frames[frames.length - 1].timestamp || 1, W = 600, H = 70;
  const area = frames.map((f) => f.mask.flat().reduce((a, b) => a + b, 0) / f.mask.flat().length);
  const rmsMax = Math.max(...frames.map((f) => f.audio_rms ?? 0), 1e-6);
  const line = (vals: number[]) => vals.map((v, i) => `${(frames[i].timestamp / end) * W},${H - v * (H - 6) - 3}`).join(' ');
  return (
    <svg className="timeline" viewBox={`0 0 ${W} ${H}`} onClick={(e) => {
      const box = e.currentTarget.getBoundingClientRect();
      onSeek(((e.clientX - box.left) / box.width) * end);
    }}>
      <polyline points={line(frames.map((f) => (f.audio_rms ?? 0) / rmsMax))} className="audio" />
      <polyline points={line(area)} className="area" />
      <polyline points={line(frames.map((f) => f.gate_mean ?? 0))} className="gate" />
      <line x1={(time / end) * W} x2={(time / end) * W} y1={0} y2={H} className="cursor" />
      <text x={4} y={12}>■ audio level  ■ mask area  ■ gate</text>
    </svg>
  );
}

function Robustness({ base, gated }: { base?: Summary; gated?: Summary }) {
  const names = Object.keys(base ?? gated ?? {});
  return <div className="bars">{names.map((n) => (
    <div key={n} className="bar-row" title={`baseline ${pct(base?.[n]?.mean_iou)} · gated ${pct(gated?.[n]?.mean_iou)}`}>
      <span>{n.replace(/_/g, ' ')}</span>
      <div><i className="b" style={{ width: `${(base?.[n]?.mean_iou ?? 0) * 100}%` }} /><i className="g" style={{ width: `${(gated?.[n]?.mean_iou ?? 0) * 100}%` }} /></div>
    </div>))}
    <p className="small"><i className="key b" /> baseline <i className="key g" /> gated</p>
  </div>;
}
