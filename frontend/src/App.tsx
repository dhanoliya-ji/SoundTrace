import { useEffect, useRef, useState } from 'react';
import {
  Activity,
  Boxes,
  CircleHelp,
  Database,
  FileVideo,
  FlaskConical,
  Gauge,
  Github,
  Layers3,
  Play,
  ShieldCheck,
  Upload,
  Waves,
} from 'lucide-react';

const API = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

type Status = {
  available: boolean;
  message: string;
  checkpoint?: string | null;
  video_count?: number;
  manifest_valid?: boolean;
  manifest_records?: number;
  manifest_video_count?: number;
  dataset?: string | null;
};

type Job = {
  id: string;
  status: string;
  error?: string;
  progress?: number;
  frame_count?: number;
};

type MaskFrame = { frame: number; timestamp: number; mask: number[][] };

type Evaluation = {
  evaluation_id: string;
  kind: string;
  dataset?: string;
  test_mean_iou?: number;
  mean_iou?: number;
  mean_test_iou_delta?: number;
  variant?: string;
  seed?: number;
  seeds?: number[];
  source_revision?: string;
};

export default function App() {
  const [dataset, setDataset] = useState<Status | null>(null);
  const [model, setModel] = useState<Status | null>(null);
  const [evaluations, setEvaluations] = useState<Evaluation[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [videoSrc, setVideoSrc] = useState('');
  const [job, setJob] = useState<Job | null>(null);
  const [maskFrames, setMaskFrames] = useState<MaskFrame[]>([]);
  const [busy, setBusy] = useState(false);
  const [opacity, setOpacity] = useState(62);
  const [playTime, setPlayTime] = useState(0);
  const [error, setError] = useState('');
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);

  async function refreshStatus() {
    setError('');
    try {
      const responses = await Promise.all([
        fetch(`${API}/api/v1/dataset/status`),
        fetch(`${API}/api/v1/model/status`),
        fetch(`${API}/api/v1/evaluations`),
      ]);
      if (responses.some((response) => !response.ok)) throw new Error('API unavailable');
      const [datasetStatus, modelStatus, evaluationList] = await Promise.all(
        responses.map((response) => response.json()),
      );
      setDataset(datasetStatus);
      setModel(modelStatus);
      setEvaluations(evaluationList);
    } catch {
      setError('Backend is offline. Start the API to load live dataset and model status.');
    }
  }

  useEffect(() => {
    void refreshStatus();
  }, []);

  useEffect(() => {
    if (!file) {
      setVideoSrc('');
      return;
    }
    const source = URL.createObjectURL(file);
    setVideoSrc(source);
    return () => URL.revokeObjectURL(source);
  }, [file]);

  useEffect(() => {
    if (job?.status !== 'completed' || !job.id) return;
    fetch(`${API}/api/v1/predictions/${job.id}/masks`)
      .then((response) => response.json())
      .then((data) => setMaskFrames(data.frames || []))
      .catch(() => setError('Prediction is ready, but its mask timeline could not be loaded.'));
  }, [job?.status, job?.id]);

  useEffect(() => {
    const frame = maskFrames.reduce<MaskFrame | undefined>(
      (best, item) => item.timestamp <= playTime && (!best || item.timestamp > best.timestamp)
        ? item
        : best,
      undefined,
    );
    const element = canvas.current;
    if (!element || !frame) return;
    element.width = frame.mask[0]?.length || 128;
    element.height = frame.mask.length;
    const context = element.getContext('2d');
    if (!context) return;
    const image = context.createImageData(element.width, element.height);
    frame.mask.forEach((row, y) => row.forEach((value, x) => {
      const index = (y * element.width + x) * 4;
      image.data[index] = 83;
      image.data[index + 1] = 221;
      image.data[index + 2] = 170;
      image.data[index + 3] = value ? opacity * 2.55 : 0;
    }));
    context.putImageData(image, 0, 0);
  }, [maskFrames, playTime, opacity]);

  async function runPrediction() {
    if (!file) return;
    setError('');
    setBusy(true);
    setJob(null);
    setMaskFrames([]);
    try {
      const form = new FormData();
      form.append('file', file);
      const response = await fetch(`${API}/api/v1/predictions`, { method: 'POST', body: form });
      if (!response.ok) throw new Error((await response.json()).detail || 'Upload failed');
      let current: Job = await response.json();
      setJob(current);
      for (let attempt = 0; attempt < 3600; attempt += 1) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        const poll = await fetch(`${API}/api/v1/predictions/${current.id}`);
        if (!poll.ok) throw new Error('Could not read prediction status');
        current = await poll.json();
        setJob(current);
        if (current.status === 'completed' || current.status === 'failed') break;
      }
      if (current.status === 'processing') {
        setError('This video is still processing. Its saved job can be checked again by refreshing the page.');
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Request failed');
    } finally {
      setBusy(false);
    }
  }

  const avatarStudy = evaluations.find((item) => item.kind === 'paired_study' && item.dataset === 'AVATAR');
  const avatarRobustness = evaluations.filter((item) => item.kind === 'robustness' && item.dataset === 'AVATAR');
  const hasAvatarEvidence = evaluations.some((item) => item.dataset === 'AVATAR');
  const hasManifest = dataset?.manifest_valid ?? hasAvatarEvidence;
  const robustnessPairs = new Set(avatarRobustness.map((item) => `${item.seed}:${item.variant}`));
  const checklist = [
    { label: 'Core model and prediction API', complete: model?.available === true, detail: model?.available ? 'READY' : 'CHECK API' },
    { label: 'Validated AVATAR pilot manifest', complete: hasManifest, detail: hasManifest ? 'READY' : 'NEEDS DATA' },
    {
      label: 'Paired AVATAR study · 3 seeds',
      complete: (avatarStudy?.seeds?.length || 0) >= 3,
      detail: (avatarStudy?.seeds?.length || 0) >= 3 ? 'READY' : 'IN PROGRESS',
    },
    {
      label: 'Matched AVATAR corruption evaluation',
      complete: robustnessPairs.size >= 6,
      detail: robustnessPairs.size >= 6 ? 'READY' : 'IN PROGRESS',
    },
    { label: 'Official AVSBench access and evaluation', complete: false, detail: 'EXTERNAL ACCESS' },
  ];
  const completedCount = checklist.filter((item) => item.complete).length;
  const latestEvaluation = evaluations[0];
  const datasetVideoCount = dataset?.manifest_video_count ?? dataset?.video_count;

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brandmark"><Waves size={20} /></div>
          <div><b>SoundTrace</b><small>RESEARCH CONSOLE</small></div>
        </div>
        <div className="workspace">
          <span className="tiny">WORKSPACE</span>
          <div className="workspace-card">
            <div className="workspace-icon"><Activity size={16} /></div>
            <span>AVATAR pilot · 100 videos</span><span className="dots">···</span>
          </div>
        </div>
        <nav>
          <span className="tiny">ANALYSIS</span>
          <a className="active" href="#overview"><Layers3 size={17} />Overview</a>
          <a href="#analyze"><FileVideo size={17} />Video analysis</a>
          <a href="#experiments"><FlaskConical size={17} />Experiments</a>
          <a href="#data"><Database size={17} />Dataset</a>
          <a href="#model"><Boxes size={17} />Model card</a>
        </nav>
        <div className="side-bottom">
          <div className="help"><CircleHelp size={17} /><span>Research guide</span></div>
          <div className="user">
            <div className="avatar">ST</div>
            <div><b>Local workspace</b><small>PRIVATE SESSION</small></div>
            <span className="dots">···</span>
          </div>
        </div>
      </aside>

      <main id="overview">
        <header>
          <div className="crumb">Research <span>/</span> <b>Overview</b></div>
          <div className="head-right">
            <span className="local"><i /> LOCAL ENVIRONMENT</span>
            <a className="icon-btn" href="https://github.com/OpenNLPLab/AVSBench" target="_blank" rel="noreferrer" aria-label="AVSBench repository"><Github size={17} /></a>
          </div>
        </header>
        <section className="content">
          <div className="title-row">
            <div>
              <div className="eyebrow">AUDIO-VISUAL PERCEPTION <span>·</span> EXPERIMENTAL</div>
              <h1>Research overview</h1>
              <p className="sub">Investigate how modality reliability shapes sounding-object segmentation.</p>
            </div>
            <button className="quiet" onClick={() => void refreshStatus()}>↻ &nbsp;Refresh status</button>
          </div>

          {error && <div className="notice"><span>!</span>{error}</div>}
          <div className="status-banner">
            <div className="pulse"><Activity size={20} /></div>
            <div className="banner-copy">
              <b>Research environment {model?.available ? 'ready' : 'partially connected'}</b>
              <span>{model?.message || 'Checking model status…'} {dataset?.message || 'Checking dataset status…'}</span>
            </div>
            <div className="banner-tag"><i /> PROTOTYPE</div>
          </div>

          <div className="metrics">
            <article id="model" className="metric-card">
              <div className="metric-head"><span>MODEL CHECKPOINT</span><Boxes size={16} /></div>
              <strong className="state">{model === null ? 'Checking…' : model.available ? 'Checkpoint ready' : 'Not configured'}</strong>
              <div className="metric-foot">
                <span className={model?.available ? 'green' : ''}><i /> {model?.available ? 'TRAINED MODEL' : model === null ? 'CHECKING' : 'NO TRAINED WEIGHTS'}</span>
                <span>{model?.checkpoint || '—'}</span>
              </div>
            </article>
            <article id="data" className="metric-card">
              <div className="metric-head"><span>DATASET</span><Database size={16} /></div>
              <strong className="state">{dataset === null ? 'Checking…' : hasManifest ? `${dataset.dataset || 'AVATAR'} manifest ready` : dataset.available ? 'Review required' : 'Not available'}</strong>
              <div className="metric-foot">
                <span className={hasManifest ? 'green' : dataset?.available ? 'amber' : ''}><i /> {hasManifest ? 'VALIDATED MANIFEST' : dataset?.available ? 'REVIEW REQUIRED' : 'NO DATASET'}</span>
                <span>{datasetVideoCount ?? '—'} videos</span>
              </div>
            </article>
            <article className="metric-card">
              <div className="metric-head"><span>LAST EVALUATION</span><Gauge size={16} /></div>
              <strong className="state muted">
                {latestEvaluation
                  ? latestEvaluation.mean_test_iou_delta !== undefined
                    ? `${(latestEvaluation.mean_test_iou_delta * 100).toFixed(1)} pp Δ`
                    : `${((latestEvaluation.test_mean_iou ?? latestEvaluation.mean_iou ?? 0) * 100).toFixed(1)}% IoU`
                  : 'No evaluation yet'}
              </strong>
              <div className="metric-foot">
                <span><i /> {latestEvaluation?.variant?.toUpperCase() || latestEvaluation?.kind?.replace(/_/g, ' ').toUpperCase() || 'AWAITING DATA'}</span>
                <span>{latestEvaluation?.evaluation_id || '—'}</span>
              </div>
            </article>
          </div>

          <div className="section-head">
            <div><h2>Video analysis</h2><p>Upload a clip to inspect the trained model’s mask timeline as the video plays.</p></div>
            <span className="privacy"><ShieldCheck size={15} /> LOCAL · TEMPORARY</span>
          </div>
          <section id="analyze" className="analysis">
            <div className="upload-col">
              <label className="drop">
                <input type="file" accept="video/mp4,video/quicktime,video/x-matroska,video/webm" onChange={(event) => { setFile(event.target.files?.[0] || null); setJob(null); setMaskFrames([]); }} />
                <div className="upload-icon"><Upload size={20} /></div>
                <b>{file ? file.name : 'Drop a video to begin'}</b>
                <span>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB · Ready for upload` : 'or browse files from your device'}</span>
                <em>MP4, MOV, MKV or WEBM · up to 200 MB</em>
              </label>
              <div className="field"><label>MODEL VARIANT</label><div className="select">{model?.available ? 'Configured checkpoint' : 'Unavailable until checkpoint is configured'}<span>⌄</span></div></div>
              <button className="primary" disabled={!file || busy || !model?.available} onClick={() => void runPrediction()}>
                {busy ? <span className="spinner" /> : <Play size={16} fill="currentColor" />}
                {busy ? (job?.status === 'queued' ? 'Starting prediction…' : `Processing video${job?.progress ? ` · ${Math.round(job.progress * 100)}%` : ''}`) : 'Run segmentation'}
                <span className="arrow">→</span>
              </button>
              {job && <div className={`job ${job.status}`}>{job.status === 'completed' ? `Prediction ready · ${job.frame_count} sampled masks` : job.status === 'failed' ? job.error : `Job ${job.status}`}</div>}
            </div>
            <div className="preview">
              <div className="preview-top"><span>LIVE PREVIEW</span><span className="badge">{maskFrames.length ? 'MASK TIMELINE' : job?.status === 'processing' ? 'PROCESSING' : 'AWAITING VIDEO'}</span></div>
              <div className="screen">
                {file && <video ref={video} src={videoSrc} controls onTimeUpdate={(event) => setPlayTime(event.currentTarget.currentTime)} />}
                <canvas className="mask" ref={canvas} />
                {!file && <div className="screen-empty"><div className="play-ring"><Play size={21} /></div><span>Video preview and mask overlay</span><small>Upload a clip to view model output</small></div>}
              </div>
              <div className="overlay-control"><span>MASK OVERLAY</span><input type="range" min="0" max="100" value={opacity} onChange={(event) => setOpacity(+event.target.value)} /><b>{opacity}%</b></div>
            </div>
          </section>

          <div className="lower-grid">
            <section id="experiments" className="panel">
              <div className="panel-title">
                <div><h3>Experiment activity</h3><span>Saved reports from this workspace</span></div>
                <button className="text-btn" onClick={() => void refreshStatus()}>REFRESH ↻</button>
              </div>
              {evaluations.length ? <div className="run-list">{evaluations.slice(0, 6).map((run) => (
                <article className="run-row" key={run.evaluation_id}>
                  <div className="run-mark"><FlaskConical size={15} /></div>
                  <div className="run-info">
                    <b>{(run.variant || run.kind || 'experiment').toUpperCase()} · {run.kind.replace(/_/g, ' ')}</b>
                    <span>{run.evaluation_id} · {run.seed !== undefined ? `seed ${run.seed}` : run.seeds ? `${run.seeds.length} paired seeds` : 'study'}{run.source_revision ? ` · ${run.source_revision.slice(0, 7)}` : ''}</span>
                  </div>
                  <strong>{run.mean_test_iou_delta !== undefined ? <>{(run.mean_test_iou_delta * 100).toFixed(1)}<small>pp Δ</small></> : <>{((run.test_mean_iou ?? run.mean_iou ?? 0) * 100).toFixed(1)}<small>% IoU</small></>}</strong>
                </article>
              ))}</div> : <div className="empty"><div className="empty-icon"><FlaskConical size={19} /></div><b>No saved experiment reports</b><span>Run training or held-out evaluation to create a report.</span><code>python -m soundtrace.cli study --config configs/avatar.yaml</code></div>}
            </section>
            <section className="panel">
              <div className="panel-title"><div><h3>Research status</h3><span>Evidence-backed study checklist</span></div><span className="small-count">{completedCount} / {checklist.length}</span></div>
              <div className="checklist">{checklist.map((item) => (
                <div key={item.label}>
                  <i className={item.complete ? 'done' : ''}>{item.complete ? '✓' : ''}</i>
                  <span>{item.label}</span><small>{item.detail}</small>
                </div>
              ))}</div>
            </section>
          </div>

          <div className="footnote" id="research-limits">
            <ShieldCheck size={15} />
            <span><b>Research prototype.</b> AVATAR results use a custom split and stress-test suite; they are not official benchmark scores. AVSBench requires separate authorized access.</span>
            <a href="https://github.com/dhanoliya-ji/SoundTrace/blob/main/docs/HANDOFF.md" target="_blank" rel="noreferrer">Read project status ↗</a>
          </div>
        </section>
      </main>
    </div>
  );
}
