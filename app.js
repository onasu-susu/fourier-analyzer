// 録音して、フーリエ解析
//   マイクで録音（または音声ファイルを開く）→ 波形・スペクトル・スペクトログラム・基音・フーリエ級数の係数
//   録音も計算もすべてこのページの中。どこにも送らない。計算は dsp.js。
const $ = (s) => document.querySelector(s);
const el = {
  rec: $("#rec"), play: $("#play"), save: $("#save"), file: $("#file"), level: $("#level"),
  status: $("#status"), msg: $("#msg"),
  wave: $("#wave"), zoom: $("#zoom"), spec: $("#spec"), gram: $("#gram"),
  fft: $("#fft"), fmax: $("#fmax"), fscale: $("#fscale"), yscale: $("#yscale"),
  cursor: $("#cursor"), result: $("#result"),
  coefPanel: $("#coefPanel"), coefNote: $("#coefNote"), coefTable: $("#coefTable"), toFourier: $("#toFourier"),
};
const MAX_SEC = 60; // 録音の長さの上限
const LIVE_N = 8192; // 録音中のリアルタイム表示に使う窓
const COEF_N = 12; // 係数をいくつ求めるか
const FOURIER_APP = "https://onasu-susu.github.io/fourier/";
const PAD = { l: 52, r: 12 }; // 3 つのグラフで横の余白をそろえる（時間の目盛りが縦にそろう）

// ---------- 状態 ----------
const S = {
  samples: null, // Float32Array（モノラル）
  fs: 48000,
  pos: 0, // 解析する位置の中心（秒）
  peak: 0,
  version: 0, // 音が入れかわるたびに増やす（描画の使いまわし用）
};
let G = null; // スペクトログラム
let lastSpec = null; // いま表示しているスペクトル { amp, N, fs, f0 }
let lastMap = null; // スペクトルの横軸（カーソルの読みとり用）
let lastResult = null; // テストやデバッグ用
let rec = null; // 録音中の情報
let audio = { ctx: null, workletReady: false, source: null };

// ---------- 小さな道具 ----------
const fmtHz = (f) => (f >= 1000 ? `${(f / 1000).toFixed(2)} kHz` : `${f.toFixed(f >= 100 ? 1 : 2)} Hz`);
const fmtDb = (d) => `${d <= -119 ? "−∞" : d.toFixed(1).replace("-", "−")} dB`;
const fmtNum = (v) => (v === 0 ? "0" : Math.abs(v) >= 1e-3 && Math.abs(v) < 1e5 ? String(Number(v.toPrecision(4))) : v.toExponential(2));
const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function note(text, error) {
  el.msg.textContent = text;
  el.msg.hidden = !text;
  el.msg.classList.toggle("error", !!error);
}

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = Math.round((canvas.getAttribute("height") / canvas.getAttribute("width")) * w);
  canvas.style.height = h + "px";
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const c = canvas.getContext("2d");
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, w, h);
  c.font = "12px 'Zen Maru Gothic', sans-serif";
  return { c, w, h };
}

function placeholder(c, w, h, text) {
  c.fillStyle = cssVar("--muted");
  c.textAlign = "center";
  c.textBaseline = "middle";
  c.fillText(text, w / 2, h / 2);
}

function niceCeil(v) {
  if (!(v > 0)) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  const m = v / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}

// 横軸（周波数）：直線は 0〜fmax、対数は 20 Hz〜fmax
function freqMap(pw, fmax, log) {
  const fmin = 20;
  const xOf = (f) => PAD.l + (log ? Math.log(Math.max(f, fmin) / fmin) / Math.log(fmax / fmin) : f / fmax) * pw;
  const fOf = (x) => {
    const t = clamp((x - PAD.l) / pw, 0, 1);
    return log ? fmin * Math.pow(fmax / fmin, t) : t * fmax;
  };
  let ticks;
  if (log) ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000].filter((f) => f <= fmax);
  else {
    const step = [50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000].find((s) => fmax / s <= 8) || 10000;
    ticks = [];
    for (let f = 0; f <= fmax + 1e-9; f += step) ticks.push(f);
  }
  return { xOf, fOf, ticks };
}
const tickLabel = (f) => (f >= 1000 ? `${f / 1000}k` : String(f));
const fmaxNow = () => Math.min(+el.fmax.value || S.fs / 2, S.fs / 2);
const isLog = () => el.fscale.value === "log";

// ---------- 音の読み込み・録音 ----------
function getCtx() {
  if (!audio.ctx) audio.ctx = new (window.AudioContext || window.webkitAudioContext)();
  return audio.ctx;
}

const PROCESSOR = `
class Rec extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(2048); this.n = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) for (let i = 0; i < ch.length; i++) {
      this.buf[this.n++] = ch[i];
      if (this.n === this.buf.length) { this.port.postMessage(this.buf.slice()); this.n = 0; }
    }
    return true;
  }
}
registerProcessor("rec", Rec);`;

async function ensureWorklet(ctx) {
  if (audio.workletReady) return;
  if (!ctx.audioWorklet) throw Object.assign(new Error("no worklet"), { code: "old" });
  const url = URL.createObjectURL(new Blob([PROCESSOR], { type: "text/javascript" }));
  await ctx.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);
  audio.workletReady = true;
}

function micError(e) {
  const n = e && e.name;
  if (e && e.code === "old") return "このブラウザでは録音できません（古いようです）。音声ファイルを開くことはできます。";
  if (n === "NotAllowedError" || n === "SecurityError") return "マイクの使用が許可されていません。アドレス欄の鍵（またはマイク）のマークから、このページのマイクを「許可」にして、もう一度押してください。";
  if (n === "NotFoundError" || n === "OverconstrainedError") return "マイクが見つかりません。マイクをつないで、もう一度ためしてください。";
  if (n === "NotReadableError") return "マイクを使えません。ほかのアプリがマイクを使っていないか確かめてください。";
  return "録音を始められませんでした。音声ファイルを開くことはできます。";
}

async function startRecording() {
  note("");
  stopPlay();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    note("このブラウザでは録音できません（https で開いているか、ブラウザが新しいか確かめてください）。音声ファイルを開くことはできます。", true);
    return;
  }
  const ctx = getCtx(); // iPhone などで許可されるよう、クリックの中で先に作る
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    });
    if (ctx.state === "suspended") await ctx.resume();
    await ensureWorklet(ctx);
  } catch (e) {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    note(micError(e), true);
    return;
  }
  const src = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "rec", { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1, channelCountMode: "explicit" });
  const mute = ctx.createGain(); // 音を出さずに、グラフだけ動かす
  mute.gain.value = 0;
  src.connect(node);
  node.connect(mute);
  mute.connect(ctx.destination);
  rec = { stream, src, node, mute, buf: new Float32Array(Math.round(MAX_SEC * ctx.sampleRate)), n: 0, peak: 0, hold: 0, last: 0, raf: 0, clipped: false };
  S.fs = ctx.sampleRate;
  G = null;
  node.port.onmessage = (e) => {
    if (!rec) return;
    const chunk = e.data;
    const take = Math.min(chunk.length, rec.buf.length - rec.n);
    rec.buf.set(chunk.subarray(0, take), rec.n);
    rec.n += take;
    let p = 0;
    for (let i = 0; i < take; i++) p = Math.max(p, Math.abs(chunk[i]));
    rec.peak = Math.max(rec.peak, p);
    rec.hold = Math.max(p, rec.hold * 0.9);
    if (p >= 0.99) rec.clipped = true;
    if (rec.n >= rec.buf.length) stopRecording();
  };
  el.rec.textContent = "■ 停止";
  el.rec.classList.add("on");
  el.play.disabled = true;
  el.save.disabled = true;
  el.wave.classList.remove("seek");
  el.gram.classList.remove("seek");
  el.coefPanel.hidden = true;
  drawZoom();
  drawGram();
  rec.raf = requestAnimationFrame(tickRecording);
}

function tickRecording(now) {
  if (!rec) return;
  const sec = rec.n / S.fs;
  el.level.style.width = `${Math.min(100, rec.hold * 100)}%`;
  el.level.classList.toggle("hot", rec.hold >= 0.99);
  el.status.textContent = `録音中… ${sec.toFixed(1)} 秒（最長 ${MAX_SEC} 秒）`;
  if (rec.clipped) note("音が大きすぎて割れています。マイクから少し離れるか、音を小さくしてください。", true);
  if (now - rec.last > 90) {
    rec.last = now;
    S.samples = rec.buf.subarray(0, rec.n);
    S.version++;
    S.peak = rec.peak;
    lastSpec = { amp: DSP.spectrum(rec.buf, rec.n - LIVE_N, LIVE_N), N: LIVE_N, fs: S.fs, f0: null, live: true };
    drawWave();
    drawSpectrum();
  }
  rec.raf = requestAnimationFrame(tickRecording);
}

function stopRecording() {
  if (!rec) return;
  const r = rec;
  rec = null;
  cancelAnimationFrame(r.raf);
  r.node.port.onmessage = null;
  r.src.disconnect();
  r.node.disconnect();
  r.mute.disconnect();
  r.stream.getTracks().forEach((t) => t.stop());
  el.rec.textContent = "● 録音";
  el.rec.classList.remove("on");
  el.level.style.width = "0";
  const clipped = r.clipped;
  if (r.n < S.fs * 0.2) {
    S.samples = null;
    note("録音が短すぎます（0.2 秒より長く録音してください）。", true);
    showStatus();
    redrawAll();
    return;
  }
  loaded(r.buf.slice(0, r.n), S.fs, clipped ? "音が大きすぎて割れているところがあります（マイクから少し離すと正確になります）。" : "");
}

// 音が入ったあとの共通の処理
function loaded(samples, fs, warning = "") {
  S.samples = samples;
  S.fs = fs;
  S.version++;
  let p = 0;
  for (let i = 0; i < samples.length; i++) p = Math.max(p, Math.abs(samples[i]));
  S.peak = p;
  S.pos = DSP.loudestCenter(samples, fs);
  fillFftOptions();
  computeGram();
  el.play.disabled = false;
  el.save.disabled = false;
  el.wave.classList.add("seek");
  el.gram.classList.add("seek");
  showStatus();
  note(warning, !!warning);
  analyze();
  redrawAll();
}

function showStatus() {
  if (!S.samples) {
    el.status.textContent = "マイクで録音するか、音声ファイルを開いてください。";
    return;
  }
  el.status.textContent = `${(S.samples.length / S.fs).toFixed(2)} 秒　／　標本化周波数 ${S.fs} Hz（${fmtHz(S.fs / 2)} までの音を調べられます）　／　最大の大きさ ${fmtDb(DSP.toDb(S.peak))}`;
}

el.rec.addEventListener("click", () => (rec ? stopRecording() : startRecording()));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { if (rec) stopRecording(); stopPlay(); }
});

el.file.addEventListener("change", async () => {
  const f = el.file.files[0];
  el.file.value = "";
  if (!f || rec) return;
  note("読みこんでいます…");
  stopPlay();
  try {
    const ctx = getCtx();
    const buf = await ctx.decodeAudioData(await f.arrayBuffer());
    const n = Math.min(buf.length, Math.round(MAX_SEC * buf.sampleRate));
    const mono = new Float32Array(n);
    for (let c = 0; c < buf.numberOfChannels; c++) {
      const ch = buf.getChannelData(c);
      for (let i = 0; i < n; i++) mono[i] += ch[i] / buf.numberOfChannels;
    }
    loaded(mono, buf.sampleRate, buf.length > n ? `長いので、最初の ${MAX_SEC} 秒だけを使っています。` : "");
  } catch (e) {
    note("このファイルは読みこめませんでした。WAV・MP3・M4A など、ふつうの音声ファイルを選んでください。", true);
  }
});

// ---------- 再生・保存 ----------
let playing = null;
function stopPlay() {
  if (!playing) return;
  const p = playing;
  playing = null;
  p.onended = null;
  try { p.stop(); } catch {}
  el.play.textContent = "▶ 再生";
}
el.play.addEventListener("click", async () => {
  if (playing) return stopPlay();
  if (!S.samples) return;
  const ctx = getCtx();
  if (ctx.state === "suspended") await ctx.resume();
  const buf = ctx.createBuffer(1, S.samples.length, S.fs);
  buf.copyToChannel(S.samples, 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  const g = ctx.createGain();
  g.gain.value = 0.8;
  src.connect(g);
  g.connect(ctx.destination);
  src.onended = () => { if (playing === src) { playing = null; el.play.textContent = "▶ 再生"; } };
  src.start();
  playing = src;
  el.play.textContent = "■ 止める";
});
el.save.addEventListener("click", () => {
  if (!S.samples) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([DSP.encodeWav(S.samples, S.fs)], { type: "audio/wav" }));
  a.download = "recording.wav";
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 60000);
});

// ---------- 解析 ----------
function fillFftOptions() {
  const keep = el.fft.value || "16384";
  el.fft.replaceChildren(...[2048, 4096, 8192, 16384, 32768].map((N) =>
    new Option(`${N} 個（${(N / S.fs).toFixed(2)} 秒・${(S.fs / N).toFixed(1)} Hz ごと）`, String(N), false, String(N) === keep)));
}

const CURSOR_HINT = el.cursor.textContent;
function analyze() {
  const x = S.samples;
  if (!x || !x.length) {
    lastSpec = null;
    lastResult = null;
    el.result.replaceChildren();
    el.coefPanel.hidden = true;
    drawSpectrum();
    drawZoom();
    return;
  }
  const fs = S.fs;
  const N = +el.fft.value;
  const amp = DSP.spectrum(x, Math.round(S.pos * fs - N / 2), N);
  const peaks = DSP.findPeaks(amp, fs, N, { relDb: -40, limit: 8 });

  // 基音：YIN で大まかに求めて、長めの区間で細かくする
  const W = 2048;
  const y = DSP.yin(x, Math.round(S.pos * fs - W / 2), fs, { W });
  const span = Math.round(0.4 * fs);
  const wStart = clamp(Math.round(S.pos * fs - span / 2), 0, Math.max(0, x.length - span));
  let f0 = null;
  let coef = null;
  let cStart = wStart;
  if (y) {
    f0 = DSP.refineF0Robust(x, wStart, fs, y.f0, { windowSec: 0.4 });
    // 係数は、解析する位置を中心にした 12 周期ぶんで求める（録音のはしに近いときは、はしに寄せる）
    const need = Math.round((12 * fs) / f0);
    cStart = clamp(Math.round(S.pos * fs - need / 2), 0, Math.max(0, x.length - need - 1));
    coef = DSP.fourierCoefficients(x, cStart, fs, f0, COEF_N, 12);
  }
  lastSpec = { amp, N, fs, f0 };
  lastResult = { f0, yin: y, peaks, coef, wStart: cStart };
  el.cursor.textContent = CURSOR_HINT;
  renderResult(amp, peaks, f0);
  renderCoef(coef, f0, cStart);
  drawSpectrum();
  drawZoom();
}

function harmonicLabel(f, f0) {
  if (!f0) return "";
  const r = f / f0;
  const k = Math.round(r);
  if (k >= 1 && Math.abs(r - k) < 0.04 * k) return k === 1 ? "基音" : `${k} 倍音`;
  return "";
}

function renderResult(amp, peaks, f0) {
  const quiet = !peaks.length || peaks[0].db < -80;
  let html = '<div class="stats">';
  if (f0) {
    const n = DSP.note(f0);
    html += `<div class="stat"><span>基音（推定）</span><b>${fmtHz(f0)}</b><small>${n.name}（${n.ja}）${n.cents >= 0 ? "+" : "−"}${Math.abs(n.cents)} セント</small></div>`;
  } else {
    html += `<div class="stat"><span>基音（推定）</span><b>見つかりません</b><small>${quiet ? "ほぼ無音です" : "雑音や、高さが定まらない音です"}</small></div>`;
  }
  if (peaks.length && !quiet) {
    const top = peaks[0];
    const n = DSP.note(top.freq);
    html += `<div class="stat"><span>いちばん強い成分</span><b>${fmtHz(top.freq)}</b><small>${fmtDb(top.db)}　${n.name}（${n.ja}）</small></div>`;
  } else {
    html += '<div class="stat"><span>いちばん強い成分</span><b>—</b><small>音が小さすぎます</small></div>';
  }
  html += "</div>";
  if (peaks.length && !quiet) {
    html += "<h3>目立つ成分（強い順）</h3>";
    html += '<div class="table-wrap"><table><tr><th>周波数</th><th>音名</th><th>強さ</th><th></th></tr>';
    for (const p of peaks) {
      const n = DSP.note(p.freq);
      html += `<tr><td>${fmtHz(p.freq)}</td><td>${n.name}</td><td>${fmtDb(p.db)}</td><td>${harmonicLabel(p.freq, f0)}</td></tr>`;
    }
    html += "</table></div>";
  }
  el.result.innerHTML = html;
}

function renderCoef(coef, f0, wStart) {
  if (!coef || !f0) {
    el.coefPanel.hidden = true;
    return;
  }
  el.coefPanel.hidden = false;
  el.coefNote.textContent =
    `基音 f₀ = ${fmtHz(f0)}（周期 ${(1000 / f0).toFixed(3)} ミリ秒）の ${coef.M} 周期ぶん（${(wStart / S.fs).toFixed(2)} 秒から ${((wStart + coef.Ns) / S.fs).toFixed(2)} 秒）を使って、` +
    `x(t) ≈ a₀/2 + Σ { aₖ cos(2πk f₀ t) + bₖ sin(2πk f₀ t) } の係数を求めました。`;
  const A1 = Math.hypot(coef.a[1], coef.b[1]) || 1;
  // 基音の大きさの 0.2% より小さい値は、雑音なので「≈ 0」と書く
  const fmtCoef = (v) => (Math.abs(v) < 0.002 * A1 ? "≈ 0" : fmtNum(v));
  let html = `<tr><th>k</th><th>aₖ</th><th>bₖ</th><th>振幅 √(aₖ²+bₖ²)</th><th>基音との比</th></tr>`;
  html += `<tr><td>0（a₀）</td><td>${fmtCoef(coef.a0)}</td><td>—</td><td>—</td><td>—</td></tr>`;
  for (let k = 1; k <= coef.n; k++) {
    const A = Math.hypot(coef.a[k], coef.b[k]);
    html += `<tr><td>${k}</td><td>${fmtCoef(coef.a[k])}</td><td>${fmtCoef(coef.b[k])}</td><td>${fmtCoef(A)}</td><td>${((A / A1) * 100).toFixed(1)} %</td></tr>`;
  }
  el.coefTable.innerHTML = html;
  // 前につくった「フーリエ級数で音をつくる」アプリに、この係数を渡す
  const q = new URLSearchParams({
    a0: String(Number(coef.a0.toPrecision(5))),
    ak: DSP.seriesExpression(coef.a),
    bk: DSP.seriesExpression(coef.b),
    L: `1/(2*${f0.toFixed(3)})`,
    n: String(coef.n),
  });
  el.toFourier.href = `${FOURIER_APP}?${q}`;
}

// ---------- スペクトログラム ----------
function computeGram() {
  const x = S.samples;
  const fs = S.fs;
  const N = 2048;
  const bins = N / 2 + 1;
  const cols = Math.max(1, Math.min(900, Math.floor(x.length / 256)));
  const hop = x.length / cols;
  const db = new Float32Array(cols * bins);
  let maxDb = -120;
  for (let c = 0; c < cols; c++) {
    const amp = DSP.spectrum(x, Math.round(c * hop + hop / 2 - N / 2), N);
    for (let k = 0; k < bins; k++) {
      const v = DSP.toDb(amp[k]);
      db[c * bins + k] = v;
      if (k >= 1 && v > maxDb) maxDb = v;
    }
  }
  G = { cols, bins, db, maxDb, N, fs, version: S.version, img: null, key: "" };
}

const STOPS = [[0, [27, 21, 53]], [0.25, [91, 42, 134]], [0.5, [196, 62, 127]], [0.75, [255, 138, 61]], [1, [255, 233, 168]]];
const LUT = (() => {
  const out = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let j = 1;
    while (j < STOPS.length - 1 && t > STOPS[j][0]) j++;
    const [t0, c0] = STOPS[j - 1];
    const [t1, c1] = STOPS[j];
    const u = (t - t0) / (t1 - t0);
    for (let k = 0; k < 3; k++) out[i * 3 + k] = c0[k] + (c1[k] - c0[k]) * u;
  }
  return out;
})();

// ---------- 描く ----------
function drawAxisFreq(c, map, h, pad, grid) {
  c.textAlign = "center";
  c.textBaseline = "top";
  c.lineWidth = 1;
  for (const f of map.ticks) {
    const x = map.xOf(f);
    if (grid) {
      c.strokeStyle = cssVar("--line");
      c.beginPath();
      c.moveTo(x, pad.t);
      c.lineTo(x, h - pad.b);
      c.stroke();
    }
    c.fillStyle = cssVar("--muted");
    c.fillText(tickLabel(f), x, h - pad.b + 6);
  }
}

function drawSpectrum() {
  const { c, w, h } = setupCanvas(el.spec);
  const pad = { l: PAD.l, r: PAD.r, t: 12, b: 38 };
  const pw = w - pad.l - pad.r;
  const ph = h - pad.t - pad.b;
  const sp = lastSpec;
  const fmax = fmaxNow();
  const map = freqMap(pw, fmax, isLog());
  lastMap = map;
  drawAxisFreq(c, map, h, pad, true);
  c.fillStyle = cssVar("--muted");
  c.textAlign = "center";
  c.textBaseline = "top";
  c.fillText("周波数（Hz）", pad.l + pw / 2, h - 16);
  if (!sp) return placeholder(c, w, h, "まだ音がありません。録音するか、音声ファイルを開いてください。");

  const kmax = Math.min(sp.amp.length - 1, Math.floor((fmax * sp.N) / sp.fs));
  let max = 0;
  for (let k = 1; k <= kmax; k++) if (sp.amp[k] > max) max = sp.amp[k];
  const linY = el.yscale.value === "lin";
  let Y;
  c.textAlign = "right";
  c.textBaseline = "middle";
  c.strokeStyle = cssVar("--line");
  if (linY) {
    const top = niceCeil(max * 1.1 || 1e-3);
    Y = (a) => pad.t + (1 - Math.min(a, top) / top) * ph;
    for (let i = 0; i <= 4; i++) {
      const v = (top * i) / 4;
      c.beginPath(); c.moveTo(pad.l, Y(v)); c.lineTo(w - pad.r, Y(v)); c.stroke();
      c.fillText(fmtNum(v), pad.l - 6, Y(v));
    }
  } else {
    const maxDb = max > 0 ? DSP.toDb(max) : -90;
    const top = Math.ceil((maxDb + 3) / 10) * 10;
    const bottom = top - 90;
    Y = (a) => pad.t + (1 - (clamp(DSP.toDb(a), bottom, top) - bottom) / (top - bottom)) * ph;
    for (let d = bottom; d <= top; d += 10) {
      if ((top - d) % 20) continue;
      const yy = pad.t + (1 - (d - bottom) / (top - bottom)) * ph;
      c.beginPath(); c.moveTo(pad.l, yy); c.lineTo(w - pad.r, yy); c.stroke();
      c.fillText(String(d).replace("-", "−"), pad.l - 6, yy);
    }
  }
  // 基音と倍音の位置
  if (sp.f0) {
    c.save();
    c.strokeStyle = cssVar("--accent-2");
    for (let k = 1; k <= 40 && k * sp.f0 <= fmax; k++) {
      const x = map.xOf(k * sp.f0);
      c.globalAlpha = k === 1 ? 0.9 : 0.4;
      c.setLineDash(k === 1 ? [5, 4] : [2, 4]);
      c.lineWidth = k === 1 ? 1.5 : 1;
      c.beginPath(); c.moveTo(x, pad.t); c.lineTo(x, pad.t + ph); c.stroke();
    }
    c.restore();
  }
  // スペクトルの線
  const base = pad.t + ph;
  c.beginPath();
  let started = false;
  for (let k = 1; k <= kmax; k++) {
    const x = map.xOf((k * sp.fs) / sp.N);
    const y = Y(sp.amp[k]);
    if (!started) { c.moveTo(x, base); c.lineTo(x, y); started = true; } else c.lineTo(x, y);
  }
  if (started) {
    const last = map.xOf((kmax * sp.fs) / sp.N);
    c.lineTo(last, base);
    c.closePath();
    c.fillStyle = cssVar("--accent");
    c.globalAlpha = 0.18;
    c.fill();
    c.globalAlpha = 1;
  }
  c.strokeStyle = cssVar("--accent");
  c.lineWidth = 1.6;
  c.lineJoin = "round";
  c.beginPath();
  started = false;
  for (let k = 1; k <= kmax; k++) {
    const x = map.xOf((k * sp.fs) / sp.N);
    const y = Y(sp.amp[k]);
    if (!started) { c.moveTo(x, y); started = true; } else c.lineTo(x, y);
  }
  c.stroke();
  c.fillStyle = cssVar("--muted");
  c.textAlign = "left";
  c.textBaseline = "top";
  c.fillText(linY ? "振幅" : "dB", 6, 0);
  if (sp.f0) {
    c.fillStyle = cssVar("--accent-2");
    c.textAlign = "left";
    c.fillText("基音と倍音", map.xOf(sp.f0) + 4, pad.t + 2);
  }
  if (sp.live) {
    c.fillStyle = cssVar("--accent");
    c.textAlign = "right";
    c.fillText("● リアルタイム", w - pad.r - 4, pad.t + 2);
  }
}

// 波形：1 画素ぶんの最小・最大だけ描く（長い録音でも軽い）
let waveCache = null;
function drawWave() {
  const { c, w, h } = setupCanvas(el.wave);
  const pad = { l: PAD.l, r: PAD.r, t: 10, b: 24 };
  const pw = w - pad.l - pad.r;
  const ph = h - pad.t - pad.b;
  const x = S.samples;
  c.strokeStyle = cssVar("--line");
  c.lineWidth = 1;
  c.beginPath(); c.moveTo(pad.l, pad.t + ph / 2); c.lineTo(w - pad.r, pad.t + ph / 2); c.stroke();
  if (!x || !x.length) return placeholder(c, w, h, "まだ音がありません。");
  const total = x.length;
  const span = rec ? Math.max(total, Math.round(5 * S.fs)) : total; // 録音中は 5 秒ぶんの幅から右へのびていく
  const px = Math.floor(pw);
  const key = `${S.version}|${px}|${span}`;
  if (!waveCache || waveCache.key !== key) {
    const mins = new Float32Array(px);
    const maxs = new Float32Array(px);
    for (let i = 0; i < px; i++) {
      const a = Math.floor((i * span) / px);
      const b = Math.min(total, Math.max(a + 1, Math.floor(((i + 1) * span) / px)));
      let lo = Infinity, hi = -Infinity;
      for (let j = a; j < b; j++) { const v = x[j]; if (v < lo) lo = v; if (v > hi) hi = v; }
      mins[i] = a < total ? lo : 0;
      maxs[i] = a < total ? hi : 0;
    }
    waveCache = { key, mins, maxs };
  }
  const top = Math.max(S.peak * 1.1, 0.01);
  const Y = (v) => pad.t + (1 - (v / top + 1) / 2) * ph;
  c.fillStyle = cssVar("--muted");
  c.textAlign = "right";
  c.textBaseline = "middle";
  c.fillText(fmtNum(top), pad.l - 6, pad.t);
  c.fillText("0", pad.l - 6, pad.t + ph / 2);
  c.fillText("−" + fmtNum(top), pad.l - 6, pad.t + ph);
  // 時間の目盛り
  const sec = span / S.fs;
  const step = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30].find((s) => sec / s <= 8) || 30;
  c.textAlign = "center";
  c.textBaseline = "top";
  for (let t = 0; t <= sec + 1e-9; t += step) {
    const xx = pad.l + (t / sec) * pw;
    c.strokeStyle = cssVar("--line");
    c.beginPath(); c.moveTo(xx, pad.t); c.lineTo(xx, pad.t + ph); c.stroke();
    c.fillStyle = cssVar("--muted");
    c.fillText(`${Number(t.toFixed(2))} s`, xx, h - pad.b + 6);
  }
  // 解析している範囲
  if (!rec && lastSpec && !lastSpec.live) {
    const half = lastSpec.N / 2 / S.fs;
    const x0 = pad.l + clamp((S.pos - half) / sec, 0, 1) * pw;
    const x1 = pad.l + clamp((S.pos + half) / sec, 0, 1) * pw;
    c.fillStyle = cssVar("--accent-2");
    c.globalAlpha = 0.25;
    c.fillRect(x0, pad.t, Math.max(2, x1 - x0), ph);
    c.globalAlpha = 1;
    c.strokeStyle = cssVar("--accent-2");
    c.lineWidth = 2;
    const xc = pad.l + clamp(S.pos / sec, 0, 1) * pw;
    c.beginPath(); c.moveTo(xc, pad.t); c.lineTo(xc, pad.t + ph); c.stroke();
  }
  c.strokeStyle = cssVar("--accent");
  c.lineWidth = 1;
  c.beginPath();
  for (let i = 0; i < px; i++) {
    const xx = pad.l + i + 0.5;
    c.moveTo(xx, Y(waveCache.mins[i]));
    c.lineTo(xx, Math.min(Y(waveCache.mins[i]) - 1, Y(waveCache.maxs[i])));
  }
  c.stroke();
}

// 解析している位置の拡大：基音の 5 周期ぶん（基音がないときは 20 ミリ秒）
function drawZoom() {
  const { c, w, h } = setupCanvas(el.zoom);
  const pad = { l: PAD.l, r: PAD.r, t: 10, b: 24 };
  const pw = w - pad.l - pad.r;
  const ph = h - pad.t - pad.b;
  const x = S.samples;
  c.strokeStyle = cssVar("--line");
  c.lineWidth = 1;
  c.beginPath(); c.moveTo(pad.l, pad.t + ph / 2); c.lineTo(w - pad.r, pad.t + ph / 2); c.stroke();
  if (!x || !x.length || rec) return placeholder(c, w, h, rec ? "録音が終わると、ここに出ます。" : "まだ音がありません。");
  const f0 = lastResult && lastResult.f0;
  const span = f0 ? 5 / f0 : 0.02; // 秒
  const n = Math.max(2, Math.round(span * S.fs));
  const s0 = clamp(Math.round(S.pos * S.fs - n / 2), 0, Math.max(0, x.length - n));
  const m = Math.min(n, x.length - s0);
  let peak = 0;
  for (let i = 0; i < m; i++) peak = Math.max(peak, Math.abs(x[s0 + i]));
  const top = Math.max(peak * 1.1, 0.005);
  const X = (i) => pad.l + (i / (n - 1)) * pw;
  const Y = (v) => pad.t + (1 - (v / top + 1) / 2) * ph;
  c.fillStyle = cssVar("--muted");
  c.textAlign = "right";
  c.textBaseline = "middle";
  c.fillText(fmtNum(top), pad.l - 6, pad.t);
  c.fillText("0", pad.l - 6, pad.t + ph / 2);
  c.fillText("−" + fmtNum(top), pad.l - 6, pad.t + ph);
  // 時間の目盛り（ミリ秒）
  const ms = span * 1000;
  const step = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100].find((s) => ms / s <= 8) || 100;
  c.textAlign = "center";
  c.textBaseline = "top";
  for (let t = 0; t <= ms + 1e-9; t += step) {
    const xx = pad.l + (t / ms) * pw;
    c.strokeStyle = cssVar("--line");
    c.beginPath(); c.moveTo(xx, pad.t); c.lineTo(xx, pad.t + ph); c.stroke();
    c.fillStyle = cssVar("--muted");
    c.fillText(`${Number(t.toFixed(2))} ms`, xx, h - pad.b + 6);
  }
  // 基音の周期の区切り
  if (f0) {
    c.save();
    c.strokeStyle = cssVar("--accent-2");
    c.lineWidth = 1.5;
    c.setLineDash([5, 4]);
    for (let k = 0; k <= 5; k++) {
      const xx = pad.l + (k / 5) * pw;
      c.beginPath(); c.moveTo(xx, pad.t); c.lineTo(xx, pad.t + ph); c.stroke();
    }
    c.restore();
    c.fillStyle = cssVar("--accent-2");
    c.textAlign = "left";
    c.textBaseline = "top";
    c.fillText(`1 周期 = ${(1000 / f0).toFixed(3)} ms`, pad.l + 4, pad.t + 2);
  }
  c.strokeStyle = cssVar("--accent");
  c.lineWidth = 2;
  c.lineJoin = "round";
  c.beginPath();
  for (let i = 0; i < m; i++) (i ? c.lineTo(X(i), Y(x[s0 + i])) : c.moveTo(X(i), Y(x[s0 + i])));
  c.stroke();
}

function drawGram() {
  const { c, w, h } = setupCanvas(el.gram);
  const pad = { l: PAD.l, r: PAD.r, t: 8, b: 38 };
  const pw = w - pad.l - pad.r;
  const ph = h - pad.t - pad.b;
  const fmax = fmaxNow();
  const log = isLog();
  const map = freqMap(ph, fmax, log); // 縦軸の目盛り用（横方向の長さ = ph）
  if (!G || !S.samples || rec) return placeholder(c, w, h, rec ? "録音が終わると、ここに出ます。" : "まだ音がありません。");
  // 画像（時間 × 周波数）を作って使いまわす
  const key = `${Math.floor(pw)}|${Math.floor(ph)}|${fmax}|${log}|${G.version}`;
  if (G.key !== key) {
    const W = Math.floor(pw), H = Math.floor(ph);
    const off = document.createElement("canvas");
    off.width = W; off.height = H;
    const octx = off.getContext("2d");
    const img = octx.createImageData(W, H);
    const lo = G.maxDb - 80;
    for (let y = 0; y < H; y++) {
      const t = 1 - y / (H - 1 || 1);
      const f = log ? 20 * Math.pow(fmax / 20, t) : t * fmax;
      const bin = clamp(Math.round((f * G.N) / G.fs), 0, G.bins - 1);
      for (let x = 0; x < W; x++) {
        const col = Math.min(G.cols - 1, Math.floor((x / W) * G.cols));
        const v = clamp((G.db[col * G.bins + bin] - lo) / 80, 0, 1);
        const i = Math.round(v * 255) * 3;
        const o = (y * W + x) * 4;
        img.data[o] = LUT[i]; img.data[o + 1] = LUT[i + 1]; img.data[o + 2] = LUT[i + 2]; img.data[o + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    G.img = off;
    G.key = key;
  }
  c.imageSmoothingEnabled = false;
  c.drawImage(G.img, pad.l, pad.t, Math.floor(pw), Math.floor(ph));
  // 縦軸（周波数）
  c.fillStyle = cssVar("--muted");
  c.textAlign = "right";
  c.textBaseline = "middle";
  for (const f of map.ticks) {
    const t = log ? Math.log(Math.max(f, 20) / 20) / Math.log(fmax / 20) : f / fmax;
    c.fillText(tickLabel(f), pad.l - 6, pad.t + (1 - t) * ph);
  }
  // 横軸（時間）
  const sec = S.samples.length / S.fs;
  const step = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30].find((s) => sec / s <= 8) || 30;
  c.textAlign = "center";
  c.textBaseline = "top";
  for (let t = 0; t <= sec + 1e-9; t += step) c.fillText(`${Number(t.toFixed(2))} s`, pad.l + (t / sec) * pw, h - pad.b + 6);
  c.fillText("時間", pad.l + pw / 2, h - 16);
  // 解析している位置
  const xc = pad.l + clamp(S.pos / sec, 0, 1) * pw;
  c.strokeStyle = "#ffffff";
  c.lineWidth = 2;
  c.beginPath(); c.moveTo(xc, pad.t); c.lineTo(xc, pad.t + ph); c.stroke();
}

function redrawAll() {
  drawWave();
  drawZoom();
  drawSpectrum();
  drawGram();
}

// ---------- 位置を選ぶ・カーソルで読む ----------
function bindSeek(canvas) {
  let down = false;
  const set = (e) => {
    if (!S.samples || rec) return;
    const r = canvas.getBoundingClientRect();
    const pw = r.width - PAD.l - PAD.r;
    S.pos = clamp(((e.clientX - r.left - PAD.l) / pw) * (S.samples.length / S.fs), 0, S.samples.length / S.fs);
    drawWave();
    drawGram();
    clearTimeout(set.t);
    set.t = setTimeout(analyze, 40);
  };
  canvas.addEventListener("pointerdown", (e) => { down = true; canvas.setPointerCapture(e.pointerId); set(e); });
  canvas.addEventListener("pointermove", (e) => { if (down) set(e); });
  const up = () => { down = false; };
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);
}
bindSeek(el.wave);
bindSeek(el.gram);

function readCursor(e) {
  const sp = lastSpec;
  if (!sp || !lastMap) return;
  const r = el.spec.getBoundingClientRect();
  const f = lastMap.fOf(e.clientX - r.left);
  const k = clamp(Math.round((f * sp.N) / sp.fs), 0, sp.amp.length - 1);
  const n = f >= 20 ? DSP.note(f) : null;
  el.cursor.textContent = `${fmtHz(f)}　${fmtDb(DSP.toDb(sp.amp[k]))}${n ? `　${n.name}（${n.ja}）` : ""}`;
}
el.spec.addEventListener("pointermove", readCursor);
el.spec.addEventListener("pointerdown", readCursor);

[el.fft, el.fmax, el.fscale, el.yscale].forEach((s) =>
  s.addEventListener("change", () => {
    if (s === el.fft) analyze();
    redrawAll();
  }));
let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(redrawAll, 100); });
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawAll);

// テストやデバッグ用
window.analyzer = { state: S, result: () => lastResult, analyze };

fillFftOptions();
redrawAll();
