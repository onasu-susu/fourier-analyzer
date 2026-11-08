// 音の解析に使う計算（画面とは関係なく、数だけを扱う）
//   FFT・振幅スペクトル・ピーク探し・基音の推定（YIN）・音名・フーリエ級数の係数・WAV への書き出し
const DSP = (() => {
  // ---------- FFT（要素数が 2 のべき乗） ----------
  const plans = {};
  function plan(N) {
    if (plans[N]) return plans[N];
    const bits = Math.round(Math.log2(N));
    const rev = new Uint32Array(N);
    for (let i = 0; i < N; i++) {
      let r = 0;
      for (let b = 0, x = i; b < bits; b++, x >>= 1) r = (r << 1) | (x & 1);
      rev[i] = r;
    }
    const cos = new Float64Array(N / 2);
    const sin = new Float64Array(N / 2);
    for (let i = 0; i < N / 2; i++) {
      cos[i] = Math.cos((2 * Math.PI * i) / N);
      sin[i] = -Math.sin((2 * Math.PI * i) / N);
    }
    return (plans[N] = { rev, cos, sin });
  }

  // re, im をその場で書きかえる（時間 → 周波数）
  function fft(re, im) {
    const N = re.length;
    const { rev, cos, sin } = plan(N);
    for (let i = 0; i < N; i++) {
      const j = rev[i];
      if (j > i) {
        const tr = re[i]; re[i] = re[j]; re[j] = tr;
        const ti = im[i]; im[i] = im[j]; im[j] = ti;
      }
    }
    for (let size = 2; size <= N; size <<= 1) {
      const half = size >> 1;
      const step = N / size;
      for (let start = 0; start < N; start += size) {
        for (let k = 0, t = 0; k < half; k++, t += step) {
          const a = start + k;
          const b = a + half;
          const tr = re[b] * cos[t] - im[b] * sin[t];
          const ti = re[b] * sin[t] + im[b] * cos[t];
          re[b] = re[a] - tr;
          im[b] = im[a] - ti;
          re[a] += tr;
          im[a] += ti;
        }
      }
    }
  }

  const hanns = {};
  function hann(N) {
    if (hanns[N]) return hanns[N];
    const w = new Float64Array(N);
    for (let i = 0; i < N; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N);
    return (hanns[N] = w);
  }

  // 振幅スペクトル：samples の start から N 個（はみ出した分は 0）にハン窓をかけて FFT する。
  // 振幅 A の正弦波は、そのまま A（満点の 1.0 が 0 dB）と読めるように整えてある。長さ N/2 + 1。
  function spectrum(samples, start, N) {
    const re = new Float64Array(N);
    const im = new Float64Array(N);
    const w = hann(N);
    for (let i = 0; i < N; i++) {
      const j = start + i;
      re[i] = j >= 0 && j < samples.length ? samples[j] * w[i] : 0;
    }
    fft(re, im);
    const half = N / 2;
    const amp = new Float64Array(half + 1);
    for (let k = 1; k < half; k++) amp[k] = (4 * Math.hypot(re[k], im[k])) / N;
    amp[0] = (2 * Math.hypot(re[0], im[0])) / N;
    amp[half] = (2 * Math.hypot(re[half], im[half])) / N;
    return amp;
  }

  const toDb = (a) => 20 * Math.log10(Math.max(a, 1e-6)); // 小さすぎる値は -120 dB で止める

  // ---------- ピーク（山）探し ----------
  // 近くの nb 本ぶんの線より高い山だけを、強い順に返す（ハン窓の「すそ」の小さな山は数えない）
  function findPeaks(amp, fs, N, opts = {}) {
    const { relDb = -40, absDb = -90, nb = 4, minHz = 0, maxHz = Infinity, limit = 8 } = opts;
    const half = amp.length - 1;
    let max = 0;
    for (let k = 1; k < half; k++) if (amp[k] > max) max = amp[k];
    if (max <= 0) return [];
    const thr = Math.max(max * Math.pow(10, relDb / 20), Math.pow(10, absDb / 20));
    const k0 = Math.max(nb, Math.ceil((minHz * N) / fs));
    const k1 = Math.min(half - nb, Math.floor((maxHz * N) / fs));
    const peaks = [];
    for (let k = k0; k <= k1; k++) {
      const v = amp[k];
      if (v < thr) continue;
      let ok = true;
      for (let j = 1; j <= nb; j++) {
        if (amp[k - j] > v || amp[k + j] >= v) { ok = false; break; }
      }
      if (!ok) continue;
      // 山の頂点を、前後 1 本ずつを使った放物線（対数の振幅）で求める
      const a = Math.log(Math.max(amp[k - 1], 1e-12));
      const b = Math.log(v);
      const c = Math.log(Math.max(amp[k + 1], 1e-12));
      const den = a - 2 * b + c;
      const d = den !== 0 ? Math.max(-0.5, Math.min(0.5, (0.5 * (a - c)) / den)) : 0;
      const pa = Math.exp(b - 0.25 * (a - c) * d);
      peaks.push({ freq: ((k + d) * fs) / N, amp: pa, db: toDb(pa) });
    }
    peaks.sort((x, y) => y.amp - x.amp);
    return peaks.slice(0, limit);
  }

  // ---------- 基音の推定（YIN） ----------
  // samples の start から W 個ぶんを使い、「何サンプルずれると波形が重なるか」で周期を求める
  // 見つからなければ null（雑音など、くり返しのない音）
  function yin(samples, start, fs, opts = {}) {
    const { minHz = 40, maxHz = 2000, threshold = 0.15 } = opts;
    let W = opts.W || 2048;
    const tauMax = Math.floor(fs / minHz);
    const tauMin = Math.max(2, Math.floor(fs / maxHz));
    start = Math.max(0, Math.min(start, samples.length - (W + tauMax)));
    W = Math.min(W, samples.length - start - tauMax);
    if (W < 512 || start < 0) return null;
    const d = new Float64Array(tauMax + 1);
    for (let tau = 1; tau <= tauMax; tau++) {
      let s = 0;
      for (let j = 0; j < W; j++) {
        const x = samples[start + j] - samples[start + j + tau];
        s += x * x;
      }
      d[tau] = s;
    }
    // 差の累積平均で正規化
    const cm = new Float64Array(tauMax + 1);
    cm[0] = 1;
    let run = 0;
    for (let tau = 1; tau <= tauMax; tau++) {
      run += d[tau];
      cm[tau] = run > 0 ? (d[tau] * tau) / run : 1;
    }
    let tau = -1;
    for (let t = tauMin; t < tauMax; t++) {
      if (cm[t] < threshold) {
        while (t + 1 < tauMax && cm[t + 1] < cm[t]) t++;
        tau = t;
        break;
      }
    }
    if (tau < 0) return null;
    // 谷の底を放物線で求める
    let t = tau;
    if (tau > 1 && tau < tauMax) {
      const a = cm[tau - 1], b = cm[tau], c = cm[tau + 1];
      const den = a - 2 * b + c;
      if (den !== 0) t = tau + (0.5 * (a - c)) / den;
    }
    return { f0: fs / t, clarity: 1 - cm[tau] };
  }

  // ---------- 音名 ----------
  const NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
  const JA = ["ド", "ド♯", "レ", "レ♯", "ミ", "ファ", "ファ♯", "ソ", "ソ♯", "ラ", "ラ♯", "シ"];
  // 周波数 → 音名（A4 = 440 Hz）と、そこからのずれ（セント。100 セントで半音）
  function note(f) {
    const midi = 69 + 12 * Math.log2(f / 440);
    const m = Math.round(midi);
    const idx = ((m % 12) + 12) % 12;
    const octave = Math.floor(m / 12) - 1;
    return { name: NAMES[idx] + octave, ja: JA[idx], cents: Math.round((midi - m) * 100) };
  }

  // ---------- フーリエ級数の係数 ----------
  // 基音 f0 の周期の整数倍（最大 maxPeriods 周期）ぶんを使って
  //   x(t) ≈ a0/2 + Σ { a_k cos(2πk f0 t) + b_k sin(2πk f0 t) }   （t は start の位置から測る）
  // の a0・a_k・b_k（k = 1..n）を求める。周期 1 つぶんにも足りなければ null
  function fourierCoefficients(samples, start, fs, f0, n, maxPeriods = 16) {
    const avail = Math.floor(((samples.length - start) * f0) / fs);
    const M = Math.min(maxPeriods, avail);
    if (M < 1 || start < 0) return null;
    const exact = (M * fs) / f0; // M 周期ぶんのサンプル数（小数）
    const Ns = Math.floor(exact);
    const tail = exact - Ns; // 切れ目のはんぱな部分は、重みをつけて数える
    n = Math.min(n, Math.floor(fs / 2 / f0 - 1e-9));
    const weight = (i) => (i < Ns ? 1 : start + i < samples.length ? tail : 0);
    let sum = 0;
    for (let i = 0; i <= Ns; i++) sum += samples[start + i] * weight(i);
    const a = [0];
    const b = [0];
    for (let k = 1; k <= n; k++) {
      let ca = 0, cb = 0;
      const w = (2 * Math.PI * k * f0) / fs;
      for (let i = 0; i <= Ns; i++) {
        const x = samples[start + i] * weight(i);
        ca += x * Math.cos(w * i);
        cb += x * Math.sin(w * i);
      }
      a.push((2 * ca) / exact);
      b.push((2 * cb) / exact);
    }
    return { a0: (2 * sum) / exact, a, b, n, M, Ns };
  }

  // 係数の列（添字 1..n）を、フーリエ級数アプリに入れられる k の式にする。例：0.3*eq(k,1) - 0.2*eq(k,2)
  function seriesExpression(values, rel = 1e-3) {
    let top = 0;
    for (let k = 1; k < values.length; k++) top = Math.max(top, Math.abs(values[k]));
    const eps = Math.max(1e-6, top * rel); // いちばん大きい値の 0.1% より小さい項は、雑音なので省く
    const num = (v) => String(Number(Math.abs(v).toPrecision(5)));
    let out = "";
    for (let k = 1; k < values.length; k++) {
      const v = values[k];
      if (!(Math.abs(v) >= eps)) continue;
      const term = `${num(v)}*eq(k,${k})`;
      out += out ? (v < 0 ? " - " : " + ") + term : (v < 0 ? "-" : "") + term;
    }
    return out || "0";
  }

  // ---------- WAV（16 ビット・モノラル）に書き出す ----------
  function encodeWav(samples, fs) {
    const n = samples.length;
    const buf = new ArrayBuffer(44 + n * 2);
    const v = new DataView(buf);
    const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    str(0, "RIFF");
    v.setUint32(4, 36 + n * 2, true);
    str(8, "WAVE");
    str(12, "fmt ");
    v.setUint32(16, 16, true);
    v.setUint16(20, 1, true); // PCM
    v.setUint16(22, 1, true); // モノラル
    v.setUint32(24, fs, true);
    v.setUint32(28, fs * 2, true);
    v.setUint16(32, 2, true);
    v.setUint16(34, 16, true);
    str(36, "data");
    v.setUint32(40, n * 2, true);
    for (let i = 0; i < n; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Uint8Array(buf);
  }

  // いちばん音が大きいところの中心（秒）。録音のあと、最初に解析する位置に使う
  function loudestCenter(samples, fs, block = 2048) {
    if (samples.length < block) return samples.length / 2 / fs;
    let best = -1;
    let at = 0;
    for (let s = 0; s + block <= samples.length; s += block >> 1) {
      let e = 0;
      for (let i = 0; i < block; i++) e += samples[s + i] * samples[s + i];
      if (e > best) { best = e; at = s; }
    }
    return (at + block / 2) / fs;
  }

  // ---------- 基音の高さを、もっと細かく ----------
  // YIN で求めた f0 のまわり（±span）で、基音と倍音（第 1〜harmonics）の強さがいちばん大きくなる高さをさがす。
  // 長めの区間を使うので、録音の音がわずかに高い・低いときでも、周期の数がずれにくい
  function refineF0(samples, start, fs, f0, opts = {}) {
    const { harmonics = 3, span = 0.01, windowSec = 0.4 } = opts;
    const len = Math.min(Math.round(windowSec * fs), samples.length - start);
    if (len < fs / f0 * 4) return f0;
    const w = hann(len);
    const power = (f) => {
      let p = 0;
      for (let k = 1; k <= harmonics; k++) {
        if (k * f >= fs / 2) break;
        const om = (2 * Math.PI * k * f) / fs;
        let re = 0, im = 0;
        for (let i = 0; i < len; i++) {
          const x = samples[start + i] * w[i];
          re += x * Math.cos(om * i);
          im -= x * Math.sin(om * i);
        }
        p += re * re + im * im;
      }
      return p;
    };
    // 黄金分割法
    const g = (Math.sqrt(5) - 1) / 2;
    let lo = f0 * (1 - span), hi = f0 * (1 + span);
    let x1 = hi - g * (hi - lo), x2 = lo + g * (hi - lo);
    let p1 = power(x1), p2 = power(x2);
    for (let it = 0; it < 22; it++) {
      if (p1 < p2) { lo = x1; x1 = x2; p1 = p2; x2 = lo + g * (hi - lo); p2 = power(x2); }
      else { hi = x2; x2 = x1; p2 = p1; x1 = hi - g * (hi - lo); p1 = power(x1); }
    }
    return (lo + hi) / 2;
  }

  // 区間を parts 個に分けて、それぞれで refineF0 を行い、中央値をとる。
  // 録音のとぎれや位相の飛びが 1 か所あっても、ほかの区間の値が保たれるので、引きずられない
  function refineF0Robust(samples, start, fs, f0, opts = {}) {
    const { parts = 3, windowSec = 0.4, harmonics = 3, span = 0.01 } = opts;
    const total = Math.min(Math.round(windowSec * fs), samples.length - start);
    const len = Math.floor(total / parts);
    const vals = [];
    for (let i = 0; i < parts; i++) {
      vals.push(refineF0(samples, start + i * len, fs, f0, { harmonics, span, windowSec: len / fs }));
    }
    vals.sort((a, b) => a - b);
    return vals[Math.floor(parts / 2)];
  }

  return { fft, hann, spectrum, refineF0, refineF0Robust, toDb, findPeaks, yin, note, fourierCoefficients, seriesExpression, encodeWav, loudestCenter };
})();

if (typeof module !== "undefined") module.exports = DSP;
