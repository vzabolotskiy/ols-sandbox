// ===================== Shared statistics + plotting library =====================
const Lib = (() => {
  const MAXN = 1000;

  // ---------- random numbers (seeded, so sliders don't reshuffle the data) ----------
  function uniform(seed) {
    let a = seed >>> 0;
    return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  function normals(seed, n, m) {
    const u = uniform(seed), out = Array.from({ length: m }, () => new Float64Array(n));
    for (let j = 0; j < m; j++) for (let i = 0; i < n; i++) {
      let x = 0; while (x === 0) x = u();
      out[j][i] = Math.sqrt(-2 * Math.log(x)) * Math.cos(2 * Math.PI * u());
    }
    return out;
  }
  const nextSeed = s => (Math.imul(s, 1103515245) + 12345) >>> 0;

  // ---------- distributions ----------
  function lgamma(x) {
    const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x, t = x + 5.5; t -= (x + 0.5) * Math.log(t);
    let s = 1.000000000190015; for (const k of c) s += k / ++y;
    return -t + Math.log(2.5066282746310005 * s / x);
  }
  function betacf(a, b, x) {
    const T = 1e-300, qab = a + b, qap = a + 1, qam = a - 1; let c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < T) d = T; d = 1 / d; let h = d;
    for (let m = 1; m <= 300; m++) {
      const m2 = 2 * m; let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < T) d = T; c = 1 + aa / c; if (Math.abs(c) < T) c = T; d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < T) d = T; c = 1 + aa / c; if (Math.abs(c) < T) c = T; d = 1 / d;
      const del = d * c; h *= del; if (Math.abs(del - 1) < 3e-14) break;
    }
    return h;
  }
  function betai(a, b, x) {
    if (x <= 0) return 0; if (x >= 1) return 1;
    const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
  }
  function gammaQ(a, x) {               // upper regularized incomplete gamma
    if (x <= 0) return 1;
    const gln = lgamma(a);
    if (x < a + 1) {
      let ap = a, sum = 1 / a, del = sum;
      for (let n = 0; n < 500; n++) { ap++; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-15) break; }
      return 1 - sum * Math.exp(-x + a * Math.log(x) - gln);
    }
    const T = 1e-300; let b = x + 1 - a, c = 1 / T, d = 1 / b, h = d;
    for (let i = 1; i < 500; i++) {
      const an = -i * (i - a); b += 2; d = an * d + b; if (Math.abs(d) < T) d = T; c = b + an / c; if (Math.abs(c) < T) c = T;
      d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-15) break;
    }
    return Math.exp(-x + a * Math.log(x) - gln) * h;
  }
  const pT = (t, df) => betai(df / 2, 0.5, df / (df + t * t));                    // two-sided p
  const pF = (F, d1, d2) => !(F > 0) ? 1 : betai(d2 / 2, d1 / 2, d2 / (d2 + d1 * F)); // upper tail
  const pChi = (x, k) => gammaQ(k / 2, x / 2);                                      // upper tail
  function bisect(f, lo, hi) { for (let i = 0; i < 80; i++) { const m = (lo + hi) / 2; f(m) ? lo = m : hi = m; } return (lo + hi) / 2; }
  const tcrit = (df, alpha) => bisect(m => pT(m, df) > alpha, 0, 1000);
  const Fcrit = (d1, d2, alpha) => bisect(m => pF(m, d1, d2) > alpha, 0, 1e4);
  const chiCrit = (k, alpha) => bisect(m => pChi(m, k) > alpha, 0, 1e4);
  const tpdf = (t, df) => Math.exp(lgamma((df + 1) / 2) - lgamma(df / 2) - 0.5 * Math.log(df * Math.PI) - (df + 1) / 2 * Math.log(1 + t * t / df));
  const Fpdf = (x, d1, d2) => x <= 0 ? 0 : Math.exp((d1 / 2) * Math.log(d1 / d2) + (d1 / 2 - 1) * Math.log(x) - ((d1 + d2) / 2) * Math.log(1 + d1 * x / d2) - (lgamma(d1 / 2) + lgamma(d2 / 2) - lgamma((d1 + d2) / 2)));

  // ---------- linear algebra + OLS ----------
  function inv(A) {
    const k = A.length, M = A.map((r, i) => [...r, ...Array.from({ length: k }, (_, j) => +(i === j))]);
    for (let c = 0; c < k; c++) {
      let p = c; for (let r = c + 1; r < k; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      const d = M[c][c]; if (Math.abs(d) < 1e-12 * (1 + Math.abs(A[c][c]))) return null;   // singular
      for (let j = 0; j < 2 * k; j++) M[c][j] /= d;
      for (let r = 0; r < k; r++) if (r !== c) { const f = M[r][c]; if (f) for (let j = 0; j < 2 * k; j++) M[r][j] -= f * M[c][j]; }
    }
    return M.map(r => r.slice(k));
  }
  // cols: array of regressor arrays (no constant); returns everything a Stata table needs
  function ols(cols, y, opt = {}) {
    const n = y.length, X = [new Float64Array(n).fill(1), ...cols], k = X.length;
    const XtX = X.map(a => X.map(b => { let s = 0; for (let i = 0; i < n; i++) s += a[i] * b[i]; return s; }));
    const Vi = inv(XtX); if (!Vi) return null;
    const Xty = X.map(a => { let s = 0; for (let i = 0; i < n; i++) s += a[i] * y[i]; return s; });
    const b = Vi.map(r => r.reduce((s, v, j) => s + v * Xty[j], 0));
    let ybar = 0; for (let i = 0; i < n; i++) ybar += y[i]; ybar /= n;
    const res = new Float64Array(n), fit = new Float64Array(n); let ssr = 0, sst = 0;
    for (let i = 0; i < n; i++) { let f = 0; for (let j = 0; j < k; j++) f += b[j] * X[j][i]; fit[i] = f; res[i] = y[i] - f; ssr += res[i] ** 2; sst += (y[i] - ybar) ** 2; }
    const df = n - k, s2 = ssr / df;
    const V = Vi.map(r => r.map(v => v * s2));
    const se = V.map((r, j) => Math.sqrt(r[j]));
    const t = b.map((v, j) => v / se[j]), p = t.map(v => pT(v, df));
    const r2 = 1 - ssr / sst, adj = 1 - (ssr / df) / (sst / (n - 1));
    const F = k > 1 ? ((sst - ssr) / (k - 1)) / s2 : NaN;
    const out = { b, se, t, p, V, df, n, k, ssr, sst, sse: sst - ssr, r2, adj, F, pF: pF(F, k - 1, df), rmse: Math.sqrt(s2), res, fit };
    if (opt.robust) {                                    // HC1 = Stata's vce(robust)
      const meat = X.map(() => new Float64Array(k));
      for (let i = 0; i < n; i++) { const e2 = res[i] * res[i]; for (let a = 0; a < k; a++) { const xa = X[a][i] * e2; for (let c = a; c < k; c++) meat[a][c] += xa * X[c][i]; } }
      for (let a = 0; a < k; a++) for (let c = 0; c < a; c++) meat[a][c] = meat[c][a];
      const VM = Vi.map(r => Array.from({ length: k }, (_, c) => r.reduce((s, v, j) => s + v * meat[j][c], 0)));
      const VR = VM.map(r => Array.from({ length: k }, (_, c) => r.reduce((s, v, j) => s + v * Vi[j][c], 0) * n / df));
      out.VR = VR; out.seR = VR.map((r, j) => Math.sqrt(r[j]));
      out.tR = b.map((v, j) => v / out.seR[j]); out.pR = out.tR.map(v => pT(v, df));
    }
    return out;
  }
  const mean = a => { let s = 0; for (const v of a) s += v; return s / a.length; };
  const sd = a => { const m = mean(a); let s = 0; for (const v of a) s += (v - m) ** 2; return Math.sqrt(s / (a.length - 1)); };
  const sst = a => { const m = mean(a); let s = 0; for (const v of a) s += (v - m) ** 2; return s; };
  const corr = (a, b) => { const ma = mean(a), mb = mean(b); let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < a.length; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; } return sab / Math.sqrt(saa * sbb); };

  // ---------- formatting ----------
  const f = (v, d = 3) => !isFinite(v) ? "·" : (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toFixed(d).replace("-", "−");
  const fp = v => !isFinite(v) ? "·" : v < 0.001 ? "<0.001" : v.toFixed(3);
  const sgn = (v, d = 1) => (v < 0 ? "− " : "+ ") + Math.abs(v).toFixed(d);
  const pct = v => (100 * v).toFixed(1) + "%";

  // ---------- canvas plotting ----------
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  function hex(h) { h = h.replace("#", ""); if (h.length === 3) h = h.split("").map(c => c + c).join(""); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); }
  function mix(a, b, t) { t = Math.max(0, Math.min(1, t)); const A = hex(a), B = hex(b); return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(",")})`; }
  function alpha(c, a) { const A = hex(c); return `rgba(${A.join(",")},${a})`; }
  function nice(lo, hi, n = 5) {
    const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0))), r = step0 / mag;
    const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag, out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9 * span; v += step) out.push(+v.toPrecision(12));
    return out;
  }
  const tick = v => { const s = Math.abs(v) >= 1e4 ? v.toFixed(0) : (+v.toPrecision(4)).toString(); return s.replace("-", "−"); };
  function range(arrs, pad = 0.06, extra = []) {
    let lo = Infinity, hi = -Infinity;
    for (const a of arrs) for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; }
    for (const v of extra) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const d = (hi - lo || 1) * pad; return [lo - d, hi + d];
  }
  function chart(cv, xr, yr, o = {}) {
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    cv.width = Math.max(1, Math.round(w * dpr)); cv.height = Math.max(1, Math.round(h * dpr));
    const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    const m = { l: o.noY ? 14 : 54, r: 14, t: 12, b: 38 };
    const X = v => m.l + (v - xr[0]) / (xr[1] - xr[0]) * (w - m.l - m.r);
    const Y = v => h - m.b - (v - yr[0]) / (yr[1] - yr[0]) * (h - m.t - m.b);
    const P = { ctx, w, h, m, X, Y, xr, yr };
    ctx.font = `11px ${css("--f-mono")}`; ctx.lineWidth = 1; ctx.strokeStyle = css("--line"); ctx.fillStyle = css("--muted");
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const v of (o.xticks || nice(xr[0], xr[1], o.nx || 5))) { if (v < xr[0] || v > xr[1]) continue; ctx.beginPath(); ctx.moveTo(X(v), m.t); ctx.lineTo(X(v), h - m.b); ctx.stroke(); ctx.fillText(o.xfmt ? o.xfmt(v) : tick(v), X(v), h - m.b + 5); }
    if (!o.noY) {
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      for (const v of nice(yr[0], yr[1], o.ny || 5)) { if (v < yr[0] || v > yr[1]) continue; ctx.beginPath(); ctx.moveTo(m.l, Y(v)); ctx.lineTo(w - m.r, Y(v)); ctx.stroke(); ctx.fillText(tick(v), m.l - 6, Y(v)); }
    }
    if (o.zeroY && yr[0] < 0 && yr[1] > 0) { ctx.strokeStyle = css("--muted"); ctx.beginPath(); ctx.moveTo(m.l, Y(0)); ctx.lineTo(w - m.r, Y(0)); ctx.stroke(); }
    ctx.font = `12px ${css("--f-body")}`; ctx.fillStyle = css("--muted"); ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    if (o.xlab) ctx.fillText(o.xlab, m.l + (w - m.l - m.r) / 2, h - 1);
    if (o.ylab && !o.noY) { ctx.save(); ctx.translate(12, m.t + (h - m.t - m.b) / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = "middle"; ctx.fillText(o.ylab, 0, 0); ctx.restore(); }
    return P;
  }
  function clip(P, fn) { const { ctx, m, w, h } = P; ctx.save(); ctx.beginPath(); ctx.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b); ctx.clip(); fn(ctx); ctx.restore(); }
  function dots(P, xs, ys, color, o = {}) {
    clip(P, ctx => {
      const r = o.r || (xs.length > 400 ? 2 : xs.length > 120 ? 2.6 : 3.2); ctx.globalAlpha = o.alpha ?? 0.75;
      for (let i = 0; i < xs.length; i++) { ctx.fillStyle = typeof color === "function" ? color(i) : color; ctx.beginPath(); ctx.arc(P.X(xs[i]), P.Y(ys[i]), r, 0, 7); ctx.fill(); }
      ctx.globalAlpha = 1;
    });
  }
  function curve(P, fn, color, o = {}) {
    clip(P, ctx => {
      ctx.strokeStyle = color; ctx.lineWidth = o.width || 2.5; ctx.setLineDash(o.dash || []); ctx.beginPath();
      const N = o.N || 160; for (let i = 0; i <= N; i++) { const x = P.xr[0] + (P.xr[1] - P.xr[0]) * i / N, y = fn(x); i ? ctx.lineTo(P.X(x), P.Y(y)) : ctx.moveTo(P.X(x), P.Y(y)); }
      ctx.stroke(); ctx.setLineDash([]);
    });
  }
  function band(P, lo, hi, color, o = {}) {           // lo/hi are functions of x
    clip(P, ctx => {
      const N = 120, xs = Array.from({ length: N + 1 }, (_, i) => P.xr[0] + (P.xr[1] - P.xr[0]) * i / N);
      ctx.fillStyle = color; ctx.beginPath();
      xs.forEach((x, i) => i ? ctx.lineTo(P.X(x), P.Y(hi(x))) : ctx.moveTo(P.X(x), P.Y(hi(x))));
      for (let i = N; i >= 0; i--) ctx.lineTo(P.X(xs[i]), P.Y(lo(xs[i])));
      ctx.closePath(); ctx.fill();
    });
  }
  function area(P, fn, a, b, color) {                 // fill under pdf between a and b
    a = Math.max(a, P.xr[0]); b = Math.min(b, P.xr[1]); if (b <= a) return;
    clip(P, ctx => {
      ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(P.X(a), P.Y(0));
      const N = 80; for (let i = 0; i <= N; i++) { const x = a + (b - a) * i / N; ctx.lineTo(P.X(x), P.Y(fn(x))); }
      ctx.lineTo(P.X(b), P.Y(0)); ctx.closePath(); ctx.fill();
    });
  }
  function vline(P, x, color, o = {}) {
    const { ctx, m, h } = P; if (x < P.xr[0] || x > P.xr[1]) return;
    ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = o.width || 2; ctx.setLineDash(o.dash || []);
    ctx.beginPath(); ctx.moveTo(P.X(x), m.t); ctx.lineTo(P.X(x), h - m.b); ctx.stroke(); ctx.restore();
    if (o.label) label(P, o.label, P.X(x), m.t + 2 + (o.row || 0) * 14, color, o.align);
  }
  function label(P, text, x, y, color, align) {
    const { ctx, w, m } = P; ctx.save(); ctx.font = `600 11.5px ${css("--f-body")}`; ctx.textBaseline = "top";
    const tw = ctx.measureText(text).width; let ax = align || "left", px = x + 5;
    if (ax === "left" && px + tw > w - m.r) { ax = "right"; }
    if (ax === "right") px = x - 5;
    ctx.textAlign = ax; ctx.fillStyle = css("--surface"); ctx.globalAlpha = 0.85;
    ctx.fillRect(ax === "left" ? px - 2 : px - tw - 2, y - 1, tw + 4, 15); ctx.globalAlpha = 1;
    ctx.fillStyle = color; ctx.fillText(text, px, y); ctx.restore();
  }
  function hist(P, vals, bins, color, o = {}) {
    const [lo, hi] = P.xr, bw = (hi - lo) / bins, c = new Array(bins).fill(0);
    for (const v of vals) { const i = Math.floor((v - lo) / bw); if (i >= 0 && i < bins) c[i]++; }
    const sc = o.scale || 1;
    clip(P, ctx => { ctx.fillStyle = color; ctx.globalAlpha = o.alpha ?? 0.55;
      c.forEach((k, i) => { if (!k) return; const x0 = P.X(lo + i * bw), x1 = P.X(lo + (i + 1) * bw); ctx.fillRect(x0 + 0.5, P.Y(k * sc), Math.max(1, x1 - x0 - 1), P.Y(0) - P.Y(k * sc)); });
      ctx.globalAlpha = 1; });
    return c;
  }
  const histMax = (vals, lo, hi, bins) => { const bw = (hi - lo) / bins, c = new Array(bins).fill(0); for (const v of vals) { const i = Math.floor((v - lo) / bw); if (i >= 0 && i < bins) c[i]++; } return Math.max(1, ...c); };
  function message(cv, text) {
    const P = chart(cv, [0, 1], [0, 1], { noY: true, xticks: [] }); const { ctx, w, h } = P;
    ctx.fillStyle = css("--muted"); ctx.font = `14px ${css("--f-body")}`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    const words = text.split(" "), lines = []; let line = "";
    for (const wd of words) { const t = line ? line + " " + wd : wd; if (ctx.measureText(t).width > w - 40) { lines.push(line); line = wd; } else line = t; }
    lines.push(line); lines.forEach((l, i) => ctx.fillText(l, w / 2, h / 2 + (i - (lines.length - 1) / 2) * 20));
  }

  // ---------- Stata-like regression table ----------
  // models: [{title, cls, R, names:[...regressor keys], robust:bool}] ; rows: [[key,label]] ; stats: [[label, m=>string]]
  function regTable(models, rows, stats) {
    let h = `<thead><tr><th scope="col"></th>${models.map(m => `<th scope="col" class="${m.cls || ""}">${m.title}</th>`).join("")}</tr></thead><tbody>`;
    for (const [key, lab] of rows) {
      if (!models.some(m => key === "_cons" || m.names.includes(key))) continue;
      h += `<tr><td>${lab}</td>` + models.map(m => {
        if (!m.R) return `<td class="na">${m.note || "·"}</td>`;
        const j = key === "_cons" ? 0 : m.names.indexOf(key) + 1; if (key !== "_cons" && j === 0) return `<td class="na">${m.missing || "·"}</td>`;
        const se = m.robust ? m.R.seR[j] : m.R.se[j], t = m.robust ? m.R.tR[j] : m.R.t[j], p = m.robust ? m.R.pR[j] : m.R.p[j];
        return `<td><span class="${p < 0.05 ? "sig" : ""}">${f(m.R.b[j], m.dec || 3)}</span><span class="se">(${f(se, m.dec || 3)})  t=${f(t, 2)}  p=${fp(p)}</span></td>`;
      }).join("") + "</tr>";
    }
    if (stats && stats.length) {
      h += `<tr class="sec"><td colspan="${models.length + 1}">Fit</td></tr>`;
      for (const [lab, fn] of stats) h += `<tr><td>${lab}</td>${models.map(m => `<td>${m.R ? fn(m) : "·"}</td>`).join("")}</tr>`;
    }
    return h + "</tbody>";
  }


  // ---------- Stata-style regress output (plain text) ----------
  function g9(v) {
    if (!isFinite(v)) return ".";
    for (let p = 7; p >= 1; p--) { let s = v.toPrecision(p); if (s.includes("e")) s = (+s).toString(); if (s.includes(".") && !s.includes("e")) s = s.replace(/0+$/, "").replace(/\.$/, ""); if (s.length <= 9) return s; }
    return v.toExponential(2);
  }
  const pad = (s, w) => String(s).padStart(w);
  function stataOut(R, yname, xnames, level = 0.95) {
    const n = R.n, k = R.k - 1, tc = tcrit(R.df, 1 - level), lv = Math.round(level * 100);
    const L1 = [];
    const row = (src, ss, df, ms, lab, val) => L1.push(`${pad(src, 12)} | ${pad(g9(ss), 10)} ${pad(df, 9)} ${pad(g9(ms), 10)}   ${lab.padEnd(15)} = ${pad(val, 9)}`);
    L1.push(`${pad("Source", 12)} |       SS           df       MS      Number of obs   = ${pad(n, 9)}`);
    L1.push(`-------------+----------------------------------   ${("F(" + k + ", " + R.df + ")").padEnd(15)} = ${pad(R.F.toFixed(2), 9)}`);
    row("Model", R.sse, k, R.sse / k, "Prob > F", R.pF.toFixed(4));
    row("Residual", R.ssr, R.df, R.ssr / R.df, "R-squared", R.r2.toFixed(4));
    L1.push(`-------------+----------------------------------   Adj R-squared   = ${pad(R.adj.toFixed(4), 9)}`);
    row("Total", R.sst, n - 1, R.sst / (n - 1), "Root MSE", g9(+R.rmse.toPrecision(5)));
    L1.push("", "-".repeat(78));
    L1.push(`${pad(yname.slice(0, 12), 12)} | Coefficient  Std. err.      t    P>|t|     [${lv}% conf. interval]`);
    L1.push("-------------+" + "-".repeat(64));
    const names = [...xnames, "_cons"], idx = [...xnames.map((_, i) => i + 1), 0];
    names.forEach((nm, r) => { const j = idx[r];
      L1.push(`${pad(nm.slice(0, 12), 12)} | ${pad(g9(R.b[j]), 10)} ${pad(g9(R.se[j]), 10)} ${pad(R.t[j].toFixed(2), 8)} ${pad(R.p[j].toFixed(3), 7)}    ${pad(g9(R.b[j] - tc * R.se[j]), 9)}   ${pad(g9(R.b[j] + tc * R.se[j]), 9)}`); });
    L1.push("-".repeat(78));
    return L1.join("\n");
  }

  return { MAXN, stataOut, normals, nextSeed, lgamma, betai, gammaQ, pT, pF, pChi, tcrit, Fcrit, chiCrit, tpdf, Fpdf, inv, ols, mean, sd, sst, corr,
           f, fp, sgn, pct, css, mix, alpha, nice, range, chart, clip, dots, curve, band, area, vline, label, hist, histMax, message, regTable };
})();
if (typeof module !== "undefined") module.exports = Lib;
