// ===================== Modules =====================
(() => {
const L = Lib, $ = id => document.getElementById(id), num = id => +$(id).value;
const { f, fp, sgn, pct, css } = L;
const verdict = rej => `<span class="verdict ${rej ? "rej" : "acc"}">${rej ? "Reject H₀" : "Fail to reject H₀"}</span>`;
const stat = (k, v) => `<div><span class="k">${k}</span><span class="v">${v}</span></div>`;
const visible = cv => cv && cv.clientWidth > 0;
const modules = {};

// wire a module: inputs re-render; "new sample" reseeds; MC button runs simulation
function wire(key, mod, inputs, pre) {
  modules[key] = mod;
  inputs.forEach(id => $(id).addEventListener("input", () => { mod.mc = null; mod.render(); }));
  const nb = $(pre + "-new"); if (nb) nb.addEventListener("click", () => { mod.seed = L.nextSeed(mod.seed); mod.Z = null; mod.mc = null; mod.render(); });
  const mb = $(pre + "-mc"); if (mb) mb.addEventListener("click", () => { mb.disabled = true; setTimeout(() => { mod.runMC(); mb.disabled = false; mod.draw(); }, 10); });
}
const getZ = (mod, m) => mod.Z || (mod.Z = L.normals(mod.seed, L.MAXN, m));

// =====================================================================
// 1. Multiple regression & OVB
// =====================================================================
const ovb = {
  seed: 20261014, Z: null, mc: null, last: null,
  P() { return { b1: num("o-b1"), b2: num("o-b2"), rho: num("o-rho"), sig: num("o-sig"), n: num("o-n"), x3: $("o-x3").checked }; },
  data(Z, p, n) {
    const s = Math.sqrt(1 - p.rho * p.rho), x1 = new Float64Array(n), x2 = new Float64Array(n), x3 = new Float64Array(n), y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      x1[i] = 10 + 2 * Z[0][i]; x2[i] = 20 + 3 * (p.rho * Z[0][i] + s * Z[1][i]); x3[i] = 50 + 10 * Z[2][i];
      y[i] = 100 + p.b1 * x1[i] + p.b2 * x2[i] + p.sig * Z[3][i];
    }
    return { x1, x2, x3, y };
  },
  render() {
    const p = this.P();
    $("o-b1o").textContent = f(p.b1, 1); $("o-b2o").textContent = f(p.b2, 1); $("o-rhoo").textContent = f(p.rho, 2); $("o-sigo").textContent = f(p.sig, 1);
    $("o-dgp").innerHTML = `True model: y = 100 <b>${sgn(p.b1)}</b>·x₁ <b>${sgn(p.b2)}</b>·x₂ + u &nbsp;·&nbsp; corr(x₁, x₂) = <b>${f(p.rho, 2)}</b> &nbsp;·&nbsp; σ = <b>${p.sig.toFixed(1)}</b> &nbsp;·&nbsp; n = <b>${p.n}</b>`;
    const d = this.data(getZ(this, 4), p, p.n);
    const S = L.ols([d.x1], d.y), Lg = L.ols([d.x1, d.x2], d.y), L3 = p.x3 ? L.ols([d.x1, d.x2, d.x3], d.y) : null;
    const delta = L.ols([d.x1], d.x2).b[1];
    const vif = 1 / (1 - L.ols([d.x2], d.x1).r2), vif3 = p.x3 ? 1 / (1 - L.ols([d.x2, d.x3], d.x1).r2) : NaN;
    this.last = { p, d, S, Lg, L3, delta, vif, vif3 };
    // table
    const models = [{ title: "Short: y on x₁", cls: "c-short", R: S, names: ["x1"], vif: 1 }, { title: "Long: y on x₁, x₂", cls: "c-long", R: Lg, names: ["x1", "x2"], vif }];
    if (L3) models.push({ title: "Long + x₃", cls: "c-long", R: L3, names: ["x1", "x2", "x3"], vif: vif3 });
    $("o-tab").innerHTML = L.regTable(models, [["x1", "x₁"], ["x2", "x₂"], ["x3", "x₃"], ["_cons", "constant"]], [
      ["R²", m => f(m.R.r2, 4)], ["Adjusted R²", m => f(m.R.adj, 4)], ["Root MSE (σ̂)", m => f(m.R.rmse, 3)], ["SSR", m => f(m.R.ssr, 1)],
      ["Overall F (p)", m => `${f(m.R.F, 2)} (${fp(m.R.pF)})`], ["df residual", m => m.R.df], ["VIF for x₁", m => f(m.vif, 2)]]);
    // notes
    const out = [];
    out.push(`Adding x₂ moved β̂₁ from <code>${f(S.b[1])}</code> to <code>${f(Lg.b[1])}</code>. The true value is <code>${f(p.b1, 1)}</code>.` +
      (Math.abs(p.rho) < 0.05 ? " x₁ and x₂ are (almost) uncorrelated, so leaving x₂ out barely biases β̂₁."
        : Math.sign(S.b[1]) !== Math.sign(Lg.b[1]) && Math.abs(S.b[1]) > 0.05 ? " The sign flipped: the short regression gets even the direction wrong." : ""));
    out.push(`se(β̂₁) went from <code>${f(S.se[1])}</code> to <code>${f(Lg.se[1])}</code> (×${(Lg.se[1] / S.se[1]).toFixed(2)}). Two forces pull in opposite directions: σ̂ falls from <code>${f(S.rmse, 2)}</code> to <code>${f(Lg.rmse, 2)}</code> because x₂ leaves the error, but the VIF of <code>${f(vif, 2)}</code> inflates the variance because x₁ and x₂ move together.`);
    const Fq = (S.ssr - Lg.ssr) / (Lg.ssr / Lg.df);
    out.push(`R² rose from <code>${f(S.r2, 3)}</code> to <code>${f(Lg.r2, 3)}</code>. F-test for adding x₂: F = (SSR<sub>R</sub> − SSR<sub>U</sub>)/1 ÷ SSR<sub>U</sub>/${Lg.df} = <code>${f(Fq, 2)}</code>, which equals t² = <code>${f(Lg.t[2] ** 2, 2)}</code>.`);
    if (L3) {
      const t3 = Math.abs(L3.t[3]);
      out.push(`x₃ is pure noise. Its |t| = <code>${f(t3, 2)}</code>, so adjusted R² ${L3.adj > Lg.adj ? "rose" : "fell"} (<code>${f(Lg.adj, 4)}</code> → <code>${f(L3.adj, 4)}</code>) while plain R² still rose. Adjusted R² goes up exactly when the added variable has |t| &gt; 1.` +
        (L3.p[3] < 0.05 ? " It is “significant” at 5% in this sample: that happens 1 time in 20 by chance." : ""));
    }
    $("o-notes").innerHTML = out.map(s => `<li>${s}</li>`).join("");
    $("o-ovb").innerHTML = `Omitted variable bias, exact in-sample:<br>
      <span class="s">β̃₁</span> = <span class="l">β̂₁</span> + <span class="l">β̂₂</span> · δ̃₁<br>
      <span class="s">${f(S.b[1])}</span> = <span class="l">${f(Lg.b[1])}</span> + <span class="l">${f(Lg.b[2])}</span> · ${f(delta)} &nbsp;(bias = ${f(Lg.b[2] * delta)})<br>
      <span class="m">δ̃₁ = slope from regressing x₂ on x₁. Population value: corr · 3/2 = ${f(1.5 * p.rho, 2)}.</span>`;
    this.draw();
  },
  draw() {
    if (!this.last || !visible($("o-cvRaw"))) return;
    const { d, p, S, Lg, L3 } = this.last;
    // raw scatter
    let P = L.chart($("o-cvRaw"), L.range([d.x1]), L.range([d.y]), { xlab: "x₁", ylab: "y" });
    const [lo2, hi2] = L.range([d.x2], 0), clo = css("--seq-lo"), chi = css("--seq-hi");
    L.dots(P, d.x1, d.y, i => L.mix(clo, chi, (d.x2[i] - lo2) / (hi2 - lo2 || 1)), { alpha: 0.85 });
    const m1 = L.mean(d.x1), my = L.mean(d.y);
    L.curve(P, x => S.b[0] + S.b[1] * x, css("--short"), { N: 2 });
    L.curve(P, x => my + Lg.b[1] * (x - m1), css("--long"), { N: 2, dash: [7, 5] });
    // added-variable plot
    const others = p.x3 ? [d.x2, d.x3] : [d.x2];
    const rx = L.ols(others, d.x1).res, ry = L.ols(others, d.y).res, B = (L3 || Lg).b[1];
    P = L.chart($("o-cvAV"), L.range([rx]), L.range([ry]), { xlab: "x₁ residualized on " + (p.x3 ? "x₂, x₃" : "x₂"), ylab: "y residualized" });
    L.dots(P, rx, ry, css("--dot"), { alpha: 0.6 });
    L.curve(P, x => B * x, css("--long"), { N: 2 });
    // MC
    const cv = $("o-cvMC");
    if (!this.mc) { L.message(cv, "Press “Run 500 samples” to see how the short and long estimators behave across samples."); $("o-mcstats").innerHTML = ""; return; }
    const mc = this.mc, pred = mc.p.b1 + mc.p.b2 * 1.5 * mc.p.rho;
    const xr = L.range([mc.sh, mc.lg], 0.05, [mc.p.b1, pred]), bins = 40;
    const ymax = Math.max(L.histMax(mc.sh, xr[0], xr[1], bins), L.histMax(mc.lg, xr[0], xr[1], bins)) * 1.12;
    P = L.chart(cv, xr, [0, ymax], { xlab: "estimate of β₁", ylab: "count" });
    L.hist(P, mc.sh, bins, css("--short")); L.hist(P, mc.lg, bins, css("--long"));
    L.vline(P, mc.p.b1, css("--ink"), { label: "true β₁" });
    L.vline(P, pred, css("--short"), { dash: [4, 4], label: "OVB prediction", row: 1 });
    $("o-mcstats").innerHTML = stat("mean short β̃₁", f(L.mean(mc.sh))) + stat("OVB prediction β₁ + β₂·δ₁ (dashed)", f(pred)) +
      stat("mean long β̂₁", f(L.mean(mc.lg))) + stat("SD of long β̂₁ across samples", f(L.sd(mc.lg))) +
      stat("average reported se(β̂₁)", f(L.mean(mc.se))) + stat("95% CIs containing true β₁", pct(mc.cover));
  },
  runMC() {
    const p = this.P(), R = 500, sh = [], lg = [], se = []; let cover = 0, tc = null;
    for (let r = 0; r < R; r++) {
      const d = this.data(L.normals(this.seed + 1000 + r, p.n, 4), p, p.n);
      const S = L.ols([d.x1], d.y), Lg = L.ols([d.x1, d.x2], d.y);
      if (tc === null) tc = L.tcrit(Lg.df, 0.05);
      sh.push(S.b[1]); lg.push(Lg.b[1]); se.push(Lg.se[1]);
      if (Math.abs(Lg.b[1] - p.b1) <= tc * Lg.se[1]) cover++;
    }
    this.mc = { p, sh, lg, se, cover: cover / R };
  }
};
wire("ovb", ovb, ["o-b1", "o-b2", "o-rho", "o-sig", "o-n", "o-x3"], "o");

// =====================================================================
// 2. Multicollinearity & VIF
// =====================================================================
const vif = {
  seed: 31415, Z: null, mc: null, last: null,
  P() { return { rho: num("v-rho"), sig: num("v-sig"), n: num("v-n"), perf: $("v-perf").checked }; },
  data(Z, p, n, rho = p.rho) {
    const s = Math.sqrt(Math.max(0, 1 - rho * rho)), x1 = new Float64Array(n), x2 = new Float64Array(n), y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      x1[i] = 10 + 2 * Z[0][i];
      x2[i] = p.perf ? 5 + 1.5 * x1[i] : 20 + 3 * (rho * Z[0][i] + s * Z[1][i]);
      y[i] = 100 - x1[i] + 2 * x2[i] + p.sig * Z[2][i];
    }
    return { x1, x2, y };
  },
  render() {
    const p = this.P();
    $("v-rhoo").textContent = p.perf ? "1 (exact)" : f(p.rho, 3); $("v-sigo").textContent = f(p.sig, 1);
    $("v-rho").disabled = p.perf;
    $("v-dgp").innerHTML = `True model: y = 100 <b>− 1.0</b>·x₁ <b>+ 2.0</b>·x₂ + u &nbsp;·&nbsp; corr(x₁, x₂) = <b>${p.perf ? "1 (x₂ = 5 + 1.5·x₁)" : f(p.rho, 3)}</b> &nbsp;·&nbsp; σ = <b>${p.sig.toFixed(1)}</b> &nbsp;·&nbsp; n = <b>${p.n}</b> &nbsp;·&nbsp; population VIF = 1/(1 − corr²) = <b>${p.perf ? "∞" : f(1 / (1 - p.rho ** 2), 2)}</b>`;
    const d = this.data(getZ(this, 3), p, p.n);
    const R = L.ols([d.x1, d.x2], d.y);
    const r1 = p.perf ? 1 : L.ols([d.x2], d.x1).r2, VIF = 1 / (1 - r1), SST1 = L.sst(d.x1);
    this.last = { p, d, R, r1, VIF, SST1 };
    if (!R) {   // perfect collinearity
      const S = L.ols([d.x1], d.y);
      $("v-tab").innerHTML = `<caption style="caption-side:top;text-align:left;font-family:var(--f-mono);font-size:13px;color:var(--reject);padding-bottom:8px">note: x2 omitted because of collinearity</caption>` +
        L.regTable([{ title: "regress y x1 x2", cls: "c-long", R: S, names: ["x1"] }], [["x1", "x₁"], ["_cons", "constant"]], [["R²", m => f(m.R.r2, 4)], ["Root MSE", m => f(m.R.rmse, 3)]]);
      $("v-ftest").innerHTML = `<span class="r">X′X cannot be inverted.</span> Any change in x₁ comes with an exact change in x₂, so the data cannot separate β₁ from β₂.<br>
        <span class="m">What Stata reports for x₁ (${f(S.b[1])}) is β₁ + 1.5·β₂ = −1 + 3 = 2: the combined effect, not β₁.</span>`;
      $("v-formula").innerHTML = `r₁² = 1 &nbsp;⇒&nbsp; VIF = 1/(1 − 1) = <span class="r">∞</span><br><span class="m">The no-perfect-multicollinearity assumption fails.</span>`;
    } else {
      const se0 = R.rmse / Math.sqrt(SST1);
      $("v-formula").innerHTML = `se(β̂₁) = σ̂ / √[SST₁ (1 − r₁²)]<br>
        &nbsp; = ${f(R.rmse, 3)} / √[${f(SST1, 1)} × (1 − ${f(r1, 4)})] = <span class="l">${f(R.se[1], 4)}</span><br>
        &nbsp; = ${f(se0, 4)} × √VIF = ${f(se0, 4)} × √${f(VIF, 2)}`;
      const mx = Math.max(se0, R.se[1]) * 1.08;
      $("v-bars").innerHTML = [["if x₁ ⟂ x₂ (VIF = 1)", se0, "var(--muted)"], ["actual (VIF = " + f(VIF, 1) + ")", R.se[1], "var(--long)"]].map(([k, v, c]) =>
        `<div class="bar"><span>${k}</span><span class="track"><span class="fill" style="width:${(100 * v / mx).toFixed(1)}%;background:${c}"></span></span><span class="num">${f(v, 3)}</span></div>`).join("");
      $("v-tab").innerHTML = L.regTable([{ title: "regress y x1 x2", cls: "c-long", R, names: ["x1", "x2"] }], [["x1", "x₁"], ["x2", "x₂"], ["_cons", "constant"]], [
        ["R²", m => f(m.R.r2, 4)], ["Root MSE (σ̂)", m => f(m.R.rmse, 3)], ["r₁² (x₁ on x₂)", () => f(r1, 4)], ["VIF (same for x₁ and x₂)", () => f(VIF, 2)]]);
      const Fr = R.pF < 0.05, t1 = R.p[1] < 0.05, t2 = R.p[2] < 0.05;
      $("v-ftest").innerHTML = `Individual: t₁ = ${f(R.t[1], 2)} (p = ${fp(R.p[1])}) ${verdict(t1)} &nbsp; t₂ = ${f(R.t[2], 2)} (p = ${fp(R.p[2])}) ${verdict(t2)}<br>
        Joint H₀: β₁ = β₂ = 0: F(2, ${R.df}) = ${f(R.F, 2)} (p = ${fp(R.pF)}) ${verdict(Fr)}` +
        (!t1 && !t2 && Fr ? `<br><span class="l">Neither slope is significant alone, but together they clearly matter: the data cannot tell which one does the work.</span>` : "");
    }
    if (p.perf) $("v-bars").innerHTML = "";
    this.draw();
  },
  draw() {
    if (!this.last || !visible($("v-cvX"))) return;
    const { d, VIF, p } = this.last;
    let P = L.chart($("v-cvX"), L.range([d.x1]), L.range([d.x2]), { xlab: "x₁", ylab: "x₂" });
    L.dots(P, d.x1, d.x2, css("--dot"), { alpha: 0.6 });
    L.label(P, `r₁² = ${p.perf ? "1" : f(this.last.r1, 3)}   VIF = ${p.perf ? "∞" : f(VIF, 2)}`, P.m.l + 4, P.m.t + 4, css("--ink"));
    // VIF gauge (log scale)
    const lmax = Math.log10(200), v = Math.min(lmax, Math.log10(Math.max(1, VIF)));
    P = L.chart($("v-cvGauge"), [0, lmax], [0, 1], { noY: true, xticks: [1, 2, 5, 10, 20, 50, 100, 200].map(Math.log10), xfmt: x => String(Math.round(10 ** x)), xlab: "VIF (log scale)" });
    const { ctx } = P;
    ctx.fillStyle = L.alpha(css("--reject"), 0.14); ctx.fillRect(P.X(1), P.m.t, P.X(lmax) - P.X(1), P.h - P.m.t - P.m.b);
    L.label(P, "VIF ≥ 10: multicollinearity problem", P.X(1), P.m.t + 2, css("--reject"));
    ctx.fillStyle = css("--long"); ctx.fillRect(P.X(0), P.Y(0.62), P.X(v) - P.X(0), P.Y(0.22) - P.Y(0.62));
    L.vline(P, v, css("--ink"), { width: 2.5 });
    // MC
    const cv = $("v-cvMC");
    if (!this.mc) { L.message(cv, p.perf ? "With perfect collinearity there is nothing to simulate: OLS cannot be computed." : "Press “Run 500 samples” to draw 500 samples at this correlation and 500 at zero correlation."); $("v-mcstats").innerHTML = ""; return; }
    const mc = this.mc;
    const xr = L.range([mc.b1, mc.b1z], 0.05, [-1]), yr = L.range([mc.b2, mc.b2z], 0.05, [2]);
    P = L.chart(cv, xr, yr, { xlab: "β̂₁", ylab: "β̂₂" });
    L.dots(P, mc.b1z, mc.b2z, css("--muted"), { alpha: 0.35, r: 2 });
    L.dots(P, mc.b1, mc.b2, css("--long"), { alpha: 0.55, r: 2 });
    L.vline(P, -1, css("--ink"), { width: 1.5, dash: [4, 4] });
    P.ctx.save(); P.ctx.strokeStyle = css("--ink"); P.ctx.setLineDash([4, 4]); P.ctx.lineWidth = 1.5; P.ctx.beginPath(); P.ctx.moveTo(P.m.l, P.Y(2)); P.ctx.lineTo(P.w - P.m.r, P.Y(2)); P.ctx.stroke(); P.ctx.restore();
    $("v-mcstats").innerHTML = stat("SD of β̂₁ at this corr", f(L.sd(mc.b1), 4)) + stat("SD of β̂₁ at corr = 0", f(L.sd(mc.b1z), 4)) +
      stat("ratio (≈ √VIF = " + f(Math.sqrt(1 / (1 - mc.p.rho ** 2)), 2) + ")", f(L.sd(mc.b1) / L.sd(mc.b1z), 2)) +
      stat("mean β̂₁ (true −1)", f(L.mean(mc.b1))) + stat("corr(β̂₁, β̂₂) across samples", f(L.corr(mc.b1, mc.b2), 2)) +
      stat("samples with t₁ significant", pct(mc.s1)) + stat("samples with t₂ significant", pct(mc.s2)) +
      stat("samples where joint F rejects", pct(mc.sF)) + stat("F rejects, both t insignificant", pct(mc.sBoth));
  },
  runMC() {
    const p = this.P(); if (p.perf) { this.mc = null; return; }
    const R = 500, b1 = [], b2 = [], b1z = [], b2z = []; let s1 = 0, s2 = 0, sF = 0, sB = 0;
    for (let r = 0; r < R; r++) {
      const Z = L.normals(this.seed + 2000 + r, p.n, 3);
      const d = this.data(Z, p, p.n), E = L.ols([d.x1, d.x2], d.y);
      const dz = this.data(Z, p, p.n, 0), Ez = L.ols([dz.x1, dz.x2], dz.y);
      b1.push(E.b[1]); b2.push(E.b[2]); b1z.push(Ez.b[1]); b2z.push(Ez.b[2]);
      const a = E.p[1] < 0.05, b = E.p[2] < 0.05, c = E.pF < 0.05;
      s1 += a; s2 += b; sF += c; sB += (!a && !b && c);
    }
    this.mc = { p, b1, b2, b1z, b2z, s1: s1 / R, s2: s2 / R, sF: sF / R, sBoth: sB / R };
  }
};
wire("vif", vif, ["v-rho", "v-sig", "v-n", "v-perf"], "v");

// =====================================================================
// 3. Inference: t, F, CIs
// =====================================================================
const inf = {
  seed: 27182, Z: null, mc: null, last: null,
  P() { return { b1: num("i-b1"), c: num("i-c"), a: num("i-a"), b2: num("i-b2"), b3: num("i-b3"), sig: num("i-sig"), n: num("i-n") }; },
  data(Z, p, n) {
    const s = Math.sqrt(1 - 0.25), x1 = new Float64Array(n), x2 = new Float64Array(n), x3 = new Float64Array(n), y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      x1[i] = 10 + 2 * Z[0][i]; x2[i] = 20 + 3 * (0.5 * Z[0][i] + s * Z[1][i]); x3[i] = 50 + 10 * Z[2][i];
      y[i] = 100 + p.b1 * x1[i] + p.b2 * x2[i] + p.b3 * x3[i] + p.sig * Z[3][i];
    }
    return { x1, x2, x3, y };
  },
  stats(d, p) {
    const U = L.ols([d.x1, d.x2, d.x3], d.y), Rr = L.ols([d.x1], d.y);
    const t = (U.b[1] - p.c) / U.se[1], pt = L.pT(t, U.df);
    const F = ((Rr.ssr - U.ssr) / 2) / (U.ssr / U.df);
    return { U, Rr, t, pt, F, pF: L.pF(F, 2, U.df) };
  },
  render() {
    const p = this.P();
    $("i-b1o").textContent = f(p.b1, 1); $("i-co").textContent = f(p.c, 1); $("i-b2o").textContent = f(p.b2, 2); $("i-b3o").textContent = f(p.b3, 2); $("i-sigo").textContent = f(p.sig, 1);
    $("i-dgp").innerHTML = `True model: y = 100 <b>${sgn(p.b1)}</b>·x₁ <b>${sgn(p.b2, 2)}</b>·x₂ <b>${sgn(p.b3, 2)}</b>·x₃ + u &nbsp;·&nbsp; corr(x₁, x₂) = 0.5, x₃ ~ N(50, 10²) independent &nbsp;·&nbsp; σ = <b>${p.sig.toFixed(1)}</b> &nbsp;·&nbsp; n = <b>${p.n}</b>`;
    const d = this.data(getZ(this, 4), p, p.n), S = this.stats(d, p), U = S.U;
    const tc = L.tcrit(U.df, p.a), Fc = L.Fcrit(2, U.df, p.a), lev = Math.round(100 * (1 - p.a));
    const lo = U.b[1] - tc * U.se[1], hi = U.b[1] + tc * U.se[1];
    this.last = { p, d, S, tc, Fc, lo, hi, lev };
    const rejT = Math.abs(S.t) > tc, rejF = S.F > Fc;
    $("i-tcalc").innerHTML = `t = (β̂₁ − c) / se(β̂₁) = (${f(U.b[1])} − (${f(p.c, 1)})) / ${f(U.se[1])} = <span class="l">${f(S.t, 3)}</span><br>
      df = n − k − 1 = ${p.n} − 3 − 1 = ${U.df} &nbsp;·&nbsp; critical |t| at α = ${p.a} is <span class="r">${f(tc, 3)}</span><br>
      p-value = ${fp(S.pt)} &nbsp; ${verdict(rejT)}`;
    const r2U = U.r2, r2R = S.Rr.r2;
    $("i-fcalc").innerHTML = `Unrestricted: y on x₁, x₂, x₃ &nbsp;SSR<sub>U</sub> = ${f(U.ssr, 1)}<br>
      Restricted: &nbsp; y on x₁ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SSR<sub>R</sub> = ${f(S.Rr.ssr, 1)}<br>
      F = [(SSR<sub>R</sub> − SSR<sub>U</sub>)/q] / [SSR<sub>U</sub>/(n−k−1)] = (${f(S.Rr.ssr - U.ssr, 1)}/2) / (${f(U.ssr, 1)}/${U.df}) = <span class="l">${f(S.F, 3)}</span><br>
      <span class="m">R² version: [(${f(r2U, 4)} − ${f(r2R, 4)})/2] / [(1 − ${f(r2U, 4)})/${U.df}] = ${f(((r2U - r2R) / 2) / ((1 - r2U) / U.df), 3)}</span><br>
      critical F(2, ${U.df}) = <span class="r">${f(Fc, 3)}</span> &nbsp;·&nbsp; p = ${fp(S.pF)} &nbsp; ${verdict(rejF)}`;
    const inside = p.c >= lo && p.c <= hi;
    $("i-cicalc").innerHTML = `${lev}% CI = β̂₁ ± t<sub>crit</sub> · se(β̂₁) = ${f(U.b[1])} ± ${f(tc, 3)} × ${f(U.se[1])} = <span class="l">[${f(lo)}, ${f(hi)}]</span><br>
      c = ${f(p.c, 1)} is ${inside ? "inside" : "outside"} the interval ⇔ the two-sided test at α = ${p.a} ${inside ? "fails to reject" : "rejects"}. ${verdict(!inside)}<br>
      <span class="m">Meaning: ${lev}% of intervals built this way across repeated samples contain the true β₁. This one ${hi >= p.b1 && lo <= p.b1 ? "does" : "does not"} (true β₁ = ${f(p.b1, 1)}).</span>`;
    $("i-tab").innerHTML = L.regTable([{ title: "regress y x1 x2 x3", cls: "c-long", R: U, names: ["x1", "x2", "x3"] }], [["x1", "x₁"], ["x2", "x₂"], ["x3", "x₃"], ["_cons", "constant"]], [
      ["R²", m => f(m.R.r2, 4)], ["Adjusted R²", m => f(m.R.adj, 4)], ["Root MSE (σ̂)", m => f(m.R.rmse, 3)], ["Overall F (p)", m => `${f(m.R.F, 2)} (${fp(m.R.pF)})`], ["df residual", m => m.R.df]]);
    this.draw();
  },
  draw() {
    if (!this.last || !visible($("i-cvT"))) return;
    const { p, S, tc, Fc, lo, hi, lev } = this.last, df = S.U.df, rej = L.alpha(css("--reject"), 0.35);
    // t plot
    const lim = Math.min(14, Math.max(4.5, Math.abs(S.t) + 0.8));
    let P = L.chart($("i-cvT"), [-lim, lim], [0, L.tpdf(0, df) * 1.15], { xlab: `t under H₀ (df = ${df})`, ylab: "density" });
    const pdf = x => L.tpdf(x, df);
    L.area(P, pdf, -lim, -tc, rej); L.area(P, pdf, tc, lim, rej);
    L.curve(P, pdf, css("--ink"), { width: 2 });
    L.vline(P, -tc, css("--reject"), { width: 1.2, dash: [3, 3] }); L.vline(P, tc, css("--reject"), { width: 1.2, dash: [3, 3], label: "±" + f(tc, 2), row: 1 });
    if (Math.abs(S.t) <= lim) L.vline(P, S.t, css("--long"), { width: 2.5, label: "t = " + f(S.t, 2) });
    else L.label(P, `t = ${f(S.t, 2)} (off the chart ${S.t > 0 ? "→" : "←"})`, S.t > 0 ? P.w - P.m.r : P.m.l, P.m.t + 2, css("--long"), S.t > 0 ? "right" : "left");
    // F plot
    const fmax = Math.max(6, Fc * 1.8, Math.min(S.F * 1.15, 40));
    const fpdf = x => L.Fpdf(Math.max(x, 1e-6), 2, df);
    P = L.chart($("i-cvF"), [0, fmax], [0, 1.1], { xlab: `F under H₀ (2, ${df})`, ylab: "density" });
    L.area(P, fpdf, Fc, fmax, rej); L.curve(P, fpdf, css("--ink"), { width: 2 });
    L.vline(P, Fc, css("--reject"), { width: 1.2, dash: [3, 3], label: "critical " + f(Fc, 2), row: 1 });
    if (S.F <= fmax) L.vline(P, S.F, css("--long"), { width: 2.5, label: "F = " + f(S.F, 2) });
    else L.label(P, `F = ${f(S.F, 1)} (off the chart →)`, P.w - P.m.r, P.m.t + 2, css("--long"), "right");
    // CI strip
    const xr = L.range([[lo, hi, p.c, p.b1]], 0.15);
    P = L.chart($("i-cvCI"), xr, [0, 1], { noY: true, xlab: "β₁" });
    const { ctx } = P, y0 = P.Y(0.5);
    ctx.strokeStyle = css("--long"); ctx.lineWidth = 6; ctx.lineCap = "round"; ctx.beginPath(); ctx.moveTo(P.X(lo), y0); ctx.lineTo(P.X(hi), y0); ctx.stroke(); ctx.lineCap = "butt";
    ctx.fillStyle = css("--surface"); ctx.strokeStyle = css("--long"); ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(P.X(S.U.b[1]), y0, 6, 0, 7); ctx.fill(); ctx.stroke();
    L.vline(P, p.c, css("--short"), { width: 2.5, label: "c = " + f(p.c, 1) });
    L.vline(P, p.b1, css("--ink"), { width: 1.5, dash: [4, 3], label: "true β₁", row: 1 });
    L.label(P, `${lev}% CI`, P.X(lo), y0 + 10, css("--long"));
    // MC
    const cv = $("i-cvMC");
    if (!this.mc) { L.message(cv, "Press “Run 1,000 samples”. The first 100 confidence intervals are drawn; the rates below use all 1,000."); $("i-mcstats").innerHTML = ""; return; }
    const mc = this.mc, show = mc.ci.slice(0, 100);
    const all = show.flat().concat([mc.p.b1, mc.p.c]);
    P = L.chart(cv, L.range([all], 0.05), [0, show.length + 1], { xlab: "β₁", noY: true });
    show.forEach(([a, b], i) => {
      const hit = a <= mc.p.b1 && mc.p.b1 <= b; P.ctx.strokeStyle = hit ? css("--long") : css("--reject"); P.ctx.lineWidth = hit ? 1.6 : 2.4;
      P.ctx.beginPath(); P.ctx.moveTo(P.X(a), P.Y(i + 1)); P.ctx.lineTo(P.X(b), P.Y(i + 1)); P.ctx.stroke();
    });
    L.vline(P, mc.p.b1, css("--ink"), { width: 1.5, label: "true β₁" });
    if (Math.abs(mc.p.c - mc.p.b1) > 1e-9) L.vline(P, mc.p.c, css("--short"), { width: 1.5, dash: [4, 3], label: "c", row: 1 });
    const isNullT = Math.abs(mc.p.c - mc.p.b1) < 1e-9, isNullF = mc.p.b2 === 0 && mc.p.b3 === 0, a = mc.p.a;
    $("i-mcstats").innerHTML = stat(`${Math.round(100 * (1 - a))}% CIs containing true β₁`, pct(mc.cover)) +
      stat(`rejected H₀: β₁ = ${f(mc.p.c, 1)}`, pct(mc.rejT) + (isNullT ? ` (H₀ true: type I error, ≈ α = ${a})` : " (H₀ false: power)")) +
      stat("rejected H₀: β₂ = β₃ = 0", pct(mc.rejF) + (isNullF ? ` (H₀ true: ≈ α = ${a})` : " (H₀ false: power)"));
  },
  runMC() {
    const p = this.P(), R = 1000, ci = []; let cover = 0, rT = 0, rF = 0, tc = null, Fc = null;
    for (let r = 0; r < R; r++) {
      const S = this.stats(this.data(L.normals(this.seed + 3000 + r, p.n, 4), p, p.n), p), U = S.U;
      if (tc === null) { tc = L.tcrit(U.df, p.a); Fc = L.Fcrit(2, U.df, p.a); }
      const a = U.b[1] - tc * U.se[1], b = U.b[1] + tc * U.se[1];
      if (r < 100) ci.push([a, b]);
      if (a <= p.b1 && p.b1 <= b) cover++;
      if (Math.abs(S.t) > tc) rT++; if (S.F > Fc) rF++;
    }
    this.mc = { p, ci, cover: cover / R, rejT: rT / R, rejF: rF / R };
  }
};
wire("inf", inf, ["i-b1", "i-c", "i-a", "i-b2", "i-b3", "i-sig", "i-n"], "i");

// =====================================================================
// 4. Quadratics, adjusted R² & prediction
// =====================================================================
const form = {
  seed: 16180, Z: null, mc: null, last: null,
  P() { return { b: num("q-b"), q: num("q-q"), sig: num("q-sig"), n: num("q-n"), x0: num("q-x0"), k: num("q-k") }; },
  render() {
    const p = this.P(), Z = getZ(this, 22), n = p.n;
    $("q-bo").textContent = f(p.b, 1); $("q-qo").textContent = f(p.q, 2); $("q-sigo").textContent = f(p.sig, 1); $("q-x0o").textContent = f(p.x0, 1); $("q-ko").textContent = p.k;
    $("q-dgp").innerHTML = `True model: y = 50 <b>${sgn(p.b)}</b>·x <b>${sgn(p.q, 2)}</b>·x² + u &nbsp;·&nbsp; x ~ N(10, 3²) &nbsp;·&nbsp; σ = <b>${p.sig.toFixed(1)}</b> &nbsp;·&nbsp; n = <b>${n}</b>`;
    const x = new Float64Array(n), x2 = new Float64Array(n), y = new Float64Array(n);
    for (let i = 0; i < n; i++) { x[i] = 10 + 3 * Z[0][i]; x2[i] = x[i] * x[i]; y[i] = 50 + p.b * x[i] + p.q * x2[i] + p.sig * Z[1][i]; }
    const noise = Array.from({ length: 20 }, (_, j) => Z[2 + j].subarray(0, n));
    const Lin = L.ols([x], y), Q = L.ols([x, x2], y), path = [];
    for (let j = 0; j <= 20; j++) { const R = j ? L.ols([x, x2, ...noise.slice(0, j)], y) : Q; path.push([R.r2, R.adj]); }
    const Qk = p.k ? L.ols([x, x2, ...noise.slice(0, p.k)], y) : null;
    // marginal effect, turning point, prediction
    const V = Q.V, x0 = p.x0, ME = Q.b[1] + 2 * Q.b[2] * x0;
    const seME = Math.sqrt(V[1][1] + 4 * x0 * x0 * V[2][2] + 4 * x0 * V[1][2]);
    const g = [1, x0, x0 * x0], y0 = g.reduce((s, v, j) => s + v * Q.b[j], 0);
    const seY = Math.sqrt(g.reduce((s, a, i) => s + a * g.reduce((t, b, j) => t + b * V[i][j], 0), 0));
    const seE = Math.sqrt(seY ** 2 + Q.rmse ** 2), tc = L.tcrit(Q.df, 0.05);
    const tp = -Q.b[1] / (2 * Q.b[2]), [xmin, xmax] = L.range([x], 0);
    this.last = { p, x, x2, y, Lin, Q, Qk, path, ME, seME, y0, seY, seE, tc, tp, xmin, xmax };
    const models = [{ title: "Linear: y on x", cls: "c-short", R: Lin, names: ["x"] }, { title: "Quadratic: y on x, x²", cls: "c-long", R: Q, names: ["x", "x2"], dec: 4 }];
    if (Qk) models.push({ title: `Quadratic + ${p.k} noise`, cls: "c-long", R: Qk, names: ["x", "x2"], dec: 4 });
    $("q-tab").innerHTML = L.regTable(models, [["x", "x"], ["x2", "x²"], ["_cons", "constant"]], [
      ["R²", m => f(m.R.r2, 4)], ["Adjusted R²", m => f(m.R.adj, 4)], ["Root MSE (σ̂)", m => f(m.R.rmse, 3)], ["regressors k", m => m.R.k - 1], ["df residual", m => m.R.df]]) +
      (Qk ? `<caption style="caption-side:bottom;text-align:left;font-family:var(--f-body);font-size:12.5px;color:var(--muted);padding-top:6px">Coefficients on the ${p.k} noise regressors are not shown.</caption>` : "");
    const trueME = p.b + 2 * p.q * x0, inRange = isFinite(tp) && tp >= xmin && tp <= xmax;
    $("q-calc").innerHTML = `<b>Marginal effect at x = ${f(x0, 1)}</b><br>
      ∂ŷ/∂x = β̂₁ + 2β̂₂·x = ${f(Q.b[1], 3)} + 2 × ${f(Q.b[2], 4)} × ${f(x0, 1)} = <span class="l">${f(ME, 3)}</span> <span class="m">(se ${f(seME, 3)}; true ${f(trueME, 3)})</span><br>
      <span class="m">The linear model claims one slope everywhere: </span><span class="s">${f(Lin.b[1], 3)}</span><br>
      Turning point x* = −β̂₁ / (2β̂₂) = ${f(tp, 2)} <span class="m">${inRange ? "(inside the data range)" : `(outside the data, x ∈ [${f(xmin, 1)}, ${f(xmax, 1)}]: do not interpret)`}</span><br><br>
      <b>Prediction at x = ${f(x0, 1)}</b><br>
      ŷ₀ = ${f(y0, 2)} <span class="m">(true E[y | x₀] = ${f(50 + p.b * x0 + p.q * x0 * x0, 2)})</span><br>
      se(ŷ₀) = ${f(seY, 3)} &nbsp;·&nbsp; σ̂ = ${f(Q.rmse, 3)} &nbsp;·&nbsp; se(ê₀) = √(se(ŷ₀)² + σ̂²) = ${f(seE, 3)}<br>
      95% CI for the mean: <span class="l">[${f(y0 - tc * seY, 2)}, ${f(y0 + tc * seY, 2)}]</span><br>
      95% interval for one new observation: <span class="l">[${f(y0 - tc * seE, 2)}, ${f(y0 + tc * seE, 2)}]</span>`;
    this.draw();
  },
  draw() {
    if (!this.last || !visible($("q-cvFit"))) return;
    const { p, x, y, Lin, Q, path, ME, y0, tc, tp, xmin, xmax } = this.last, V = Q.V;
    const yhat = v => Q.b[0] + Q.b[1] * v + Q.b[2] * v * v;
    const seAt = v => { const g = [1, v, v * v]; return Math.sqrt(g.reduce((s, a, i) => s + a * g.reduce((t, b, j) => t + b * V[i][j], 0), 0)); };
    const xr = L.range([x], 0.04, [p.x0]), yr = L.range([y], 0.08);
    let P = L.chart($("q-cvFit"), xr, yr, { xlab: "x", ylab: "y" });
    const lc = css("--long");
    L.band(P, v => yhat(v) - tc * Math.sqrt(seAt(v) ** 2 + Q.rmse ** 2), v => yhat(v) + tc * Math.sqrt(seAt(v) ** 2 + Q.rmse ** 2), L.alpha(lc, 0.12));
    L.band(P, v => yhat(v) - tc * seAt(v), v => yhat(v) + tc * seAt(v), L.alpha(lc, 0.35));
    L.dots(P, x, y, css("--dot"), { alpha: 0.55 });
    L.curve(P, v => Lin.b[0] + Lin.b[1] * v, css("--short"), { N: 2 });
    L.curve(P, yhat, lc);
    const hw = (xr[1] - xr[0]) * 0.12;
    P.ctx.save(); P.ctx.strokeStyle = css("--ink"); P.ctx.lineWidth = 2; P.ctx.beginPath(); P.ctx.moveTo(P.X(p.x0 - hw), P.Y(y0 - ME * hw)); P.ctx.lineTo(P.X(p.x0 + hw), P.Y(y0 + ME * hw)); P.ctx.stroke();
    P.ctx.fillStyle = css("--ink"); P.ctx.beginPath(); P.ctx.arc(P.X(p.x0), P.Y(y0), 4, 0, 7); P.ctx.fill(); P.ctx.restore();
    if (isFinite(tp) && tp >= xmin && tp <= xmax) L.vline(P, tp, css("--muted"), { width: 1, dash: [3, 3], label: "turning point", row: 0 });
    // residuals of the linear model + binned means
    const res = Lin.res;
    P = L.chart($("q-cvRes"), L.range([x], 0.04), L.range([res], 0.08), { xlab: "x", ylab: "residual (linear model)", zeroY: true });
    L.dots(P, x, res, css("--dot"), { alpha: 0.55 });
    const B = 10, bw = (xmax - xmin) / B, sum = new Array(B).fill(0), cnt = new Array(B).fill(0);
    for (let i = 0; i < x.length; i++) { const b = Math.min(B - 1, Math.floor((x[i] - xmin) / bw)); sum[b] += res[i]; cnt[b]++; }
    L.clip(P, ctx => { ctx.strokeStyle = css("--short"); ctx.lineWidth = 2.5; ctx.beginPath(); let st = false;
      for (let b = 0; b < B; b++) { if (cnt[b] < 3) continue; const xx = P.X(xmin + (b + 0.5) * bw), yy = P.Y(sum[b] / cnt[b]); st ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy); st = true; }
      ctx.stroke(); });
    L.label(P, "binned mean", P.m.l + 4, P.m.t + 4, css("--short"));
    // R² path
    const all = path.flat(), yr2 = L.range([all], 0.15);
    P = L.chart($("q-cvR2"), [-0.5, 20.5], yr2, { xlab: "irrelevant regressors added", ylab: "R²", xticks: [0, 5, 10, 15, 20] });
    for (const [idx, col] of [[0, css("--short")], [1, css("--long")]]) {
      L.clip(P, ctx => { ctx.strokeStyle = col; ctx.lineWidth = 2.5; ctx.beginPath(); path.forEach((r, j) => j ? ctx.lineTo(P.X(j), P.Y(r[idx])) : ctx.moveTo(P.X(j), P.Y(r[idx]))); ctx.stroke();
        ctx.fillStyle = col; path.forEach((r, j) => { ctx.beginPath(); ctx.arc(P.X(j), P.Y(r[idx]), j === p.k ? 5 : 2.2, 0, 7); ctx.fill(); }); });
    }
    L.vline(P, p.k, css("--muted"), { width: 1, dash: [3, 3] });
  }
};
wire("form", form, ["q-b", "q-q", "q-sig", "q-n", "q-x0", "q-k"], "q");

// =====================================================================
// 5. Heteroskedasticity
// =====================================================================
const het = {
  seed: 14142, Z: null, mc: null, last: null,
  P() { return { g: num("h-g"), sig: num("h-sig"), n: num("h-n") }; },
  sdu(p, x1) { return p.sig * Math.exp(p.g * (x1 - 10) / 2 - p.g * p.g); },   // E[Var(u|x)] = σ²
  data(Z, p, n) {
    const s = Math.sqrt(1 - 0.09), x1 = new Float64Array(n), x2 = new Float64Array(n), y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      x1[i] = 10 + 2 * Z[0][i]; x2[i] = 20 + 3 * (0.3 * Z[0][i] + s * Z[1][i]);
      y[i] = 100 - x1[i] + 2 * x2[i] + this.sdu(p, x1[i]) * Z[2][i];
    }
    return { x1, x2, y };
  },
  tests(d, R) {
    const n = d.y.length, e2 = R.res.map(v => v * v), fit2 = R.fit.map(v => v * v);
    const bp = L.ols([d.x1, d.x2], e2), wh = L.ols([R.fit, fit2], e2);
    return { bp, wh, LMbp: n * bp.r2, pbp: L.pChi(n * bp.r2, 2), LMwh: n * wh.r2, pwh: L.pChi(n * wh.r2, 2), e2 };
  },
  render() {
    const p = this.P();
    $("h-go").textContent = f(p.g, 2); $("h-sigo").textContent = f(p.sig, 1);
    $("h-dgp").innerHTML = `True model: y = 100 <b>− 1.0</b>·x₁ <b>+ 2.0</b>·x₂ + u &nbsp;·&nbsp; SD(u | x₁) = σ·exp(<b>${f(p.g, 2)}</b>·(x₁ − 10)/2 − γ²) &nbsp;·&nbsp; ${p.g === 0 ? "<b>homoskedastic</b>" : "<b>heteroskedastic</b>"} &nbsp;·&nbsp; σ = <b>${p.sig.toFixed(1)}</b> &nbsp;·&nbsp; n = <b>${p.n}</b>`;
    const d = this.data(getZ(this, 3), p, p.n), R = L.ols([d.x1, d.x2], d.y, { robust: true }), T = this.tests(d, R);
    const tc = L.tcrit(R.df, 0.05);
    this.last = { p, d, R, T };
    const ci = (m, j) => { const se = m.robust ? m.R.seR[j] : m.R.se[j]; return `[${f(m.R.b[j] - tc * se)}, ${f(m.R.b[j] + tc * se)}]`; };
    $("h-tab").innerHTML = L.regTable([{ title: "Usual SEs", cls: "c-short", R, names: ["x1", "x2"] }, { title: "Robust SEs (HC1)", cls: "c-long", R, names: ["x1", "x2"], robust: true }],
      [["x1", "x₁"], ["x2", "x₂"], ["_cons", "constant"]], [["95% CI for β₁", m => ci(m, 1)], ["95% CI for β₂", m => ci(m, 2)], ["se ratio robust / usual (x₁)", () => f(R.seR[1] / R.se[1], 2)], ["R²", () => f(R.r2, 4)], ["Root MSE (σ̂)", () => f(R.rmse, 3)]]);
    const c2 = L.chiCrit(2, 0.05);
    $("h-tests").innerHTML = `<b>Breusch–Pagan</b>: regress û² on x₁, x₂ &nbsp;→&nbsp; R²<sub>û²</sub> = ${f(T.bp.r2, 4)}<br>
      LM = n · R²<sub>û²</sub> = ${p.n} × ${f(T.bp.r2, 4)} = <span class="l">${f(T.LMbp, 2)}</span> ~ χ²(2) under H₀ &nbsp;·&nbsp; critical ${f(c2, 2)} &nbsp;·&nbsp; p = ${fp(T.pbp)} ${verdict(T.pbp < 0.05)}<br>
      <span class="m">F version: F(2, ${T.bp.df}) = ${f(T.bp.F, 2)}, p = ${fp(T.bp.pF)}</span><br><br>
      <b>White test (special form)</b>: regress û² on ŷ, ŷ² &nbsp;→&nbsp; R²<sub>û²</sub> = ${f(T.wh.r2, 4)}<br>
      LM = n · R²<sub>û²</sub> = <span class="l">${f(T.LMwh, 2)}</span> ~ χ²(2) &nbsp;·&nbsp; p = ${fp(T.pwh)} ${verdict(T.pwh < 0.05)}<br>
      <span class="m">H₀ in both: homoskedasticity, Var(u | x) = σ². Rejecting means: use robust standard errors.</span>` +
      (T.pbp < 0.05 && T.pwh >= 0.05 ? `<br><span class="m">Why the tests disagree here: the variance depends on x₁ only, but ŷ is driven mostly by x₂ (2·x₂ moves three times as much as x₁). The special-form White test looks only at ŷ, so it misses this pattern. A test is only as good as the variables it looks at.</span>` : "");
    this.draw();
  },
  draw() {
    if (!this.last || !visible($("h-cvRes"))) return;
    const { p, d, R, T } = this.last;
    let P = L.chart($("h-cvRes"), L.range([d.x1], 0.04), L.range([R.res], 0.08), { xlab: "x₁", ylab: "residual û", zeroY: true });
    L.dots(P, d.x1, R.res, css("--dot"), { alpha: 0.55 });
    L.curve(P, v => 1.96 * this.sdu(p, v), css("--long"), { dash: [6, 4], width: 2 });
    L.curve(P, v => -1.96 * this.sdu(p, v), css("--long"), { dash: [6, 4], width: 2 });
    const e2 = T.e2, line = L.ols([d.x1], e2);
    P = L.chart($("h-cvSq"), L.range([d.x1], 0.04), [0, Math.max(...e2) * 1.05], { xlab: "x₁", ylab: "û²" });
    L.dots(P, d.x1, e2, css("--dot"), { alpha: 0.55 });
    L.curve(P, v => line.b[0] + line.b[1] * v, css("--short"), { N: 2 });
    // MC
    const cv = $("h-cvMC");
    if (!this.mc) { L.message(cv, "Press “Run 1,000 samples” to compare the true sampling spread of β̂₁ with the usual and robust standard errors."); $("h-bars").innerHTML = ""; $("h-mcstats").innerHTML = ""; return; }
    const mc = this.mc, xr = L.range([mc.b], 0.05, [-1]), bins = 40;
    P = L.chart(cv, xr, [0, L.histMax(mc.b, xr[0], xr[1], bins) * 1.12], { xlab: "β̂₁", ylab: "count" });
    L.hist(P, mc.b, bins, css("--long"));
    L.vline(P, -1, css("--ink"), { label: "true β₁ = −1" });
    const bar = (k, v, mark, col) => `<div class="bar"><span>${k}</span><span class="track"><span class="fill" style="width:${(100 * v).toFixed(1)}%;background:${col}"></span><span class="mark" style="left:${100 * mark}%"></span></span><span class="num">${pct(v)}</span></div>`;
    $("h-bars").innerHTML = bar("95% CI coverage, usual", mc.covU, 0.95, "var(--short)") + bar("95% CI coverage, robust", mc.covR, 0.95, "var(--long)") +
      bar("Breusch–Pagan rejects", mc.bp, 0.05, "var(--muted)");
    $("h-mcstats").innerHTML = stat("mean β̂₁ (true −1)", f(L.mean(mc.b))) + stat("true SD of β̂₁ across samples", f(L.sd(mc.b), 4)) +
      stat("average usual se(β̂₁)", f(L.mean(mc.seU), 4)) + stat("average robust se(β̂₁)", f(L.mean(mc.seR), 4));
  },
  runMC() {
    const p = this.P(), N = 1000, b = [], seU = [], seR = []; let cu = 0, cr = 0, bp = 0, tc = null;
    for (let r = 0; r < N; r++) {
      const d = this.data(L.normals(this.seed + 4000 + r, p.n, 3), p, p.n), R = L.ols([d.x1, d.x2], d.y, { robust: true });
      if (tc === null) tc = L.tcrit(R.df, 0.05);
      b.push(R.b[1]); seU.push(R.se[1]); seR.push(R.seR[1]);
      if (Math.abs(R.b[1] + 1) <= tc * R.se[1]) cu++; if (Math.abs(R.b[1] + 1) <= tc * R.seR[1]) cr++;
      const e2 = R.res.map(v => v * v), aux = L.ols([d.x1, d.x2], e2); if (L.pChi(p.n * aux.r2, 2) < 0.05) bp++;
    }
    this.mc = { p, b, seU, seR, covU: cu / N, covR: cr / N, bp: bp / N };
  }
};
wire("het", het, ["h-g", "h-sig", "h-n"], "h");

// =====================================================================
// Least squares & R²
// =====================================================================
const ls = {
  seed: 1601, Z: null, last: null,
  P() { return { b1: num("s-b1"), sig: num("s-sig"), n: num("s-n"), ys: num("s-ys"), yh: num("s-yh"), show: $("s-show").checked }; },
  render() {
    const p = this.P(), Z = getZ(this, 2), n = p.n;
    $("s-b1o").textContent = f(p.b1, 1); $("s-sigo").textContent = f(p.sig, 1); $("s-yso").textContent = f(p.ys, 2);
    const x = new Float64Array(n), y = new Float64Array(n);
    for (let i = 0; i < n; i++) { x[i] = 10 + 2 * Z[0][i]; y[i] = 100 + p.b1 * x[i] + p.sig * Z[1][i]; }
    const xb = L.mean(x), yb = L.mean(y);
    let Sxy = 0, Sxx = 0, Syy = 0; for (let i = 0; i < n; i++) { Sxy += (x[i] - xb) * (y[i] - yb); Sxx += (x[i] - xb) ** 2; Syy += (y[i] - yb) ** 2; }
    const R = Syy > 1e-12 ? L.ols([x], y) : null;
    const b1 = Sxy / Sxx, b0 = yb - b1 * xb;
    const ssrAt = b => Syy - 2 * b * Sxy + b * b * Sxx;          // SSR of the line through the means with slope b
    const mine = ssrAt(p.ys) + n * p.yh * p.yh;
    const ssr = ssrAt(b1), sse = Syy - ssr, r = Syy > 1e-12 ? Sxy / Math.sqrt(Sxx * Syy) : NaN;
    $("s-yho").textContent = `${f(yb + p.yh, 2)} (d = ${f(p.yh, 2)})`;
    $("s-dgp").innerHTML = `True model: y = 100 <b>${sgn(p.b1)}</b>·x + u &nbsp;·&nbsp; x ~ N(10, 2²) &nbsp;·&nbsp; σ = <b>${p.sig.toFixed(1)}</b> &nbsp;·&nbsp; n = <b>${n}</b> &nbsp;·&nbsp; your line: ŷ = ${f(yb + p.yh - p.ys * xb, 2)} ${sgn(p.ys, 2)}·x`;
    this.last = { p, x, y, xb, yb, b0, b1, Sxx, Sxy, Syy, ssr, sse, mine, r, ssrAt, R };
    // bars: SST, SSE + SSR, yours
    const mx = Math.max(Syy, mine) || 1, w = v => (100 * v / mx).toFixed(2) + "%";
    $("s-bars").innerHTML =
      `<div class="bar"><span>SST = Σ(yᵢ − ȳ)²</span><span class="track"><span class="fill" style="width:${w(Syy)};background:var(--muted)"></span></span><span class="num">${f(Syy, 0)}</span></div>
       <div class="bar"><span>OLS: SSE + SSR</span><span class="track"><span class="fill" style="width:${w(sse)};background:var(--long)"></span><span class="fill" style="left:${w(sse)};width:${w(ssr)};background:var(--long);opacity:.35"></span></span><span class="num">${f(sse, 0)} + ${f(ssr, 0)}</span></div>
       <div class="bar"><span>Your line: SSR</span><span class="track"><span class="fill" style="width:${w(mine)};background:var(--short)"></span></span><span class="num">${f(mine, 0)}</span></div>`;
    const sx = Math.sqrt(Sxx / (n - 1)), sy = Math.sqrt(Syy / (n - 1)), cov = Sxy / (n - 1), ar = Math.abs(r);
    const strength = !isFinite(r) ? "" : ar === 0 ? "none" : ar <= 0.3 ? "weak" : ar <= 0.7 ? "moderate" : "strong";
    const sumRes = R ? R.res.reduce((a, v) => a + v, 0) : 0;
    $("s-calc").innerHTML = `<b>Least squares formulas</b><br>
      β̂₁ = Σ(xᵢ − x̄)(yᵢ − ȳ) / Σ(xᵢ − x̄)² = ${f(Sxy, 2)} / ${f(Sxx, 2)} = <span class="l">${f(b1, 4)}</span><br>
      β̂₀ = ȳ − β̂₁·x̄ = ${f(yb, 3)} − ${f(b1, 4)} × ${f(xb, 3)} = <span class="l">${f(b0, 3)}</span><br>
      Σûᵢ = ${f(Math.abs(sumRes) < 1e-8 ? 0 : sumRes, 6)} <span class="m">(OLS residuals always sum to zero)</span><br><br>
      <b>Goodness of fit</b><br>
      SST = SSE + SSR: ${f(Syy, 1)} = ${f(sse, 1)} + ${f(ssr, 1)}<br>
      r² = SSE / SST = ${f(sse, 1)} / ${f(Syy, 1)} = <span class="l">${f(sse / Syy, 4)}</span> <span class="m">→ x explains ${f(100 * sse / Syy, 1)}% of the variability of y</span><br>
      Cov(x, y) = ${f(cov, 3)} &nbsp;·&nbsp; r = Cov / (sₓ·s_y) = ${f(cov, 3)} / (${f(sx, 3)} × ${f(sy, 3)}) = <span class="l">${f(r, 4)}</span> <span class="m">(${strength})</span><br>
      r² = (${f(r, 4)})² = ${f(r * r, 4)} &nbsp;·&nbsp; r = β̂₁·sₓ/s_y = ${f(b1 * sx / sy, 4)}<br><br>
      <b>Your line</b>: SSR = ${f(mine, 1)} = ${f(ssrAt(p.ys), 1)} (slope) + ${n}·${f(p.yh, 2)}² (height) &nbsp;·&nbsp; ${mine <= ssr + 1e-6 ? `<span class="l">matches OLS</span>` : `<span class="s">${f(100 * (mine / ssr - 1), 1)}% worse than OLS</span>`}`;
    $("s-stata").textContent = R ? L.stataOut(R, "y", ["x"]) : "σ = 0 and β₁ = 0: y is constant, there is nothing to explain.";
    this.draw();
  },
  draw() {
    if (!this.last || !visible($("s-cvFit"))) return;
    const { p, x, y, xb, yb, b0, b1, ssr, mine, ssrAt } = this.last;
    const yl = v => yb + p.yh + p.ys * (v - xb);
    const P = L.chart($("s-cvFit"), L.range([x], 0.06), L.range([y], 0.12, [yl(Math.min(...x)), yl(Math.max(...x))]), { xlab: "x", ylab: "y" });
    L.clip(P, ctx => {
      ctx.strokeStyle = L.alpha(css("--short"), 0.55); ctx.lineWidth = 2;
      for (let i = 0; i < x.length && x.length <= 120; i++) { ctx.beginPath(); ctx.moveTo(P.X(x[i]), P.Y(y[i])); ctx.lineTo(P.X(x[i]), P.Y(yl(x[i]))); ctx.stroke(); }
    });
    L.dots(P, x, y, css("--dot"), { alpha: 0.85 });
    if (p.show) L.curve(P, v => b0 + b1 * v, css("--long"), { N: 2, width: 2.5 });
    L.curve(P, yl, css("--short"), { N: 2, width: 2.5 });
    P.ctx.fillStyle = css("--ink"); P.ctx.beginPath(); P.ctx.arc(P.X(xb), P.Y(yb), 5, 0, 7); P.ctx.fill();
    L.label(P, "(x̄, ȳ)", P.X(xb), P.Y(yb) + 6, css("--ink"));
    // SSR curve
    const lo = Math.min(p.ys, b1) - 3, hi = Math.max(p.ys, b1) + 3;
    const Q = L.chart($("s-cvSSR"), [lo, hi], [0, Math.max(ssrAt(lo), ssrAt(hi), mine) * 1.05], { xlab: "slope", ylab: "SSR" });
    L.curve(Q, ssrAt, css("--long"));
    L.vline(Q, b1, css("--long"), { width: 1.2, dash: [3, 3], label: "OLS β̂₁ = " + f(b1, 3), row: 1 });
    const c = Q.ctx; c.fillStyle = css("--long"); c.beginPath(); c.arc(Q.X(b1), Q.Y(ssr), 5, 0, 7); c.fill();
    c.strokeStyle = css("--short"); c.setLineDash([3, 3]); c.lineWidth = 1.5; c.beginPath(); c.moveTo(Q.X(p.ys), Q.Y(ssrAt(p.ys))); c.lineTo(Q.X(p.ys), Q.Y(mine)); c.stroke(); c.setLineDash([]);
    c.fillStyle = css("--short"); c.beginPath(); c.arc(Q.X(p.ys), Q.Y(mine), 6, 0, 7); c.fill();
    L.label(Q, "your SSR", Q.X(p.ys), Q.Y(mine) - 18, css("--short"));
  }
};
wire("ls", ls, ["s-b1", "s-sig", "s-n", "s-ys", "s-yh", "s-show"], "s");
$("s-snap").addEventListener("click", () => { const l = ls.last; if (!l) return; $("s-ys").value = l.b1; $("s-yh").value = 0; $("s-show").checked = true; ls.render(); });

// =====================================================================
// Logarithms
// =====================================================================
const FORMS = {
  ll: { name: "level-level", eq: b => `y = 20 ${sgn(b, 2)}·x + u`, rng: [-1, 3, 0.05, 1.5], sd: 3, logy: false, logx: false, b0: 20 },
  gl: { name: "log-level", eq: b => `ln y = 2 ${sgn(b, 3)}·x + u`, rng: [-0.1, 0.1, 0.005, 0.05], sd: 0.15, logy: true, logx: false, b0: 2 },
  lg: { name: "level-log", eq: b => `y = 20 ${sgn(b, 2)}·ln x + u`, rng: [-10, 20, 0.5, 10], sd: 3, logy: false, logx: true, b0: 20 },
  gg: { name: "log-log", eq: b => `ln y = 1 ${sgn(b, 2)}·ln x + u`, rng: [-1, 2, 0.05, 0.8], sd: 0.15, logy: true, logx: true, b0: 1 }
};
const logs = {
  seed: 2718, Z: null, last: null, form: "gg",
  P() { return { form: $("g-form").value, fit: $("g-fit").value, b1: num("g-b1"), k: num("g-noise"), n: num("g-n") }; },
  render() {
    let p = this.P();
    if (p.form !== this.form) {        // new true form: reset the slider to that form's scale
      const [a, b, st, v] = FORMS[p.form].rng, s = $("g-b1"); s.min = a; s.max = b; s.step = st; s.value = v; this.form = p.form; p = this.P();
    }
    const F = FORMS[p.form], Z = getZ(this, 2), n = p.n, sd = F.sd * p.k;
    const dec = F.rng[2] < 0.01 ? 3 : 2;
    $("g-b1o").textContent = f(p.b1, dec); $("g-noiseo").textContent = "σ = " + f(sd, 3);
    $("g-dgp").innerHTML = `True model (${F.name}): <b>${F.eq(p.b1)}</b> &nbsp;·&nbsp; x = 10·e^(0.5z), so x is positive and right-skewed &nbsp;·&nbsp; σ = <b>${f(sd, 3)}</b> &nbsp;·&nbsp; n = <b>${n}</b>`;
    const x = new Float64Array(n), y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      x[i] = 10 * Math.exp(0.5 * Z[0][i]);
      const rhs = F.b0 + p.b1 * (F.logx ? Math.log(x[i]) : x[i]) + sd * Z[1][i];
      y[i] = F.logy ? Math.exp(rhs) : rhs;
    }
    const pos = y.every(v => v > 0), lx = x.map(Math.log), ly = pos ? y.map(Math.log) : null;
    const fits = {};
    for (const k of ["ll", "gl", "lg", "gg"]) {
      const G = FORMS[k]; if (G.logy && !pos) { fits[k] = null; continue; }
      fits[k] = L.ols([G.logx ? lx : x], G.logy ? ly : y);
    }
    const trueMean = v => { const rhs = F.b0 + p.b1 * (F.logx ? Math.log(v) : v); return F.logy ? Math.exp(rhs + sd * sd / 2) : rhs; };
    this.last = { p, F, x, y, lx, ly, fits, pos, trueMean };
    // table
    const models = ["ll", "gl", "lg", "gg"].map(k => ({ title: FORMS[k].name, cls: k === p.fit ? "c-long" : "", R: fits[k], names: ["x"], note: "needs y > 0", dec: 4 }));
    $("g-tab").innerHTML = L.regTable(models, [["x", "β̂₁ (on x or ln x)"], ["_cons", "constant"]], [
      ["dependent variable", m => FORMS[["ll", "gl", "lg", "gg"][models.indexOf(m)]].logy ? "ln y" : "y"],
      ["R²", m => f(m.R.r2, 4)], ["Root MSE", m => f(m.R.rmse, 4)]]);
    // interpretation
    const R = fits[p.fit], G = FORMS[p.fit];
    if (!R) { $("g-interp").innerHTML = `<span class="r">Some y values are ≤ 0, so ln y does not exist. Lower the noise or pick a level-y model.</span>`; this.draw(); return; }
    const b = R.b[1], xm = L.mean(x), ym = L.mean(y);
    const yhat = v => { const rhs = R.b[0] + b * (G.logx ? Math.log(v) : v); return G.logy ? Math.exp(rhs + R.rmse ** 2 / 2) : rhs; };
    const elas = { ll: b * xm / yhat(xm), gl: b * xm, lg: b / yhat(xm), gg: b }[p.fit];
    const txt = {
      ll: `If x goes up by 1 unit, y changes by β̂₁ = <span class="l">${f(b, 3)}</span> units. &nbsp;Δy = β̂₁·Δx`,
      gl: `If x goes up by 1 unit, y changes by about 100·β̂₁ = <span class="l">${f(100 * b, 2)}%</span>. &nbsp;%Δy = 100·β̂₁·Δx <span class="m">(exact: 100·(e^β̂₁ − 1) = ${f(100 * (Math.exp(b) - 1), 2)}%)</span>`,
      lg: `If x goes up by 1%, y changes by β̂₁/100 = <span class="l">${f(b / 100, 4)}</span> units. &nbsp;Δy = (β̂₁/100)·%Δx`,
      gg: `If x goes up by 1%, y changes by about β̂₁ = <span class="l">${f(b, 3)}%</span>. &nbsp;%Δy = β̂₁·%Δx: β̂₁ is a constant elasticity`
    }[p.fit];
    const truth = p.fit === p.form ? `<span class="m">You are fitting the true functional form; true β₁ = ${f(p.b1, dec)}.</span>` :
      `<span class="s">The data come from a ${F.name} model, so this functional form is misspecified.</span>`;
    $("g-interp").innerHTML = `<b>${G.name}</b>: ${G.logy ? "ln y" : "y"} = ${f(R.b[0], 4)} ${sgn(b, 4)}·${G.logx ? "ln x" : "x"} &nbsp;(R² = ${f(R.r2, 4)})<br>
      ${txt}<br>
      Elasticity at the mean of x (x̄ = ${f(xm, 2)}): <span class="l">${f(elas, 3)}</span> <span class="m">${{ ll: "= β̂₁·x̄/ŷ, changes along the line", gl: "= β̂₁·x, grows with x", lg: "= β̂₁/ŷ, changes along the curve", gg: "= β̂₁, the same at every x" }[p.fit]}</span><br>
      ${truth}`;
    $("g-levnote").textContent = G.logy ? "For log-y models the curve is exp(fitted ln y) × e^(σ̂²/2): the usual adjustment, because exp(mean of ln y) understates the mean of y." : "";
    this.draw();
  },
  draw() {
    if (!this.last || !visible($("g-cvLev"))) return;
    const { p, x, y, lx, ly, fits, trueMean } = this.last, G = FORMS[p.fit], R = fits[p.fit];
    let P = L.chart($("g-cvLev"), L.range([x], 0.04), L.range([y], 0.08), { xlab: "x", ylab: "y" });
    L.dots(P, x, y, css("--dot"), { alpha: 0.55 });
    L.curve(P, trueMean, css("--ink"), { dash: [6, 4], width: 1.8 });
    if (R) L.curve(P, v => { const rhs = R.b[0] + R.b[1] * (G.logx ? Math.log(Math.max(v, 1e-9)) : v); return G.logy ? Math.exp(rhs + R.rmse ** 2 / 2) : rhs; }, css("--long"));
    const tx = G.logx ? lx : x, ty = G.logy ? ly : y;
    if (!R || !ty) { L.message($("g-cvTr"), "ln y is undefined for y ≤ 0."); return; }
    P = L.chart($("g-cvTr"), L.range([tx], 0.04), L.range([ty], 0.08), { xlab: G.logx ? "ln x" : "x", ylab: G.logy ? "ln y" : "y" });
    L.dots(P, tx, ty, css("--dot"), { alpha: 0.55 });
    L.curve(P, v => R.b[0] + R.b[1] * v, css("--long"), { N: 2 });
  }
};
wire("logs", logs, ["g-form", "g-fit", "g-b1", "g-noise", "g-n"], "g");

// =====================================================================
// Sampling distribution, CIs and the t-test
// =====================================================================
const ci = {
  seed: 1402, Z: null, mc: null, last: null,
  P() { return { b1: num("c-b1"), sig: num("c-sig"), n: num("c-n"), lev: num("c-lev") }; },
  xs(n) { const Z = getZ(this, 2); return Z[0].subarray(0, n).map(z => 10 + 2 * z); },
  ys(x, p, e) { return x.map((v, i) => 100 + p.b1 * v + p.sig * e[i]); },
  render() {
    const p = this.P(), Z = getZ(this, 2), n = p.n, x = this.xs(n), y = this.ys(x, p, Z[1]);
    $("c-b1o").textContent = f(p.b1, 1); $("c-sigo").textContent = f(p.sig, 1);
    $("c-dgp").innerHTML = `True model: y = 100 <b>${sgn(p.b1)}</b>·x + u, &nbsp;u ~ N(0, σ²) &nbsp;·&nbsp; x ~ N(10, 2²) &nbsp;·&nbsp; σ = <b>${p.sig.toFixed(1)}</b> &nbsp;·&nbsp; n = <b>${n}</b>`;
    const R = L.ols([x], y), SSTx = L.sst(x), a = 1 - p.lev, tc = L.tcrit(R.df, a), lv = Math.round(100 * p.lev);
    const lo = R.b[1] - tc * R.se[1], hi = R.b[1] + tc * R.se[1], sdTrue = p.sig / Math.sqrt(SSTx);
    const others = []; for (let r = 0; r < 30; r++) { const e = L.normals(this.seed + 500 + r, n, 1)[0]; others.push(L.ols([x], this.ys(x, p, e)).b); }
    this.last = { p, x, y, R, SSTx, tc, lo, hi, sdTrue, others, lv };
    const rej = Math.abs(R.t[1]) > tc;
    $("c-calc").innerHTML = `<b>Standard error</b><br>
      σ̂ = √[SSR/(n − 2)] = √(${f(R.ssr, 1)} / ${R.df}) = ${f(R.rmse, 3)} <span class="m">(Root MSE)</span><br>
      se(β̂₁) = σ̂ / √Σ(xᵢ − x̄)² = ${f(R.rmse, 3)} / √${f(SSTx, 2)} = <span class="l">${f(R.se[1], 4)}</span> <span class="m">(true sd(β̂₁) = σ/√SSTₓ = ${f(sdTrue, 4)})</span><br><br>
      <b>${lv}% confidence interval</b><br>
      β̂₁ ± t·se(β̂₁) = ${f(R.b[1], 3)} ± ${f(tc, 3)} × ${f(R.se[1], 4)} = <span class="l">[${f(lo, 3)}, ${f(hi, 3)}]</span> <span class="m">(t with df = n − 2 = ${R.df})</span><br>
      <span class="m">${lo <= p.b1 && p.b1 <= hi ? "This interval contains the true β₁." : "This interval misses the true β₁: it happens to " + (100 - lv) + "% of intervals."}</span><br><br>
      <b>t-test of H₀: β₁ = 0</b>, α = ${f(a, 2)}<br>
      t = β̂₁ / se(β̂₁) = ${f(R.b[1], 3)} / ${f(R.se[1], 4)} = <span class="l">${f(R.t[1], 3)}</span> &nbsp;·&nbsp; critical |t| = ${f(tc, 3)} &nbsp;·&nbsp; p = ${fp(R.p[1])} &nbsp;${verdict(rej)}`;
    $("c-stata").textContent = L.stataOut(R, "y", ["x"], p.lev);
    this.draw();
  },
  draw() {
    if (!this.last || !visible($("c-cvFit"))) return;
    const { p, x, y, R, lo, hi, sdTrue, others } = this.last;
    let P = L.chart($("c-cvFit"), L.range([x], 0.06), L.range([y], 0.12), { xlab: "x", ylab: "y" });
    for (const b of others) L.curve(P, v => b[0] + b[1] * v, L.alpha(css("--muted"), 0.45), { N: 2, width: 1 });
    L.dots(P, x, y, css("--dot"), { alpha: 0.85 });
    L.curve(P, v => 100 + p.b1 * v, css("--ink"), { N: 2, width: 1.8, dash: [6, 4] });
    L.curve(P, v => R.b[0] + R.b[1] * v, css("--long"), { N: 2 });
    // sampling distribution
    const w = 4.2 * Math.max(sdTrue, R.se[1]), xr = [Math.min(p.b1 - w, lo), Math.max(p.b1 + w, hi)];
    const pdf = v => Math.exp(-0.5 * ((v - p.b1) / sdTrue) ** 2) / (sdTrue * Math.sqrt(2 * Math.PI));
    const ymax = pdf(p.b1) * 1.25;
    P = L.chart($("c-cvDist"), xr, [0, ymax], { xlab: "β̂₁", ylab: "density" });
    if (this.mc) { const bins = 36, bw = (xr[1] - xr[0]) / bins; L.hist(P, this.mc.b, bins, css("--long"), { scale: 1 / (this.mc.b.length * bw), alpha: 0.45 }); }
    L.curve(P, pdf, css("--ink"), { width: 2 });
    L.vline(P, p.b1, css("--ink"), { width: 1.2, dash: [3, 3], label: "true β₁", row: 0 });
    const c = P.ctx, yy = P.Y(ymax * 0.06);
    c.strokeStyle = css("--short"); c.lineWidth = 5; c.lineCap = "round"; c.beginPath(); c.moveTo(P.X(lo), yy); c.lineTo(P.X(hi), yy); c.stroke(); c.lineCap = "butt";
    L.vline(P, R.b[1], css("--short"), { width: 2.5, label: "β̂₁ = " + f(R.b[1], 3), row: 1 });
    // MC stats
    if (!this.mc) { $("c-mcstats").innerHTML = `<div><span class="k">Press “Run 1,000 samples” to fill the histogram.</span></div>`; return; }
    const m = this.mc;
    $("c-mcstats").innerHTML = stat("mean β̂₁ (true " + f(m.p.b1, 1) + ")", f(L.mean(m.b), 4)) + stat("SD of β̂₁ across samples", f(L.sd(m.b), 4)) +
      stat("σ / √SSTₓ (theory)", f(m.sdTrue, 4)) + stat("average se(β̂₁)", f(L.mean(m.se), 4)) +
      stat(`${Math.round(100 * m.p.lev)}% CIs containing true β₁`, pct(m.cover)) + stat("rejected H₀: β₁ = 0", pct(m.rej) + (m.p.b1 === 0 ? " (H₀ true: ≈ α)" : " (H₀ false: power)"));
  },
  runMC() {
    const p = this.P(), n = p.n, x = this.xs(n), N = 1000, b = [], se = []; let cover = 0, rej = 0, tc = null;
    for (let r = 0; r < N; r++) {
      const R = L.ols([x], this.ys(x, p, L.normals(this.seed + 5000 + r, n, 1)[0]));
      if (tc === null) tc = L.tcrit(R.df, 1 - p.lev);
      b.push(R.b[1]); se.push(R.se[1]);
      if (Math.abs(R.b[1] - p.b1) <= tc * R.se[1]) cover++; if (Math.abs(R.t[1]) > tc) rej++;
    }
    this.mc = { p, b, se, cover: cover / N, rej: rej / N, sdTrue: p.sig / Math.sqrt(L.sst(x)) };
  }
};
wire("ci", ci, ["c-b1", "c-sig", "c-n", "c-lev"], "c");

// =====================================================================
// ANOVA, F-test and prediction
// =====================================================================
const anova = {
  seed: 1714, Z: null, last: null,
  P() { return { b1: num("a-b1"), sig: num("a-sig"), n: num("a-n"), a: num("a-a"), xs: num("a-xs") }; },
  render() {
    const p = this.P(), Z = getZ(this, 2), n = p.n;
    $("a-b1o").textContent = f(p.b1, 1); $("a-sigo").textContent = f(p.sig, 1); $("a-xso").textContent = f(p.xs, 2);
    $("a-dgp").innerHTML = `True model: y = 100 <b>${sgn(p.b1)}</b>·x + u &nbsp;·&nbsp; x ~ N(10, 2²) &nbsp;·&nbsp; σ = <b>${p.sig.toFixed(1)}</b> &nbsp;·&nbsp; n = <b>${n}</b>`;
    const x = new Float64Array(n), y = new Float64Array(n);
    for (let i = 0; i < n; i++) { x[i] = 10 + 2 * Z[0][i]; y[i] = 100 + p.b1 * x[i] + p.sig * Z[1][i]; }
    const R = L.ols([x], y), xb = L.mean(x), SSTx = L.sst(x), k = 1, df = n - 2;
    const MSE = R.sse / k, MSR = R.ssr / df, MST = R.sst / (n - 1), F = MSE / MSR, Fc = L.Fcrit(1, df, p.a), pF = L.pF(F, 1, df);
    $("a-anova").className = "anova";
    $("a-anova").innerHTML = `<thead><tr><th scope="col"></th><th scope="col">Sum of squares</th><th scope="col">df</th><th scope="col">Mean square = SS/df</th></tr></thead><tbody>
      <tr><td>Explained (Stata: Model)</td><td>SSE = ${f(R.sse, 2)}</td><td>k = ${k}</td><td>${f(MSE, 2)}</td></tr>
      <tr><td>Unexplained (Stata: Residual)</td><td>SSR = ${f(R.ssr, 2)}</td><td>n − k − 1 = ${df}</td><td>${f(MSR, 2)} <span class="se">= σ̂², so σ̂ = ${f(R.rmse, 3)}</span></td></tr>
      <tr><td>Total</td><td>SST = ${f(R.sst, 2)}</td><td>n − 1 = ${n - 1}</td><td>${f(MST, 2)} <span class="se">= variance of y</span></td></tr></tbody>`;
    $("a-fcalc").innerHTML = `F = [SSE/k] / [SSR/(n − k − 1)] = ${f(MSE, 2)} / ${f(MSR, 2)} = <span class="l">${f(F, 3)}</span><br>
      F = [r²/k] / [(1 − r²)/(n − k − 1)] = (${f(R.r2, 4)}/1) / (${f(1 - R.r2, 4)}/${df}) = ${f((R.r2) / ((1 - R.r2) / df), 3)}<br>
      df₁ = ${k}, df₂ = ${df} &nbsp;·&nbsp; critical F at α = ${p.a}: <span class="r">${f(Fc, 3)}</span> &nbsp;·&nbsp; p = ${fp(pF)} &nbsp;${verdict(F > Fc)}<br>
      √F = ${f(Math.sqrt(F), 3)} = |t| on the slope (${f(Math.abs(R.t[1]), 3)}): <span class="m">with one regressor the F-test and the t-test are identical.</span>`;
    const t1 = R.rmse ** 2, t2 = t1 / n, t3 = t1 * (p.xs - xb) ** 2 / SSTx, seP = Math.sqrt(t1 + t2 + t3), seM = Math.sqrt(t2 + t3), tc = L.tcrit(df, 0.05);
    const yh = R.b[0] + R.b[1] * p.xs;
    $("a-pcalc").innerHTML = `ŷ* = β̂₀ + β̂₁·x* = ${f(R.b[0], 3)} ${sgn(R.b[1], 3)} × ${f(p.xs, 2)} = <span class="l">${f(yh, 3)}</span><br>
      se(pred) = σ̂·√[1 + 1/n + (x* − x̄)²/SSTₓ] = ${f(R.rmse, 3)} × √[1 + 1/${n} + (${f(p.xs, 2)} − ${f(xb, 2)})²/${f(SSTx, 2)}] = <span class="l">${f(seP, 3)}</span><br>
      Var(pred) = σ̂² + σ̂²/n + σ̂²(x* − x̄)²/SSTₓ = ${f(t1, 2)} + ${f(t2, 2)} + ${f(t3, 2)}<br>
      95% interval for the new observation: ${f(yh, 2)} ± ${f(tc, 3)} × ${f(seP, 3)} = <span class="l">[${f(yh - tc * seP, 2)}, ${f(yh + tc * seP, 2)}]</span><br>
      <span class="m">95% CI for the mean E[y | x*] drops the σ̂² term: [${f(yh - tc * seM, 2)}, ${f(yh + tc * seM, 2)}]</span>`;
    this.last = { p, x, y, R, xb, SSTx, F, Fc, df, t1, t2, t3, tc, yh, seP };
    this.draw();
  },
  draw() {
    if (!this.last || !visible($("a-cvFit"))) return;
    const { p, x, y, R, xb, SSTx, F, Fc, df, t1, t2, t3, tc, yh, seP } = this.last, s2 = R.rmse ** 2, n = x.length;
    const fit = v => R.b[0] + R.b[1] * v, vm = v => s2 / n + s2 * (v - xb) ** 2 / SSTx;
    let P = L.chart($("a-cvFit"), L.range([x], 0.06, [p.xs]), L.range([y], 0.25), { xlab: "x", ylab: "y" });
    const lc = css("--long");
    L.band(P, v => fit(v) - tc * Math.sqrt(s2 + vm(v)), v => fit(v) + tc * Math.sqrt(s2 + vm(v)), L.alpha(lc, 0.12));
    L.band(P, v => fit(v) - tc * Math.sqrt(vm(v)), v => fit(v) + tc * Math.sqrt(vm(v)), L.alpha(lc, 0.35));
    L.dots(P, x, y, css("--dot"), { alpha: 0.8 });
    L.curve(P, fit, lc, { N: 2 });
    L.vline(P, p.xs, css("--short"), { width: 1.5, dash: [4, 3], label: "x*" });
    L.vline(P, xb, css("--muted"), { width: 1, dash: [2, 3], label: "x̄", row: 1 });
    const c = P.ctx; c.strokeStyle = css("--short"); c.lineWidth = 4; c.beginPath(); c.moveTo(P.X(p.xs), P.Y(yh - tc * seP)); c.lineTo(P.X(p.xs), P.Y(yh + tc * seP)); c.stroke();
    c.fillStyle = css("--short"); c.beginPath(); c.arc(P.X(p.xs), P.Y(yh), 5, 0, 7); c.fill();
    // F distribution
    const fmax = Math.max(8, Fc * 1.8, Math.min(F * 1.15, 60)), fpdf = v => L.Fpdf(Math.max(v, 0.02), 1, df);
    P = L.chart($("a-cvF"), [0, fmax], [0, Math.min(1.6, fpdf(0.15)) * 1.05], { xlab: `F under H₀ (1, ${df})`, ylab: "density" });
    L.area(P, fpdf, Fc, fmax, L.alpha(css("--reject"), 0.35)); L.curve(P, fpdf, css("--ink"), { width: 2 });
    L.vline(P, Fc, css("--reject"), { width: 1.2, dash: [3, 3], label: "critical " + f(Fc, 2), row: 1 });
    if (F <= fmax) L.vline(P, F, lc, { width: 2.5, label: "F = " + f(F, 2) }); else L.label(P, `F = ${f(F, 1)} (off the chart →)`, P.w - P.m.r, P.m.t + 2, lc, "right");
    // variance decomposition bar
    const tot = t1 + t2 + t3;
    P = L.chart($("a-cvVar"), [0, tot], [0, 1], { noY: true, xlab: "Var(pred) = σ̂² + σ̂²/n + σ̂²(x* − x̄)²/SSTₓ", nx: 4 });
    let acc = 0; const h0 = P.Y(0.78), h1 = P.Y(0.22);
    for (const [v, col] of [[t1, css("--muted")], [t2, css("--short")], [t3, lc]]) { P.ctx.fillStyle = col; P.ctx.fillRect(P.X(acc), h0, P.X(acc + v) - P.X(acc), h1 - h0); acc += v; }
  }
};
wire("anova", anova, ["a-b1", "a-sig", "a-n", "a-a", "a-xs"], "a");

// =====================================================================
// Topic TOC, resizing, theme
// =====================================================================
const keys = ["ls", "ovb", "vif", "ci", "anova", "inf", "logs", "form", "het"];
const btns = keys.map(k => $("toc-" + k));
keys.forEach((k, i) => {                       // pane heading from the TOC entry
  const b = btns[i], chap = b.closest(".chap"), h = document.createElement("h2");
  h.className = "pane-title"; h.innerHTML = `<span class="eyebrow">${chap.querySelector(".cht").textContent}</span>${b.innerHTML}`;
  $("pane-" + k).prepend(h);
});
let active = "ls";
function show(k) {
  if (!modules[k]) k = "ls"; active = k;
  keys.forEach((kk, i) => { const on = kk === k; if (on) btns[i].setAttribute("aria-current", "page"); else btns[i].removeAttribute("aria-current"); $("pane-" + kk).hidden = !on; });
  try { localStorage.setItem("olsSandboxTab", k); } catch (e) {}
  try { history.replaceState(null, "", "#" + k); } catch (e) {}
  if (!modules[k].last) modules[k].render(); else modules[k].draw();
}
btns.forEach((b, i) => b.addEventListener("click", () => {
  show(keys[i]);
  if (window.matchMedia("(max-width: 1180px)").matches) $("pane-" + keys[i]).scrollIntoView({ block: "start" });
}));
const redraw = () => modules[active] && modules[active].draw();
let raf = 0; const soon = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(redraw); };
const ro = new ResizeObserver(soon); document.querySelectorAll(".pane main").forEach(el => ro.observe(el));
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", soon);
new MutationObserver(soon).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
if (document.fonts && document.fonts.ready) document.fonts.ready.then(soon);

let start = (location.hash || "").replace("#", "");
if (!modules[start]) { try { start = localStorage.getItem("olsSandboxTab") || "ls"; } catch (e) { start = "ls"; } }
show(modules[start] ? start : "ls");
})();
