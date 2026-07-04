#!/usr/bin/env python
"""build_explorer.py — Genera defense/assets/explorer.html (T4, Sprint 6).

HTML autocontenido y OFFLINE (sin build; Chart.js vendorizado e incrustado inline, sin CDN) con
3 vistas cableadas 100% a CSVs REALES del repo. LEE los CSV canónicos, los EMBEBE verbatim en el
HTML (máxima trazabilidad: el dato es literalmente el CSV), y VALIDA cada valor antes de
escribir. Si algo no cuadra (F1 fuera de [0,1], celda vacía, punto Pareto que no reproduce la
Tabla III), LEVANTA excepción y NO escribe — cero datos inventados.

Fuentes:
  Vista 1 (Pareto Corte B):  results/summary_corteB.csv           (+ robertuito JSON, referencia)
  Vista 2 (cruce Corte A):   results/summary_corteA.csv
  Vista 3 (transfer Corte C):results/sprint4/transfer_matrix_qwen1.7b.csv y _qwen4b.csv

Uso: python scripts/build_explorer.py
"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "defense" / "assets" / "explorer.html"

B_PATH = ROOT / "results" / "summary_corteB.csv"
A_PATH = ROOT / "results" / "summary_corteA.csv"
TR17_PATH = ROOT / "results" / "sprint4" / "transfer_matrix_qwen1.7b.csv"
TR4B_PATH = ROOT / "results" / "sprint4" / "transfer_matrix_qwen4b.csv"
ROB_PATH = ROOT / "results" / "baselines" / "robertuito_offtheshelf.json"
CHARTJS_PATH = ROOT / "defense" / "assets" / "chart.umd.min.js"  # vendorizado (sin CDN)

# Objetivos de la Tabla III (solo para verificar la derivación; no se teclean en el HTML)
TARGETS = {
    "LoRA 4B": 0.707, "LoRA 1.7B": 0.698, "QLoRA 4B": 0.701, "QLoRA 1.7B": 0.681,
    "Full-FT 1.7B": 0.696, "BETO": 0.661, "XLM-R": 0.646, "Prompting k=4": 0.652,
}


def die(msg: str):
    print(f"\n[PARA] {msg}", file=sys.stderr)
    sys.exit(1)


def read(p: Path) -> str:
    if not p.exists():
        die(f"No existe la fuente: {p}")
    return p.read_text(encoding="utf-8")


def rows(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def f(x) -> float:
    return float(x)


def in01(x: float) -> bool:
    return 0.0 <= x <= 1.0


# --------------------------------------------------------------------------- #
# Validación / derivación en Python (espejo de lo que hará el JS)
# --------------------------------------------------------------------------- #

def validate_and_summarize():
    B = rows(read(B_PATH))
    A = rows(read(A_PATH))
    T17 = rows(read(TR17_PATH))
    T4 = rows(read(TR4B_PATH))
    rob = json.loads(read(ROB_PATH))

    by_name = {r["experiment_name"]: r for r in B}

    def mean_f1(names):
        return sum(f(by_name[n]["f1_macro"]) for n in names) / len(names)

    def mean_col(names, col):
        return sum(f(by_name[n][col]) for n in names) / len(names)

    lora17 = ["lora_qwen1.7b_nfull_s42", "lora_qwen1.7b_nfull_s43", "lora_qwen1.7b_nfull_s44"]
    lora4b = ["lora_qwen4b_nfull_s42", "lora_qwen4b_nfull_s43", "lora_qwen4b_nfull_s44"]

    pts = [
        ("LoRA 4B",     "4b",  "lora",    mean_f1(lora4b), 11796480, mean_col(lora4b, "peak_vram_gb")),
        ("LoRA 1.7B",   "17b", "lora",    mean_f1(lora17), 6422528,  mean_col(lora17, "peak_vram_gb")),
        ("QLoRA 4B",    "4b",  "qlora",   f(by_name["qlora_qwen4b_nfull_s42"]["f1_macro"]),   11796480, f(by_name["qlora_qwen4b_nfull_s42"]["peak_vram_gb"])),
        ("QLoRA 1.7B",  "17b", "qlora",   f(by_name["qlora_qwen1.7b_nfull_s42"]["f1_macro"]), 6422528,  f(by_name["qlora_qwen1.7b_nfull_s42"]["peak_vram_gb"])),
        ("Full-FT 1.7B","17b", "full_ft", f(by_name["full_ft_qwen1.7b_nfull_s42"]["f1_macro"]), 1720574976, f(by_name["full_ft_qwen1.7b_nfull_s42"]["peak_vram_gb"])),
        ("BETO",        "enc", "encoder", f(by_name["encoder_finetuned_dccuchile_bert-base-spanish-wwm-cased_cardiff_es"]["f1_macro"]), 109853187, f(by_name["encoder_finetuned_dccuchile_bert-base-spanish-wwm-cased_cardiff_es"]["peak_vram_gb"])),
        ("XLM-R",       "enc", "encoder", f(by_name["encoder_finetuned_xlm-roberta-base_cardiff_es"]["f1_macro"]), 278045955, f(by_name["encoder_finetuned_xlm-roberta-base_cardiff_es"]["peak_vram_gb"])),
        ("Prompting k=4","prompt","prompting", f(by_name["prompting_fewshot_k4_Qwen_Qwen3-4B"]["f1_macro"]), 0, f(by_name["prompting_fewshot_k4_Qwen_Qwen3-4B"]["peak_vram_gb"])),
    ]

    print("== VISTA 1 — 8 puntos Pareto (derivados de summary_corteB.csv) ==")
    print(f"{'punto':<14}{'F1':>9}{'objetivo':>10}{'params':>13}{'VRAM(GB)':>11}")
    for label, fam, meth, f1v, params, vram in pts:
        if not in01(f1v):
            die(f"F1 fuera de [0,1] en {label}: {f1v}")
        tgt = TARGETS[label]
        ok = abs(round(f1v, 3) - tgt) <= 0.001
        flag = "" if ok else f"  <-- NO CUADRA (obj {tgt})"
        print(f"{label:<14}{f1v:>9.4f}{tgt:>10.3f}{params:>13,}{vram:>11.3f}{flag}")
        if not ok:
            die(f"El punto {label} F1={f1v:.4f} no reproduce la Tabla III ({tgt}).")
    vram_lora4b = pts[0][5]
    print(f"\n  VRAM exacta LoRA-4B (media 3 semillas) = {vram_lora4b:.4f} GB "
          f"(s42={f(by_name['lora_qwen4b_nfull_s42']['peak_vram_gb']):.4f})")

    # Pareto por eje (minimiza coste x, maximiza F1)
    def pareto(items, xi):
        front = [p for p in items if not any(
            q is not p and q[xi] <= p[xi] and q[3] >= p[3] and (q[xi] < p[xi] or q[3] > p[3])
            for q in items)]
        return sorted(front, key=lambda p: p[xi])
    fp = pareto(pts, 4)  # params index=4
    fv = pareto(pts, 5)  # vram index=5
    print("  Frontera Pareto (params):", " -> ".join(p[0] for p in fp))
    print("  Frontera Pareto (VRAM)  :", " -> ".join(p[0] for p in fv))

    rob_f1 = f(rob["f1_macro"])
    if not in01(rob_f1):
        die(f"RoBERTuito F1 fuera de [0,1]: {rob_f1}")
    print(f"  Referencia RoBERTuito (fuera de dominio) = {rob_f1:.4f}")

    # -- Vista 2 --
    print("\n== VISTA 2 — curvas Corte A + cruces (derivados de summary_corteA.csv) ==")
    def curve(model):
        c = sorted(((int(r["n_train"]), f(r["f1_mean"]), f(r["f1_std"]))
                    for r in A if r["model_short"] == model), key=lambda t: t[0])
        for n, m, s in c:
            if not in01(m):
                die(f"F1_mean fuera de [0,1] en {model} n={n}: {m}")
        return c
    c17, c4 = curve("Qwen3-1.7B"), curve("Qwen3-4B")

    def best_prompt(modeltag):
        vals = [f(r["f1_macro"]) for r in B if r["method"].startswith("prompting")
                and r["model_short"] == modeltag]
        return max(vals)
    floor17, floor4 = best_prompt("Qwen3-1.7B"), best_prompt("Qwen3-4B")
    beto = f(by_name["encoder_finetuned_dccuchile_bert-base-spanish-wwm-cased_cardiff_es"]["f1_macro"])
    xlmr = f(by_name["encoder_finetuned_xlm-roberta-base_cardiff_es"]["f1_macro"])

    def crossover(c, floor):
        for n, m, s in c:
            if m > floor:
                return n
        return None
    x17, x4 = crossover(c17, floor17), crossover(c4, floor4)
    print(f"  suelo prompting 1.7B={floor17:.4f}  4B={floor4:.4f} | encoders BETO={beto:.4f} XLM-R={xlmr:.4f}")
    print(f"  CRUCE (1er n con LoRA_media > suelo): 1.7B -> n={x17}   4B -> n={x4}")
    print(f"  (n disponibles: {[n for n,_,_ in c17]})")

    # -- Vista 3 --
    print("\n== VISTA 3 — matrices de transferencia (Corte C) ==")
    for tag, T in [("1.7B", T17), ("4B", T4)]:
        if len(T) != 9:
            die(f"Matriz {tag}: se esperaban 9 celdas, hay {len(T)}")
        for r in T:
            if r["f1_macro_mean"] == "" or r["f1_macro_std"] == "":
                die(f"Celda vacía en matriz {tag}: {r}")
            if not in01(f(r["f1_macro_mean"])):
                die(f"F1 fuera de [0,1] en matriz {tag}: {r}")
        vals = {(r["train"], r["eval"]): f(r["f1_macro_mean"]) for r in T}
        into_es = [vals[(t, "ES")] for t in ("CR", "PE")]
        diag = [vals[(d, d)] for d in ("ES", "CR", "PE")]
        print(f"  {tag}: 9/9 celdas OK, F1∈[{min(vals.values()):.4f},{max(vals.values()):.4f}]; "
              f"hacia ES (off-diag)={[round(x,4) for x in into_es]} vs diagonal={[round(x,4) for x in diag]}")

    print("\n[OK] Todos los valores cuadran (F1∈[0,1], sin celdas vacías, 8 puntos = Tabla III).")
    return dict(rob_f1=rob_f1)


# --------------------------------------------------------------------------- #
# Plantilla HTML (los CSV se inyectan verbatim; JS parsea y dibuja)
# --------------------------------------------------------------------------- #

def build_html(ctx: dict) -> str:
    tpl = HTML_TEMPLATE
    tpl = tpl.replace("__CORTE_B_CSV__", read(B_PATH).strip())
    tpl = tpl.replace("__CORTE_A_CSV__", read(A_PATH).strip())
    tpl = tpl.replace("__TR17_CSV__", read(TR17_PATH).strip())
    tpl = tpl.replace("__TR4B_CSV__", read(TR4B_PATH).strip())
    tpl = tpl.replace("__ROBERTUITO_F1__", repr(ctx["rob_f1"]))
    # Chart.js vendorizado, incrustado INLINE (explorer.html abre offline con doble clic).
    chartjs = read(CHARTJS_PATH)
    if "</script" in chartjs.lower():
        die("chart.umd.min.js contiene '</script>'; no es seguro incrustarlo inline.")
    if "Chart" not in chartjs or len(chartjs) < 100_000:
        die(f"chart.umd.min.js parece incompleto ({len(chartjs)} bytes).")
    tpl = tpl.replace("__CHARTJS__", chartjs)
    return tpl


def main():
    ctx = validate_and_summarize()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_html(ctx), encoding="utf-8")
    print(f"\nEscrito: {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


# El template vive al final para no estorbar la lógica.
HTML_TEMPLATE = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Explorador — Ajuste eficiente de LLMs abiertos en español</title>
<script>/* Chart.js v4.4.4 — vendorizado e incrustado inline (sin CDN; abre offline) */
__CHARTJS__
</script>
<style>
  :root{ --b17:#1f4e9c; --b17l:#7aa0d6; --r4:#b5322f; --r4l:#d98b89; --navy:#12306b;
         --grey:#6b7280; --greyl:#9ca3af; --slate:#334155; --ink:#1f2933; --line:#e5e7eb; }
  *{box-sizing:border-box}
  body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);
       margin:0;background:#fafafa;line-height:1.45}
  header{padding:18px 22px 6px}
  header h1{font-size:19px;margin:0 0 3px}
  header p{margin:0;color:var(--grey);font-size:13px}
  .tabs{display:flex;gap:2px;padding:0 22px;border-bottom:1px solid var(--line);margin-top:10px}
  .tab{border:none;background:none;padding:10px 16px;font-size:14px;color:var(--grey);
       cursor:pointer;border-bottom:2px solid transparent}
  .tab.active{color:var(--ink);border-bottom-color:var(--ink);font-weight:600}
  main{padding:16px 22px 40px;max-width:980px}
  .view{display:none} .view.active{display:block}
  .view h2{font-size:16px;margin:6px 0 2px}
  .view .sub{color:var(--grey);font-size:13px;margin:0 0 12px}
  .controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:6px 0 12px}
  .controls .lbl{font-size:13px;color:var(--grey);margin-right:2px}
  .seg{display:inline-flex;border:1px solid var(--line);border-radius:7px;overflow:hidden}
  .seg button{border:none;background:#fff;padding:6px 12px;font-size:13px;cursor:pointer;color:var(--slate)}
  .seg button.active{background:var(--ink);color:#fff}
  .card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px}
  .chartwrap{position:relative;height:460px}
  .foot{font-size:11.5px;color:var(--grey);margin-top:10px}
  .foot code{background:#f3f4f6;padding:1px 4px;border-radius:3px}
  .legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--slate);margin:8px 0 0}
  .legend span{display:inline-flex;align-items:center;gap:5px}
  .dot{width:11px;height:11px;border-radius:50%;display:inline-block}
  /* Heatmap Vista 3 */
  .hm{display:grid;grid-template-columns:60px repeat(3,1fr);gap:4px;max-width:520px}
  .hm .h{font-size:12px;color:var(--grey);display:flex;align-items:center;justify-content:center;padding:4px}
  .hm .rowh{font-weight:600;color:var(--slate)}
  .cell{position:relative;border-radius:6px;padding:14px 6px;text-align:center;color:#fff;
        font-weight:600;font-size:15px;cursor:default}
  .cell small{display:block;font-weight:400;font-size:11px;opacity:.9}
  .cell.diag{outline:3px solid #111;outline-offset:-3px}
  .hm .colh.es{color:var(--r4);font-weight:700}
  .tip{position:fixed;pointer-events:none;background:#111;color:#fff;font-size:12px;padding:6px 8px;
       border-radius:6px;opacity:0;transition:opacity .1s;z-index:10;max-width:220px}
  .note{background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:9px 12px;font-size:12.5px;color:#9a3412;margin:10px 0}
</style>
</head>
<body>
<header>
  <h1>Explorador de resultados — ajuste eficiente de LLMs abiertos en español</h1>
  <p>Datos reales del repo (cero cifras inventadas). LoRA/QLoRA = medias de 3 semillas {42,43,44} donde aplica. <span style="color:var(--b17)">azul = 1.7B</span> · <span style="color:var(--r4)">rojo = 4B</span> · gris = encoders/prompting.</p>
</header>
<nav class="tabs">
  <button class="tab active" data-v="1">1 · Frontera de Pareto (Corte B)</button>
  <button class="tab" data-v="2">2 · Cruce F1-vs-n (Corte A)</button>
  <button class="tab" data-v="3">3 · Transferencia dialectal (Corte C)</button>
</nav>
<main>
  <!-- VISTA 1 -->
  <section class="view active" id="v1">
    <h2>Frontera de Pareto calidad–coste</h2>
    <p class="sub">F1-macro frente al coste. LoRA domina la frontera. RoBERTuito (0.758) es <b>fuera de dominio</b> y no entra en el Pareto.</p>
    <div class="controls">
      <span class="lbl">Eje de coste (log):</span>
      <span class="seg" id="v1seg">
        <button data-x="params" class="active">parámetros entrenables</button>
        <button data-x="vram">VRAM pico (GB)</button>
      </span>
    </div>
    <div class="card"><div class="chartwrap"><canvas id="c1"></canvas></div>
      <div class="legend" id="leg1"></div>
    </div>
    <div class="note" id="v1note"></div>
    <p class="foot">Fuente: <code>results/summary_corteB.csv</code> (8 configs de la Tabla III). LoRA = media de {42,43,44}; QLoRA/Full-FT/encoders/prompting = semilla 42. Referencia: <code>results/baselines/robertuito_offtheshelf.json</code>.</p>
  </section>

  <!-- VISTA 2 -->
  <section class="view" id="v2">
    <h2>Curva de cruce F1-vs-n</h2>
    <p class="sub">¿Desde cuántos ejemplos LoRA supera al mejor prompting? El cruce depende del tamaño del modelo.</p>
    <div class="controls">
      <span class="lbl">Modelo:</span>
      <span class="seg" id="v2seg">
        <button data-m="both" class="active">ambos</button>
        <button data-m="17b">Qwen3-1.7B</button>
        <button data-m="4b">Qwen3-4B</button>
      </span>
    </div>
    <div class="card"><div class="chartwrap"><canvas id="c2"></canvas></div>
      <div class="legend" id="leg2"></div>
    </div>
    <div class="note" id="v2note"></div>
    <p class="foot">Fuente: <code>results/summary_corteA.csv</code> (f1_mean ± f1_std, medias de {42,43,44}; n=1839 = "full"). Suelos de prompting y anclas de encoders de <code>results/summary_corteB.csv</code>. El punto de cruce se calcula del dato: primer n con LoRA(media) &gt; suelo de prompting.</p>
  </section>

  <!-- VISTA 3 -->
  <section class="view" id="v3">
    <h2>Matriz de transferencia dialectal</h2>
    <p class="sub">Entrenar en una variedad (fila) y evaluar en otra (columna). La <b>columna ES</b> es la penalizada: transferir <b>hacia</b> ES cuesta; hacia CR/PE, casi gratis.</p>
    <div class="controls">
      <span class="lbl">Modelo:</span>
      <span class="seg" id="v3seg">
        <button data-m="17b" class="active">Qwen3-1.7B</button>
        <button data-m="4b">Qwen3-4B</button>
      </span>
    </div>
    <div class="card">
      <div class="hm" id="hm"></div>
      <div class="legend"><span>Fila = entrenamiento · Columna = evaluación · borde negro = in-domain (diagonal) · <span style="color:var(--r4)">ES</span> = objetivo penalizado</span></div>
    </div>
    <div class="note" id="v3note"></div>
    <p class="foot">Fuente: <code>results/sprint4/transfer_matrix_qwen1.7b.csv</code> y <code>..._qwen4b.csv</code> (f1_macro_mean ± std, medias de {42,43,44}).</p>
  </section>
</main>

<div class="tip" id="tip"></div>

<script id="csv_b" type="text/plain">
__CORTE_B_CSV__
</script>
<script id="csv_a" type="text/plain">
__CORTE_A_CSV__
</script>
<script id="csv_t17" type="text/plain">
__TR17_CSV__
</script>
<script id="csv_t4b" type="text/plain">
__TR4B_CSV__
</script>

<script>
const ROBERTUITO_F1 = __ROBERTUITO_F1__;  // results/baselines/robertuito_offtheshelf.json
const COL = {b17:'#1f4e9c', b17l:'#7aa0d6', r4:'#b5322f', r4l:'#d98b89', navy:'#12306b',
             grey:'#6b7280', greyl:'#9ca3af', slate:'#334155'};

// ---- CSV parser (soporta campos entrecomillados con comas, p.ej. seeds "42,43,44") ----
function splitLine(line){
  const out=[]; let cur='', q=false;
  for(let i=0;i<line.length;i++){const c=line[i];
    if(q){ if(c==='"'){ if(line[i+1]==='"'){cur+='"';i++;} else q=false; } else cur+=c; }
    else { if(c==='"')q=true; else if(c===','){out.push(cur);cur='';} else cur+=c; } }
  out.push(cur); return out;
}
function parseCSV(text){
  const lines=text.trim().split(/\r?\n/); const H=splitLine(lines[0]); const rows=[];
  for(let i=1;i<lines.length;i++){ if(!lines[i].trim())continue;
    const c=splitLine(lines[i]); const o={}; H.forEach((h,j)=>o[h]=c[j]); rows.push(o);} return rows;
}
const CSV = id => parseCSV(document.getElementById(id).textContent);

// ---- Tabs ----
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));
  t.classList.add('active'); document.getElementById('v'+t.dataset.v).classList.add('active');
});

// =========================== VISTA 1 — Pareto ===========================
const B = CSV('csv_b');
const byName = Object.fromEntries(B.map(r=>[r.experiment_name,r]));
const num = (n,c)=>parseFloat(byName[n][c]);
const meanF1 = ns => ns.reduce((s,n)=>s+num(n,'f1_macro'),0)/ns.length;
const meanCol = (ns,c) => ns.reduce((s,n)=>s+parseFloat(byName[n][c]),0)/ns.length;
const LORA17=['lora_qwen1.7b_nfull_s42','lora_qwen1.7b_nfull_s43','lora_qwen1.7b_nfull_s44'];
const LORA4B=['lora_qwen4b_nfull_s42','lora_qwen4b_nfull_s43','lora_qwen4b_nfull_s44'];
const BETO='encoder_finetuned_dccuchile_bert-base-spanish-wwm-cased_cardiff_es';
const XLMR='encoder_finetuned_xlm-roberta-base_cardiff_es';

const PTS = [
  {label:'LoRA 4B',     fam:'4b', meth:'lora',   f1:meanF1(LORA4B), params:11796480, vram:meanCol(LORA4B,'peak_vram_gb'), col:COL.r4,  shape:'circle', r:9},
  {label:'LoRA 1.7B',   fam:'17b',meth:'lora',   f1:meanF1(LORA17), params:6422528,  vram:meanCol(LORA17,'peak_vram_gb'), col:COL.b17, shape:'circle', r:9},
  {label:'QLoRA 4B',    fam:'4b', meth:'qlora',  f1:num('qlora_qwen4b_nfull_s42','f1_macro'),   params:11796480, vram:num('qlora_qwen4b_nfull_s42','peak_vram_gb'), col:COL.r4l, shape:'triangle', r:8},
  {label:'QLoRA 1.7B',  fam:'17b',meth:'qlora',  f1:num('qlora_qwen1.7b_nfull_s42','f1_macro'), params:6422528,  vram:num('qlora_qwen1.7b_nfull_s42','peak_vram_gb'), col:COL.b17l,shape:'triangle', r:8},
  {label:'Full-FT 1.7B',fam:'17b',meth:'full_ft',f1:num('full_ft_qwen1.7b_nfull_s42','f1_macro'), params:1720574976, vram:num('full_ft_qwen1.7b_nfull_s42','peak_vram_gb'), col:COL.navy, shape:'rect', r:8},
  {label:'BETO',        fam:'enc',meth:'encoder',f1:num(BETO,'f1_macro'), params:109853187, vram:num(BETO,'peak_vram_gb'), col:COL.grey, shape:'crossRot', r:8},
  {label:'XLM-R',       fam:'enc',meth:'encoder',f1:num(XLMR,'f1_macro'), params:278045955, vram:num(XLMR,'peak_vram_gb'), col:COL.greyl, shape:'crossRot', r:8},
  {label:'Prompting k=4',fam:'prompt',meth:'prompting',f1:num('prompting_fewshot_k4_Qwen_Qwen3-4B','f1_macro'), params:0, vram:num('prompting_fewshot_k4_Qwen_Qwen3-4B','peak_vram_gb'), col:COL.slate, shape:'rectRot', r:8},
];
const PARAMS_FLOOR = 300000; // prompting tiene 0 params entrenables; en eje log se sitúa en el borde (tooltip muestra 0)
function xval(p, axis){ if(axis==='params') return p.params===0?PARAMS_FLOOR:p.params; return p.vram; }
function pareto(axis){
  return PTS.filter(p=>!PTS.some(q=>q!==p && xval(q,axis)<=xval(p,axis) && q.f1>=p.f1 &&
                        (xval(q,axis)<xval(p,axis)||q.f1>p.f1)))
           .sort((a,b)=>xval(a,axis)-xval(b,axis));
}
let chart1, v1axis='params';
function fmtParams(v){ return v>=1e9?(v/1e9).toFixed(2)+'B':v>=1e6?(v/1e6).toFixed(2)+'M':(v/1e3).toFixed(0)+'k'; }
function buildV1(){
  const axis=v1axis;
  const scatter=PTS.map(p=>({x:xval(p,axis),y:p.f1,pt:p}));
  const front=pareto(axis).map(p=>({x:xval(p,axis),y:p.f1}));
  const xmin = axis==='params'? PARAMS_FLOOR*0.6 : 1.8;
  const xmax = axis==='params'? 3.0e9 : 20;
  const ds=[
    {label:'Frontera de Pareto', type:'line', data:front, borderColor:'#111', borderWidth:1.5,
     borderDash:[5,4], pointRadius:0, fill:false, order:5, tension:0},
    {label:'RoBERTuito 0.758 (fuera de dominio)', type:'line',
     data:[{x:xmin,y:ROBERTUITO_F1},{x:xmax,y:ROBERTUITO_F1}], borderColor:'#a855f7',
     borderWidth:1.3, borderDash:[2,3], pointRadius:0, fill:false, order:6},
    ...PTS.map((p,i)=>({label:p.label, data:[{x:xval(p,axis),y:p.f1,pt:p}], showLine:false,
       pointStyle:p.shape, pointBackgroundColor:p.col, pointBorderColor:p.col,
       pointRadius:p.r, pointHoverRadius:p.r+2, order:1}))
  ];
  const cfg={type:'scatter', data:{datasets:ds},
    options:{maintainAspectRatio:false, animation:false,
      scales:{ x:{type:'logarithmic', min:xmin, max:xmax,
                  title:{display:true,text: axis==='params'?'parámetros entrenables (log)':'VRAM pico GB (log)'},
                  ticks:{callback:v=> axis==='params'?fmtParams(v):v+'' }},
               y:{min:0.60, max:0.77, title:{display:true,text:'F1-macro (Cardiff ES)'}}},
      plugins:{ legend:{display:false},
        tooltip:{callbacks:{ label:(ctx)=>{ const p=ctx.raw.pt; if(!p) return ctx.dataset.label;
          const xtxt = axis==='params'? (p.params===0?'0 (sin entrenamiento)':p.params.toLocaleString('es')+' params')
                                       : p.vram.toFixed(2)+' GB';
          return `${p.label}: F1 ${p.f1.toFixed(4)} · ${xtxt}`; }}}}}};
  if(chart1) chart1.destroy(); chart1=new Chart(document.getElementById('c1'), cfg);
  document.getElementById('v1note').innerHTML = axis==='params'
    ? 'Nota: <b>Prompting k=4</b> tiene <b>0 parámetros entrenables</b>; en un eje logarítmico no existe el 0, así que se dibuja en el borde izquierdo (el tooltip indica 0). Es el punto más barato en parámetros, por eso está en la frontera.'
    : 'En el eje de VRAM, la frontera se recalcula: BETO (menor VRAM) → QLoRA-1.7B → QLoRA-4B → LoRA-4B. Prompting deja de ser Pareto-óptimo (BETO usa menos VRAM con más F1).';
}
document.querySelectorAll('#v1seg button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('#v1seg button').forEach(x=>x.classList.remove('active'));
  b.classList.add('active'); v1axis=b.dataset.x; buildV1();
});
document.getElementById('leg1').innerHTML =
  '<span><span class="dot" style="background:'+COL.b17+'"></span>1.7B</span>'+
  '<span><span class="dot" style="background:'+COL.r4+'"></span>4B</span>'+
  '<span><span class="dot" style="background:'+COL.grey+'"></span>encoders / prompting</span>'+
  '<span>● LoRA · ▲ QLoRA · ■ Full-FT · ✚ encoder · ◆ prompting</span>';

// =========================== VISTA 2 — cruce ===========================
const A = CSV('csv_a');
function curve(model){ return A.filter(r=>r.model_short===model)
  .map(r=>({n:+r.n_train, f1:+r.f1_mean, sd:+r.f1_std})).sort((a,b)=>a.n-b.n); }
const C17=curve('Qwen3-1.7B'), C4=curve('Qwen3-4B');
const bestPrompt=tag=>Math.max(...B.filter(r=>r.method.startsWith('prompting')&&r.model_short===tag).map(r=>+r.f1_macro));
const FLOOR17=bestPrompt('Qwen3-1.7B'), FLOOR4=bestPrompt('Qwen3-4B');
const A_BETO=+byName[BETO].f1_macro, A_XLMR=+byName[XLMR].f1_macro;
function crossover(c,floor){ for(const p of c){ if(p.f1>floor) return p.n; } return null; }
const X17=crossover(C17,FLOOR17), X4=crossover(C4,FLOOR4);
const NMIN=4, NMAX=1839;
let chart2, v2m='both';
function bandDs(c,color){ // banda f1±std como área rellena entre dos líneas
  const up=c.map(p=>({x:p.n,y:Math.min(1,p.f1+p.sd)})), lo=c.map(p=>({x:p.n,y:p.f1-p.sd}));
  return [ {data:lo, borderColor:'transparent', pointRadius:0, fill:false, order:9},
           {data:up, borderColor:'transparent', pointRadius:0, backgroundColor:color+'22', fill:'-1', order:9} ];
}
function hline(y,color,label){ return {label, data:[{x:NMIN,y},{x:NMAX,y}], borderColor:color,
  borderWidth:1.2, borderDash:[4,4], pointRadius:0, fill:false, order:7}; }
function buildV2(){
  const ds=[]; const show17=(v2m==='both'||v2m==='17b'), show4=(v2m==='both'||v2m==='4b');
  if(show17){ ds.push(...bandDs(C17,COL.b17));
    ds.push({label:'LoRA 1.7B', data:C17.map(p=>({x:p.n,y:p.f1})), borderColor:COL.b17,
      backgroundColor:COL.b17, pointRadius:3, borderWidth:2, fill:false, tension:0.15, order:2});
    ds.push(hline(FLOOR17,COL.b17l,'suelo prompting 1.7B'));
    const cp=C17.find(p=>p.n===X17); if(cp) ds.push({label:'cruce 1.7B (n='+X17+')',
      data:[{x:cp.n,y:cp.f1}], showLine:false, pointStyle:'star', pointRadius:13,
      pointBackgroundColor:COL.b17, pointBorderColor:'#111', pointBorderWidth:1.5, order:1}); }
  if(show4){ ds.push(...bandDs(C4,COL.r4));
    ds.push({label:'LoRA 4B', data:C4.map(p=>({x:p.n,y:p.f1})), borderColor:COL.r4,
      backgroundColor:COL.r4, pointRadius:3, borderWidth:2, fill:false, tension:0.15, order:2});
    ds.push(hline(FLOOR4,COL.r4l,'suelo prompting 4B'));
    const cp=C4.find(p=>p.n===X4); if(cp) ds.push({label:'cruce 4B (n='+X4+')',
      data:[{x:cp.n,y:cp.f1}], showLine:false, pointStyle:'star', pointRadius:13,
      pointBackgroundColor:COL.r4, pointBorderColor:'#111', pointBorderWidth:1.5, order:1}); }
  ds.push(hline(A_BETO,COL.grey,'BETO 0.661')); ds.push(hline(A_XLMR,COL.greyl,'XLM-R 0.646'));
  const cfg={type:'line', data:{datasets:ds}, options:{maintainAspectRatio:false, animation:false,
    scales:{ x:{type:'logarithmic', min:NMIN, max:NMAX, title:{display:true,text:'nº de ejemplos de entrenamiento (log; 1839 = full)'},
                ticks:{callback:v=>[4,7,16,50,100,250,1000,1839].includes(v)?v:''}},
             y:{min:0.45, max:0.75, title:{display:true,text:'F1-macro (media 3 semillas)'}}},
    plugins:{ legend:{display:false},
      tooltip:{callbacks:{ label:(ctx)=>{ const d=ctx.dataset.label||''; if(d.startsWith('suelo')||d.startsWith('BETO')||d.startsWith('XLM')) return d;
        if(d.startsWith('cruce')) return d;
        if(d.startsWith('LoRA')){ const p=(d.includes('1.7')?C17:C4).find(q=>q.n===ctx.raw.x);
          return p? `${d} · n=${p.n}: F1 ${p.f1.toFixed(4)} ± ${p.sd.toFixed(4)}`:d; } return ''; }}}}}};
  if(chart2) chart2.destroy(); chart2=new Chart(document.getElementById('c2'), cfg);
  document.getElementById('v2note').innerHTML =
    `Cruce (primer n con LoRA&gt;suelo de su prompting): <b>1.7B → n=${X17}</b> (suelo ${FLOOR17.toFixed(3)}) · <b>4B → n=${X4}</b> (suelo ${FLOOR4.toFixed(3)}). Calculado del dato, no dibujado a mano.`;
}
document.querySelectorAll('#v2seg button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('#v2seg button').forEach(x=>x.classList.remove('active'));
  b.classList.add('active'); v2m=b.dataset.m; buildV2();
});
document.getElementById('leg2').innerHTML =
  '<span><span class="dot" style="background:'+COL.b17+'"></span>LoRA 1.7B</span>'+
  '<span><span class="dot" style="background:'+COL.r4+'"></span>LoRA 4B</span>'+
  '<span>— banda = ± std · ⁃⁃ líneas = suelos prompting / encoders · ★ = punto de cruce</span>';

// =========================== VISTA 3 — heatmap ===========================
const T={ '17b':CSV('csv_t17'), '4b':CSV('csv_t4b') };
const ORDER=['ES','CR','PE'];
const tip=document.getElementById('tip');
function colScale(v){ // 0.60->claro, 0.70->verde oscuro
  const t=Math.max(0,Math.min(1,(v-0.60)/0.10));
  const r=Math.round(237-(237-22)*t), g=Math.round(242-(242-120)*t), b=Math.round(233-(233-70)*t);
  return `rgb(${r},${g},${b})`;
}
let v3m='17b';
function buildV3(){
  const rowsT=T[v3m]; const val={};
  rowsT.forEach(r=>val[r.train+'>'+r.eval]={m:+r.f1_macro_mean, s:+r.f1_macro_std, diag:r.diagonal==='True'});
  const hm=document.getElementById('hm'); hm.innerHTML='';
  hm.appendChild(cellHTML('h','')); // esquina
  ORDER.forEach(ev=>{ const d=document.createElement('div'); d.className='h colh'+(ev==='ES'?' es':'');
    d.textContent='eval '+ev; hm.appendChild(d); });
  ORDER.forEach(tr=>{
    const rh=document.createElement('div'); rh.className='h rowh'; rh.textContent='train '+tr; hm.appendChild(rh);
    ORDER.forEach(ev=>{ const k=tr+'>'+ev, o=val[k]; const c=document.createElement('div');
      c.className='cell'+(o.diag?' diag':''); c.style.background=colScale(o.m);
      c.innerHTML=o.m.toFixed(3)+'<small>± '+o.s.toFixed(3)+'</small>';
      c.onmousemove=(e)=>{ tip.style.opacity=1; tip.style.left=(e.clientX+12)+'px'; tip.style.top=(e.clientY+12)+'px';
        tip.innerHTML=`<b>train ${tr} → eval ${ev}</b><br>F1 ${o.m.toFixed(4)} ± ${o.s.toFixed(4)}<br>${o.diag?'in-domain (diagonal)':'transferencia'}`; };
      c.onmouseleave=()=>tip.style.opacity=0; hm.appendChild(c); });
  });
  // asimetría hacia ES (media off-diagonal a cada objetivo)
  const off=(ev)=>ORDER.filter(t=>t!==ev).reduce((s,t)=>s+val[t+'>'+ev].m,0)/2;
  document.getElementById('v3note').innerHTML =
    `Asimetría: transferir <b>hacia ES</b> rinde ${off('ES').toFixed(3)} de media (off-diagonal), por debajo de hacia CR (${off('CR').toFixed(3)}) y PE (${off('PE').toFixed(3)}). La diagonal ES→ES (${val['ES>ES'].m.toFixed(3)}) es además la más baja: ES es el objetivo difícil.`;
}
function cellHTML(cls,txt){ const d=document.createElement('div'); d.className=cls; d.textContent=txt; return d; }
document.querySelectorAll('#v3seg button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('#v3seg button').forEach(x=>x.classList.remove('active'));
  b.classList.add('active'); v3m=b.dataset.m; buildV3();
});

// init
buildV1(); buildV2(); buildV3();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
