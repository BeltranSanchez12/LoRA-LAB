"""app.py — Demo de inferencia (T5). Dos modos:

  * GPU (principal, por defecto): compara EN VIVO los 5 modelos del paper sobre una frase
    (LoRA-1.7B, LoRA-4B, prompting k=4, BETO, XLM-R), con predicción + probabilidad por clase,
    más el botón "colapso cased" (perturbación MAYÚSCULAS del Sprint 6) para ver cómo los
    encoders cased se desploman y los LoRA aguantan. Pensado para la DGX (H200), servido por
    Gradio y accesible por túnel SSH desde el portátil.

  * CPU (fallback, `--cpu`): la versión ligera que ya funciona en portátil sin GPU
    (solo LoRA-1.7B + prompting k=4). Red de seguridad por si el día de la defensa falla la DGX.

El modelo se carga UNA sola vez al arrancar. Solo inferencia: cero reentrenamiento, cero datos
inventados. Mapeo de etiquetas = el del paper. Ver demo/README.md para arranque y túnel SSH.

Arranque:
    python demo/app.py            # modo GPU (DGX)
    python demo/app.py --cpu      # modo CPU fallback (portátil)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parse_args():
    ap = argparse.ArgumentParser(description="Demo de sentimiento ES (T5)")
    ap.add_argument("--cpu", action="store_true",
                    help="modo fallback CPU: solo LoRA-1.7B + prompting (portátil sin GPU)")
    ap.add_argument("--host", default=os.environ.get("DEMO_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("DEMO_PORT", "7860")))
    return ap.parse_args()


ARGS = _parse_args()
if ARGS.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # forzar CPU ANTES de importar torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gradio as gr  # noqa: E402

EXAMPLES = [
    "Qué maravilla de película, me encantó de principio a fin.",
    "El paquete llegó tarde y encima roto, un auténtico desastre.",
    "La reunión es el jueves a las seis en el auditorio principal.",
]

CORTE_C_NOTE = (
    "### Transferencia dialectal (Corte C del paper)\n"
    "Los adaptadores de esta demo se entrenaron sobre **Cardiff ES** (español general). El paper "
    "mide además la **transferencia entre variedades** (entrenar en una, evaluar en otra): la "
    "columna **España (ES)** es la más penalizada como destino. No cargamos aquí un adaptador por "
    "variedad (solo están reconstruidos los de ES); para el detalle 3×3 por modelo, ver el "
    "**explorador interactivo** (`defense/deck_assets/explorer.html`, Vista 3) y la Figura C. "
    "_Nota informativa: cero datos nuevos, todo sale de los CSV del repo._"
)


# ===================================================================== GPU
def build_gpu_app():
    from multimodel import MultiModelDemo, MODEL_ORDER, MODEL_LABELS, LABELS, LABEL_ES

    print("Cargando los 5 modelos en GPU (una sola vez)...", flush=True)
    D = MultiModelDemo(device="cuda")
    # Warmup: dispara la compilación de kernels CUDA de los 5 modelos con frases de VARIAS
    # longitudes (los encoders re-autotunean por forma de entrada), para que el PRIMER clic real
    # en la defensa ya vaya a tiempo pleno (~0.1 s los 5) y no ~0.5 s por el warmup en frío.
    print("Warmup de los 5 modelos...", flush=True)
    _warm = [
        "Bien.",
        "Frase de calentamiento para inicializar los kernels de GPU.",
        "Esta es una frase de calentamiento algo más larga, con varias cláusulas y bastantes "
        "palabras, para cubrir longitudes de secuencia distintas y compilar sus kernels.",
    ]
    for _ in range(2):
        for _s in _warm:
            D.classify_all(_s)
    print("Modelos listos (warmup hecho): el primer clic ya va a tiempo pleno.", flush=True)

    def do_classify(text):
        text = (text or "").strip()
        if not text:
            return [], "_Escribe una frase en español y pulsa **Clasificar**._"
        r = D.classify_all(text)
        rows = []
        for k in MODEL_ORDER:
            pr = r[k]["probs"]
            rows.append([MODEL_LABELS[k], LABEL_ES[r[k]["pred"]].upper(),
                         round(pr["negative"] * 100, 1), round(pr["neutral"] * 100, 1),
                         round(pr["positive"] * 100, 1), round(r[k]["latency_s"] * 1000)])
        preds = {LABEL_ES[r[k]["pred"]] for k in MODEL_ORDER}
        if len(preds) == 1:
            head = f"**Acuerdo de los 5 modelos → {next(iter(preds)).upper()}**"
        else:
            head = "**Desacuerdo:** " + " · ".join(
                f"{MODEL_LABELS[k]}={LABEL_ES[r[k]['pred']]}" for k in MODEL_ORDER)
        return rows, head

    def do_cased(text):
        text = (text or "").strip()
        if not text:
            return [], "_Escribe una frase y pulsa **Colapso cased**._"
        c = D.classify_cased(text)
        rows = []
        for k in MODEL_ORDER:
            a, b = c["clean"][k], c["upper"][k]
            am, bm = a["probs"][a["pred"]], b["probs"][b["pred"]]
            flip = a["pred"] != b["pred"]
            rows.append([MODEL_LABELS[k],
                         f"{LABEL_ES[a['pred']].upper()} ({am*100:.0f}%)",
                         f"{LABEL_ES[b['pred']].upper()} ({bm*100:.0f}%)",
                         "⚠️ VOLTEA" if flip else "estable"])
        enc_flip = [MODEL_LABELS[k] for k in ("beto", "xlmr")
                    if c["clean"][k]["pred"] != c["upper"][k]["pred"]]
        lora_flip = [MODEL_LABELS[k] for k in ("lora_1.7b", "lora_4b")
                     if c["clean"][k]["pred"] != c["upper"][k]["pred"]]
        head = (f"**MAYÚSCULAS:** _{c['upper_text'][:140]}_\n\n"
                f"- Encoders que voltean: **{', '.join(enc_flip) or 'ninguno'}**\n"
                f"- LoRA que voltean: **{', '.join(lora_flip) or 'ninguno (aguantan)'}**\n\n"
                "_Hallazgo Sprint 6: MAYÚSCULAS colapsa los encoders cased "
                "(BETO −0.184 / XLM-R −0.110 Macro-F1 en test), mientras los LoRA generativos "
                "apenas se mueven._")
        return rows, head

    with gr.Blocks(title="Sentimiento ES — 5 modelos (GPU)") as app:
        gr.Markdown(
            "# Análisis de sentimiento en español — comparación en vivo de 5 modelos (GPU)\n"
            "Una frase → predicción y probabilidad por clase de **LoRA-1.7B, LoRA-4B, prompting "
            "k=4 (Qwen3-4B), BETO y XLM-R** a la vez. Todo inferencia sobre pesos congelados."
        )
        inp = gr.Textbox(label="Frase en español", lines=2,
                         placeholder="Escribe aquí una frase...")
        gr.Examples(examples=[[s] for s in EXAMPLES], inputs=inp)
        with gr.Tabs():
            with gr.Tab("A · Comparación 5 modelos"):
                btn_a = gr.Button("Clasificar (5 modelos)", variant="primary")
                head_a = gr.Markdown()
                df_a = gr.Dataframe(
                    headers=["Modelo", "Predicción", "P(neg) %", "P(neu) %", "P(pos) %", "ms"],
                    datatype=["str", "str", "number", "number", "number", "number"],
                    interactive=False, wrap=True)
            with gr.Tab("B · Colapso cased (MAYÚSCULAS)"):
                gr.Markdown("Aplica la perturbación **MAYÚSCULAS** del Sprint 6 y compara "
                            "antes/después: mira cómo BETO/XLM-R se desploman y los LoRA aguantan.")
                btn_b = gr.Button("Colapso cased (MAYÚSCULAS)", variant="primary")
                head_b = gr.Markdown()
                df_b = gr.Dataframe(
                    headers=["Modelo", "Limpio", "MAYÚSCULAS", "¿voltea?"],
                    datatype=["str", "str", "str", "str"], interactive=False, wrap=True)
            with gr.Tab("C · Transferencia dialectal"):
                gr.Markdown(CORTE_C_NOTE)

        btn_a.click(do_classify, inp, [df_a, head_a])
        inp.submit(do_classify, inp, [df_a, head_a])
        btn_b.click(do_cased, inp, [df_b, head_b])
    app.fn_classify, app.fn_cased = do_classify, do_cased   # hooks para el smoke test
    return app


# ===================================================================== CPU
def build_cpu_app():
    from infer import SentimentDemo, LABELS, LABEL_ES

    print("Cargando Qwen3-1.7B + adaptador LoRA en CPU (una sola vez)...", flush=True)
    _threads = os.environ.get("DEMO_THREADS")
    D = SentimentDemo(with_prompting=True, threads=int(_threads) if _threads else None)
    print(f"Modelo listo. Adaptador: {D.adapter_dir}", flush=True)

    def classify(text, do_prompting):
        text = (text or "").strip()
        if not text:
            return {}, "_Escribe una frase en español y pulsa **Clasificar**._"
        r = D.classify_lora(text)
        probs = {LABEL_ES[w]: float(r["probs"][w]) for w in LABELS}
        note = f"**LoRA-1.7B** → **{LABEL_ES[r['pred']].upper()}**  ·  {r['latency_s']:.2f}s en CPU"
        if do_prompting:
            p = D.classify_prompting(text)
            pred_es = LABEL_ES.get(p["pred"], p["pred"]).upper()
            note += (f"\n\n**Prompting k=4** (Qwen3-1.7B base, sin LoRA) → **{pred_es}**  ·  "
                     f"{p['latency_s']:.2f}s en CPU")
            if p.get("fallback"):
                note += "  \n_(respuesta no parseable → fallback a neutral)_"
        return probs, note

    with gr.Blocks(title="Sentimiento ES — LoRA Qwen3-1.7B (CPU, fallback)") as app:
        gr.Markdown(
            "# Análisis de sentimiento en español — LoRA sobre Qwen3-1.7B (CPU · fallback)\n"
            "Modo de **red de seguridad** en CPU (sin GPU). Etiquetas: **negativo · neutral · "
            "positivo**. Adaptador LoRA frente al baseline de *prompting* k=4 sobre el mismo base."
        )
        with gr.Row():
            with gr.Column(scale=3):
                inp = gr.Textbox(label="Frase en español", lines=3,
                                 placeholder="Escribe aquí una frase...")
                with gr.Row():
                    chk = gr.Checkbox(value=True,
                                      label="Comparar con prompting k=4 (base, sin LoRA)")
                    btn = gr.Button("Clasificar", variant="primary")
            with gr.Column(scale=2):
                out_label = gr.Label(label="Probabilidades por clase (LoRA-1.7B)")
                out_note = gr.Markdown()
        gr.Examples(examples=[[s] for s in EXAMPLES], inputs=inp)
        btn.click(classify, [inp, chk], [out_label, out_note])
        inp.submit(classify, [inp, chk], [out_label, out_note])
    app.fn_classify = classify   # hook para el smoke test
    return app


app = build_cpu_app() if ARGS.cpu else build_gpu_app()

if __name__ == "__main__":
    app.launch(server_name=ARGS.host, server_port=ARGS.port)
