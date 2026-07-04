"""infer.py — Núcleo de inferencia de la demo (T5), pensado para CPU.

Carga UNA vez Qwen3-1.7B (float32, CPU) + el adaptador LoRA reconstruido y clasifica una frase
en español en {negativo, neutral, positivo} con la PROBABILIDAD de cada clase. Cero
reentrenamiento: solo inferencia sobre pesos congelados.

Probabilidades por clase = **verbalizador**: probabilidad normalizada (softmax) de que el modelo
genere cada palabra-etiqueta ("negative"/"neutral"/"positive") tras el mismo prompt con el que se
entrenó/evaluó (`PromptingBaseline.render_prompt`). Es exactamente el objetivo con el que se
entrenó el adaptador (ver `build_tokenized_dataset`), así que la clase de máxima probabilidad
coincide en la práctica con la predicción generativa greedy del paper (se verifica en el test).

El baseline de prompting k=4 (opcional) reutiliza el MISMO modelo cargado con el adaptador
DESACTIVADO (base Qwen3-1.7B), few-shot fijo semilla 42 — misma comparación de modelo.

Mapeo de etiquetas idéntico al paper: {0:negative, 1:neutral, 2:positive}.
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.prompting import PromptingBaseline  # noqa: E402

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABELS = ["negative", "neutral", "positive"]
LABEL_ES = {"negative": "negativo", "neutral": "neutral", "positive": "positivo"}
BASE_MODEL = "Qwen/Qwen3-1.7B"


def find_adapter() -> str:
    """Localiza el adaptador LoRA-1.7B. Prioridad: env var, adapters/ del repo, demo/adapter/."""
    cands = [os.environ.get("DEMO_ADAPTER_DIR"),
             ROOT / "adapters" / "lora_qwen1.7b_nfull_s42",
             Path(__file__).resolve().parent / "adapter"]
    for p in cands:
        if p and Path(p).exists() and (Path(p) / "adapter_config.json").exists():
            return str(p)
    raise FileNotFoundError(
        "No encuentro el adaptador LoRA-1.7B. Colócalo en 'adapters/lora_qwen1.7b_nfull_s42/' "
        "o exporta DEMO_ADAPTER_DIR. Ver demo/README.md (§ Adaptador).")


class SentimentDemo:
    def __init__(self, with_prompting: bool = False, threads: int | None = None):
        if threads:
            torch.set_num_threads(int(threads))
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.adapter_dir = find_adapter()
        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # torch_dtype: kwarg estable en transformers 4.51→5.x (float32 = compatible en CPU).
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32, trust_remote_code=True)
        self.model = PeftModel.from_pretrained(base, self.adapter_dir)
        self.model.to("cpu").eval()
        self.model.config.use_cache = True

        proto = ROOT / "configs" / "prompting_protocol.yaml"
        self.pb0 = PromptingBaseline(config_path=proto, k=0)   # prompt zero-shot (= formato de train)
        self.label_ids = {w: self.tok(w, add_special_tokens=False)["input_ids"] for w in LABELS}
        self.pad_id = self.tok.pad_token_id

        self.with_prompting = with_prompting
        self.pb4 = None
        self.fewshot = None
        if with_prompting:
            # Ejemplos k=4 FIJOS (selección estratificada semilla 42, la misma del paper),
            # embebidos en demo/fewshot_k4.json -> la demo NO depende del dataset procesado.
            import json
            fs_path = Path(__file__).resolve().parent / "fewshot_k4.json"
            self.fewshot = json.loads(fs_path.read_text(encoding="utf-8"))
            self.pb4 = PromptingBaseline(config_path=proto, k=4)

    # --------------------------------------------------------------- LoRA (verbalizador)
    @torch.no_grad()
    def classify_lora(self, text: str) -> dict:
        t0 = time.perf_counter()
        prompt = self.pb0.render_prompt(text, [], self.tok)
        pids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        P = len(pids)
        # 3 secuencias (prompt + etiqueta_i), right-pad (posiciones reales antes del pad)
        seqs = [pids + self.label_ids[w] for w in LABELS]
        maxlen = max(len(s) for s in seqs)
        input_ids = torch.tensor([s + [self.pad_id] * (maxlen - len(s)) for s in seqs])
        attn = torch.tensor([[1] * len(s) + [0] * (maxlen - len(s)) for s in seqs])
        logits = self.model(input_ids=input_ids, attention_mask=attn).logits  # [3, maxlen, V]
        logp = {}
        for r, w in enumerate(LABELS):
            ids = self.label_ids[w]
            lp = 0.0
            for k, tokid in enumerate(ids):
                pos = P + k                       # posición de este token de etiqueta
                dist = torch.log_softmax(logits[r, pos - 1, :].float(), dim=-1)
                lp += dist[tokid].item()
            logp[w] = lp
        m = max(logp.values())
        exps = {w: math.exp(logp[w] - m) for w in LABELS}
        Z = sum(exps.values())
        probs = {w: exps[w] / Z for w in LABELS}
        pred = max(probs, key=probs.get)
        return {"probs": probs, "pred": pred, "latency_s": time.perf_counter() - t0}

    # --------------------------------------------------------------- Greedy (fidelidad al paper)
    @torch.no_grad()
    def classify_lora_greedy(self, text: str) -> str:
        preds, _ = self.pb0.predict_batch([text], self.model, self.tok, examples=[], device="cpu")
        return preds[0]

    # --------------------------------------------------------------- Prompting k=4 (opcional)
    @torch.no_grad()
    def classify_prompting(self, text: str) -> dict | None:
        if not self.with_prompting:
            return None
        t0 = time.perf_counter()
        with self.model.disable_adapter():   # base Qwen3-1.7B sin LoRA
            preds, fb = self.pb4.predict_batch([text], self.model, self.tok,
                                               examples=self.fewshot, device="cpu")
        return {"pred": preds[0], "fallback": fb, "latency_s": time.perf_counter() - t0}
