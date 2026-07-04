"""multimodel.py — Motor multi-modelo para la demo GPU (T5, modo principal).

Carga los CINCO modelos clave del paper con sus pesos CONGELADOS y clasifica una frase en
español con los cinco a la vez (predicción + probabilidad por clase), para mostrarlos lado a
lado. CERO reentrenamiento: solo inferencia; lo único que cambia es el texto de entrada.

Modelos (mismos loaders que scripts/sprint6_robustness.py, para que casen con el paper):
  - lora_1.7b : Qwen3-1.7B (bf16) + adaptador LoRA reconstruido (semilla 42)
  - lora_4b   : Qwen3-4B  (bf16) + adaptador LoRA reconstruido (semilla 42)
  - prompt_k4 : Qwen3-4B  4-bit NF4, few-shot k=4 (selección semilla 42 = la del paper)
  - beto      : BETO fine-tuneado (clasificador de secuencia)
  - xlmr      : XLM-R base fine-tuneado (clasificador de secuencia)

Probabilidades:
  - Generativos (LoRA 1.7B/4B, prompting): **verbalizador** = softmax sobre la verosimilitud
    normalizada de cada palabra-etiqueta (negative/neutral/positive) tras el mismo prompt de
    entrenamiento/evaluación. El argmax coincide con la predicción greedy del paper (se verifica).
  - Encoders: softmax sobre los 3 logits del clasificador.

Función "colapso cased": aplica la perturbación MAYÚSCULAS del Sprint 6
(`src.data.perturbations`, sin reimplementar) y reevalúa los 5 → antes/después en vivo.

Mapeo de etiquetas = el del paper: {0:negative, 1:neutral, 2:positive}.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.prompting import PromptingBaseline  # noqa: E402
from src.data.perturbations import apply_perturbation  # noqa: E402

LABELS = ["negative", "neutral", "positive"]
LABEL_ES = {"negative": "negativo", "neutral": "neutral", "positive": "positivo"}
MODEL_ORDER = ["lora_1.7b", "lora_4b", "prompt_k4", "beto", "xlmr"]
MODEL_LABELS = {
    "lora_1.7b": "LoRA Qwen3-1.7B",
    "lora_4b": "LoRA Qwen3-4B",
    "prompt_k4": "Prompting k=4 (Qwen3-4B)",
    "beto": "BETO",
    "xlmr": "XLM-R",
}


@torch.no_grad()
def _verbalizer_probs(model, tok, prompt: str, label_ids: dict, pad_id: int, device: str) -> dict:
    """P(clase) = softmax sobre la log-verosimilitud de cada palabra-etiqueta tras el prompt."""
    pids = tok(prompt, add_special_tokens=False)["input_ids"]
    P = len(pids)
    seqs = [pids + label_ids[w] for w in LABELS]
    maxlen = max(len(s) for s in seqs)
    input_ids = torch.tensor([s + [pad_id] * (maxlen - len(s)) for s in seqs], device=device)
    attn = torch.tensor([[1] * len(s) + [0] * (maxlen - len(s)) for s in seqs], device=device)
    logits = model(input_ids=input_ids, attention_mask=attn).logits  # [3, maxlen, V]
    logp = {}
    for r, w in enumerate(LABELS):
        lp = 0.0
        for k, tokid in enumerate(label_ids[w]):
            dist = torch.log_softmax(logits[r, P + k - 1, :].float(), dim=-1)
            lp += dist[tokid].item()
        logp[w] = lp
    m = max(logp.values())
    exps = {w: math.exp(logp[w] - m) for w in LABELS}
    Z = sum(exps.values())
    return {w: exps[w] / Z for w in LABELS}


class MultiModelDemo:
    def __init__(self, device: str = "cuda", dtype=None,
                 fewshot: list[dict] | None = None):
        self.device = device
        self.dtype = dtype or torch.bfloat16
        self.proto = ROOT / "configs" / "prompting_protocol.yaml"
        self.gen = {}   # modelos generativos: key -> dict(tok, model, pb, examples, label_ids, pad_id)
        self.enc = {}   # encoders: key -> dict(tok, model)

        # few-shot k=4 fijo (semilla 42, el del paper); embebido para autocontención
        if fewshot is None:
            import json
            fewshot = json.loads((Path(__file__).resolve().parent / "fewshot_k4.json")
                                 .read_text(encoding="utf-8"))
        self.fewshot = fewshot

        self._load_lora("lora_1.7b", "Qwen/Qwen3-1.7B",
                        ROOT / "adapters" / "lora_qwen1.7b_nfull_s42")
        self._load_lora("lora_4b", "Qwen/Qwen3-4B",
                        ROOT / "adapters" / "lora_qwen4b_nfull_s42")
        self._load_prompting("prompt_k4", "Qwen/Qwen3-4B")
        self._load_encoder("beto", ROOT / "results" / "checkpoints" / "encoder_beto_cardiff_es")
        self._load_encoder("xlmr", ROOT / "results" / "checkpoints" / "encoder_xlmr_base_cardiff_es")

    # ------------------------------------------------------------------ carga
    def _qwen_common(self, tok, pb, examples):
        label_ids = {w: tok(w, add_special_tokens=False)["input_ids"] for w in LABELS}
        return {"tok": tok, "pb": pb, "examples": examples,
                "label_ids": label_ids, "pad_id": tok.pad_token_id}

    def _load_lora(self, key, base_model, adapter_dir):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=self.dtype, device_map={"": 0}, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, str(adapter_dir))
        model.eval()
        model.config.use_cache = True
        pb = PromptingBaseline(config_path=self.proto, k=0)   # prompt = formato de train (k=0)
        d = self._qwen_common(tok, pb, [])
        d["model"] = model
        self.gen[key] = d

    def _load_prompting(self, key, model_id):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map={"": 0}, trust_remote_code=True)
        model.eval()
        pb = PromptingBaseline(config_path=self.proto, k=4)
        d = self._qwen_common(tok, pb, self.fewshot)
        d["model"] = model
        self.gen[key] = d

    def _load_encoder(self, key, ckpt_dir):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(ckpt_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir)).to(self.device).eval()
        self.enc[key] = {"tok": tok, "model": model}

    # -------------------------------------------------------------- inferencia
    @torch.no_grad()
    def _classify_gen(self, key: str, text: str) -> dict:
        d = self.gen[key]
        prompt = d["pb"].render_prompt(text, d["examples"], d["tok"])
        probs = _verbalizer_probs(d["model"], d["tok"], prompt, d["label_ids"],
                                  d["pad_id"], self.device)
        pred = max(probs, key=probs.get)
        return {"probs": probs, "pred": pred}

    @torch.no_grad()
    def _classify_enc(self, key: str, text: str, max_length: int = 128) -> dict:
        d = self.enc[key]
        enc = d["tok"]([text], return_tensors="pt", padding=True, truncation=True,
                       max_length=max_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = d["model"](**enc).logits[0].float()
        p = torch.softmax(logits, dim=-1).cpu().tolist()   # índice 0=neg,1=neu,2=pos
        probs = {LABELS[i]: p[i] for i in range(3)}
        pred = max(probs, key=probs.get)
        return {"probs": probs, "pred": pred}

    def classify_one(self, key: str, text: str) -> dict:
        t0 = time.perf_counter()
        r = self._classify_gen(key, text) if key in self.gen else self._classify_enc(key, text)
        r["latency_s"] = time.perf_counter() - t0
        return r

    def classify_all(self, text: str) -> dict:
        return {key: self.classify_one(key, text) for key in MODEL_ORDER}

    @torch.no_grad()
    def greedy_gen(self, key: str, text: str) -> str:
        """Predicción greedy generativa del paper (para verificar fidelidad vs verbalizador)."""
        d = self.gen[key]
        preds, _ = d["pb"].predict_batch([text], d["model"], d["tok"],
                                         examples=d["examples"], device=self.device)
        return preds[0]

    # --------------------------------------------------------- colapso cased
    def uppercase_text(self, text: str) -> str:
        return apply_perturbation("mayusculas", [text], seed=42)[0]

    def classify_cased(self, text: str) -> dict:
        """Antes (limpio) vs después (MAYÚSCULAS) para los 5 modelos."""
        upper = self.uppercase_text(text)
        return {"clean_text": text, "upper_text": upper,
                "clean": self.classify_all(text), "upper": self.classify_all(upper)}
