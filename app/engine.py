"""Inference engine: photo -> attributes (with abstention) -> description -> verification.

This is the B-Abs method from the research project, packaged for a website:

1. SigLIP scores the photo against prompt-ensembled label descriptions
   (category, color, material) exactly as in the paper.
2. Each attribute below its dev-tuned confidence threshold becomes "unknown"
   (abstention), so the writer is never handed a guess.
3. An instruction-tuned LLM writes the description from that JSON.
4. The lexicon claim checker reads the description back. If it states any
   category, color or material that differs from the evidence, or one that is
   unknown, the description is replaced by the APG (slot) rendering, which is
   safe by construction.
"""

from __future__ import annotations

import json
import re
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from groundedgen import apg, claims, prompts
from groundedgen import taxonomy as tx

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

MODES = ("safe", "strict", "forced")

# App-only additions for the small CPU writer (the research prompts stay frozen and unchanged).
# The 1.5B model pads unknown material with fabric talk and misreads the slot format without
# an example; the 7B models in the study did neither.
APP_SYSTEM_SUFFIX = (
    " If the material is unknown, do not mention fabric, material, texture or quality at all."
    " Do not describe fit, sizing or care."
)
APP_SLOT_EXAMPLE = (
    "\n\nExample of the format (for a product with category and color listed): "
    "\"Meet our {color} {category}, an easy everyday essential. It brings a clean, polished "
    "look to any outfit.\" Write full sentences, not a list."
)


FABRIC_TALK = re.compile(r"\b(fabric|fabrics|material|materials|textile|texture|woven|crafted from|made from|made of)\b",
                         re.IGNORECASE)


def drop_fabric_talk(text: str, evidence: dict[str, str]) -> tuple[str, list[str]]:
    """With material unknown, remove sentences that talk about fabric at all
    ("crafted from an unidentifiable material"): no fabric word is a claim the
    checker can verify, and none of it helps a buyer."""
    if evidence["material"] != tx.UNKNOWN:
        return text, []
    kept, dropped = [], []
    for sentence in claims.split_sentences(text):
        (dropped if FABRIC_TALK.search(sentence) else kept).append(sentence)
    return " ".join(kept), dropped


def well_formed(text: str) -> bool:
    """Reject list-like or fragmentary output (``"T-shirt: T-shirt"``) before it reaches a buyer."""
    words = text.split()
    return len(words) >= 6 and ":" not in text and "\n" not in text.strip()


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def _features(out: Any, key: str):
    """``get_*_features`` returns a tensor or, in newer transformers, an output object."""
    if hasattr(out, "detach"):
        return out
    value = getattr(out, key, None)
    if value is None:
        value = getattr(out, "pooler_output")
    return value


@dataclass
class Result:
    evidence: dict[str, str]
    forced: dict[str, str]
    scores: dict[str, dict[str, Any]]
    description: str
    mode: str
    guard: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"evidence": self.evidence, "forced": self.forced, "scores": self.scores,
                "description": self.description, "mode": self.mode, "guard": self.guard,
                "timings": self.timings}


def resolve(name: str, config: dict[str, Any] = CONFIG) -> tuple[str, str | None]:
    """A model's local folder from ``download_models.py`` if present, else its Hub id + pinned revision."""
    folder = ROOT / "models" / name.replace("/", "__")
    if (folder / "config.json").exists():
        return str(folder), None
    return name, config.get("revisions", {}).get(name)


class Engine:
    """Loads both models once; ``describe`` is thread-safe (one request at a time)."""

    def __init__(self, config: dict[str, Any] = CONFIG):
        self.config = config
        self.thresholds = config["thresholds"]
        self.lock = threading.Lock()
        self.ready = False
        self.error: str | None = None

    # ---- loading ---------------------------------------------------------------

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        t0 = time.time()
        torch.set_grad_enabled(False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        name, rev = resolve(self.config["extractor"], self.config)
        LOGGER.info("loading extractor %s on %s", name, self.device)
        self.processor = AutoProcessor.from_pretrained(name, revision=rev)
        self.encoder = AutoModel.from_pretrained(name, revision=rev, dtype=dtype).to(self.device).eval()
        self.logit_scale = float(self.encoder.logit_scale.detach().float().exp().item())
        self.classes = self._class_embeddings()

        self.llm_name = os.environ.get("GHD_LLM", self.config["llm"])
        name, rev = resolve(self.llm_name, self.config)
        LOGGER.info("loading writer %s", name)
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=rev)
        self.llm = AutoModelForCausalLM.from_pretrained(name, revision=rev, dtype=dtype).to(self.device).eval()
        self.ready = True
        LOGGER.info("models ready in %.0f s", time.time() - t0)

    def _encode_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, padding="max_length", truncation=True,
                                return_tensors="pt").to(self.device)
        out = self.encoder.get_text_features(**inputs)
        return _features(out, "text_embeds").detach().float().cpu().numpy()

    def _class_embeddings(self) -> dict[str, tuple[list[str], np.ndarray]]:
        """Per attribute: labels and their 4-template ensembled embeddings (as in the paper)."""
        out = {}
        for attribute in tx.ATTRIBUTES:
            filled = tx.clip_prompts(attribute)
            labels = list(filled)
            n = len(tx.CLIP_TEMPLATES[attribute])
            emb = _l2(self._encode_text([p for label in labels for p in filled[label]]))
            out[attribute] = (labels, _l2(emb.reshape(len(labels), n, -1).mean(axis=1)))
        return out

    # ---- stages ----------------------------------------------------------------

    def extract(self, image) -> dict[str, dict[str, Any]]:
        """Softmax over ``logit_scale x cosine`` per attribute; top label and its confidence."""
        inputs = self.processor(images=[image.convert("RGB")], return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.encoder.dtype)
        img = _l2(_features(self.encoder.get_image_features(**inputs), "image_embeds")
                  .detach().float().cpu().numpy())
        scores = {}
        for attribute, (labels, cls) in self.classes.items():
            z = self.logit_scale * img @ cls.T
            p = np.exp(z - z.max())
            p = (p / p.sum())[0]
            order = np.argsort(-p, kind="stable")
            scores[attribute] = {
                "pred": labels[order[0]],
                "confidence": float(p[order[0]]),
                "threshold": self.thresholds[attribute],
                "top": [{"label": labels[i], "p": round(float(p[i]), 4)} for i in order[:3]],
            }
        return scores

    def evidence(self, scores: dict[str, dict[str, Any]], abstain: bool = True) -> dict[str, str]:
        out = {}
        for attribute in tx.ATTRIBUTES:
            s = scores[attribute]
            t = s["threshold"]
            keep = not abstain or (t is not None and s["confidence"] >= t)
            out[attribute] = s["pred"] if keep else tx.UNKNOWN
        return out

    def write(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.llm.generate(**inputs, do_sample=False, max_new_tokens=self.config["max_new_tokens"],
                                pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()

    def free_text(self, evidence: dict[str, str]) -> str:
        user = prompts.GROUNDED_USER.format(evidence=prompts.evidence_json(evidence))
        return self.write(prompts.FAITHFUL_SYSTEM_TEXT + APP_SYSTEM_SUFFIX, user)

    def slot_text(self, evidence: dict[str, str]) -> tuple[str, dict[str, Any]]:
        user = prompts.apg_user(evidence)
        if prompts.apg_placeholders(evidence):
            user += APP_SLOT_EXAMPLE
        raw = self.write(prompts.FAITHFUL_SYSTEM_TEXT + APP_SYSTEM_SUFFIX, user)
        rendered = apg.render(raw, evidence)
        log = dict(rendered.log)
        if not well_formed(rendered.text) and not log.get("fallback_used"):
            log["fallback_used"] = True
            log["rejected"] = rendered.text
            return apg.fallback_text(evidence), log
        return rendered.text, log

    # ---- verification ------------------------------------------------------------

    @staticmethod
    def check(text: str, evidence: dict[str, str]) -> list[dict[str, Any]]:
        """Attribute claims in ``text`` that the evidence does not support."""
        problems = []
        for attribute, info in claims.extract_claims(text).items():
            if not info["stated"]:
                continue
            e = evidence[attribute]
            wrong = [v for v in info["values"] if v != e]
            if e == tx.UNKNOWN:
                problems.append({"attribute": attribute, "stated": info["values"],
                                 "reason": "states an attribute the image does not show clearly"})
            elif wrong:
                problems.append({"attribute": attribute, "stated": wrong,
                                 "reason": f"differs from the detected {attribute} ({e})"})
        return problems

    def mentions(self, text: str) -> dict[str, list[str]]:
        found = claims.extract_claims(text)
        return {a: found[a]["values"] for a in tx.ATTRIBUTES}

    # ---- the whole pipeline -----------------------------------------------------------

    def describe(self, image, mode: str = "safe") -> Result:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        import torch

        with self.lock, torch.inference_mode():
            t = {}
            t0 = time.time()
            scores = self.extract(image)
            t["extract_s"] = round(time.time() - t0, 2)
            safe_evidence = self.evidence(scores, abstain=True)
            forced = self.evidence(scores, abstain=False)
            evidence = forced if mode == "forced" else safe_evidence

            t0 = time.time()
            guard: dict[str, Any] = {}
            if mode == "strict":
                text, log = self.slot_text(evidence)
                guard = {"method": "slots", "fallback": bool(log.get("fallback_used"))}
            else:
                text = self.free_text(evidence)
                problems = self.check(text, evidence)
                guard = {"method": "free text", "problems": problems, "replaced": False}
                if mode == "safe" and not problems:
                    text, dropped = drop_fabric_talk(text, evidence)
                    guard["dropped_sentences"] = dropped
                    if not well_formed(text):  # too little left: rebuild with slots
                        problems = [{"attribute": "material", "stated": [],
                                     "reason": "talks about fabric that the photo does not show"}]
                        guard["problems"] = problems
                if problems and mode == "safe":
                    guard["original"] = text
                    text, log = self.slot_text(evidence)
                    guard["replaced"] = True
            t["write_s"] = round(time.time() - t0, 2)
            guard["mentions"] = self.mentions(text)
            return Result(evidence, forced, scores, text, mode, guard, t)
