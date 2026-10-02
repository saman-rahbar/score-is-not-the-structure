#!/usr/bin/env python
"""Anchor the fine-tuning regime: evaluate the PRETRAINED Pythia models (no
fine-tuning) on the same 8 BLiMP paradigms, so we can show whether six epochs on
627 sentences moves BLiMP at all relative to the base model (answers the
'floor-effect' reviewer question). Eval only -- no training. Reuses the loaders
and scorer from experiment_full so the protocol is identical.

Run on a compute node with the caches already populated:  python pretrained_blimp.py
Writes pretrained_blimp.json next to it."""
import os
import json
import torch
import experiment_full as E  # reuse load_blimp, blimp_accuracy, device, CONFIG
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = ["EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1.4b"]


def main():
    blimp = E.load_blimp()
    print(f"Loaded BLiMP: {len(blimp)} pairs across {len(E.CONFIG['blimp_configs'])} paradigms")
    out = {}
    for m in MODELS:
        tok = AutoTokenizer.from_pretrained(m)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.float32)
        model.to(E.device).eval()
        acc, by_cfg = E.blimp_accuracy(model, tok, blimp)
        print(f"{m}: pretrained BLiMP = {acc:.4f}")
        out[m.split('/')[-1]] = {"pretrained_blimp": acc, "by_config": by_cfg}
        del model
        if E.device.type == "cuda":
            torch.cuda.empty_cache()
    with open("pretrained_blimp.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + json.dumps(out))


if __name__ == "__main__":
    main()
