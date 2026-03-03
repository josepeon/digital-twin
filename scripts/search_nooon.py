#!/usr/bin/env python3
"""Search the v0.6 dataset for Nooon and fashion-related messages."""
import json

nooon_msgs = []
fashion_msgs = []
keywords = [
    "nooon", "fashion", "brand", "clothing", "collection", "drop",
    "ocean", "art director", "garment", "design", "shopify", "store",
    "complexcon", "showroom",
]

with open("data/personal/processed/v0.6.jsonl") as f:
    for line in f:
        d = json.loads(line)
        text_lower = (d.get("text", "") + " " + d.get("context", "")).lower()
        if "nooon" in text_lower:
            nooon_msgs.append(d)
        if any(k in text_lower for k in keywords):
            fashion_msgs.append(d)

print(f"=== NOOON-SPECIFIC MESSAGES: {len(nooon_msgs)} ===")
for m in nooon_msgs[:30]:
    src = m.get("source", "?")
    ctx = m.get("context", "")[:200]
    txt = m.get("text", "")[:400]
    print(f"  [{src}] Context: {ctx}")
    print(f"  Jose: {txt}")
    print()

print(f"\n=== FASHION/BRAND RELATED MESSAGES: {len(fashion_msgs)} ===")
for m in fashion_msgs[:50]:
    src = m.get("source", "?")
    ctx = m.get("context", "")[:200]
    txt = m.get("text", "")[:400]
    print(f"  [{src}] Context: {ctx}")
    print(f"  Jose: {txt}")
    print()
