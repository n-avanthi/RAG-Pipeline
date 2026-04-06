# download_papers.py
# Run from your project root: python download_papers.py
# Requires: pip install requests

import os, time, requests

PAPERS = [
    # Transformers & Attention
    ("1706.03762", "attention_is_all_you_need.pdf"),
    ("2005.14165", "gpt3.pdf"),
    ("1810.04805", "bert.pdf"),

    # Efficient / Sparse Transformers
    ("2009.14794", "longformer.pdf"),
    ("2009.06732", "linformer.pdf"),
    ("2205.01068", "flashattention.pdf"),

    # Reinforcement Learning
    ("1707.06347", "ppo.pdf"),
    ("2005.05719", "dreamer_v2.pdf"),
    ("1509.02971", "ddpg.pdf"),

    # Optimization & Training
    ("1412.6980", "adam.pdf"),
    ("1803.05407", "cyclical_lr.pdf"),
    ("1607.06450", "layer_norm.pdf"),

    # Vision & Multimodal
    ("2010.11929", "vit.pdf"),
    ("2103.00020", "clip.pdf"),
    ("2102.12092", "dino.pdf"),

    # RAG & Retrieval
    ("2005.11401", "rag_original.pdf"),
    ("2212.10560", "self_rag.pdf"),
    ("2310.11511", "corrective_rag.pdf"),

    # Alignment & RLHF
    ("2203.02155", "instructgpt.pdf"),
    ("2305.18290", "direct_preference_optimization.pdf"),
]

OUT = "data/raw_docs"
os.makedirs(OUT, exist_ok=True)

for arxiv_id, fname in PAPERS:
    dest = os.path.join(OUT, fname)
    if os.path.exists(dest):
        print(f"  skip {fname}")
        continue
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print(f"Downloading {fname} ...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        print(f"OK ({len(r.content)//1024} KB)")
    except Exception as e:
        print(f"FAILED: {e}")
    time.sleep(1.2)  # be polite to arXiv

print("\nDone. Check data/raw_docs/")