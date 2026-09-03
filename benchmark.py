import os
import shutil
import time
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from src.load_and_extract_text import extract_text_from_pdf

load_dotenv()

# --- Configurations ---
PDF_FILE_PATH = "uploads/paper.pdf"  # Path to your PDF
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# 10 Ground-Truth Technical Questions for Attention Is All You Need
TEST_DATASET = [
    {
        "question": "What is the total number of layers N in the Transformer encoder and decoder stacks?",
        "ground_truth": "Both encoder and decoder stacks consist of N = 6 identical layers.",
        "keywords": ["6 identical layers", "N = 6", "N=6"],
    },
    {
        "question": "What are the exact dimensions of d_model and the inner feed-forward layer dff?",
        "ground_truth": "The model dimension d_model is 512, and the inner-layer dimension dff is 2048.",
        "keywords": ["512", "2048"],
    },
    {
        "question": "How many parallel attention heads are used, and what are their dimensions dk and dv?",
        "ground_truth": "There are h = 8 parallel attention heads with dk = dv = 64.",
        "keywords": ["8", "64", "dk = dv = 64"],
    },
    {
        "question": "What BLEU score did the Transformer big model achieve on the WMT 2014 English-to-German task?",
        "ground_truth": "The Transformer big model achieved 28.4 BLEU on English-to-German.",
        "keywords": ["28.4"],
    },
    {
        "question": "What was the hardware setup and training schedule for the base Transformer model?",
        "ground_truth": "The base model was trained on 8 NVIDIA P100 GPUs for 100,000 steps (12 hours).",
        "keywords": ["8", "P100", "100,000", "12 hours"],
    },
    {
        "question": "What are the optimizer hyperparameters beta1, beta2, epsilon, and warmup steps in Adam?",
        "ground_truth": "Adam optimizer with beta1=0.9, beta2=0.98, epsilon=10^-9, and warmup_steps=4000.",
        "keywords": ["0.9", "0.98", "10^-9", "4000"],
    },
    {
        "question": "What dropout rate Pdrop and label smoothing value were applied to the base model?",
        "ground_truth": "Dropout rate Pdrop = 0.1 and label smoothing epsilon_ls = 0.1.",
        "keywords": ["0.1"],
    },
    {
        "question": "What is the maximum path length and per-layer complexity for self-attention vs recurrent layers?",
        "ground_truth": "Self-attention has O(1) path length and O(n^2 * d) complexity; recurrent has O(n) path length and O(n * d^2) complexity.",
        "keywords": ["O(1)", "O(n)", "O(n^2"],
    },
    {
        "question": "What beam size and length penalty alpha were used during inference decoding?",
        "ground_truth": "Beam size of 4 and length penalty alpha = 0.6.",
        "keywords": ["4", "0.6"],
    },
    {
        "question": "What F1 score did the 4-layer Transformer achieve on the WSJ English constituency parsing task?",
        "ground_truth": "The 4-layer Transformer achieved 92.7 F1 in the semi-supervised setting on WSJ Section 23.",
        "keywords": ["92.7"],
    },
]

# --- Initialize Models ---
embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL, temperature=0)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# --- Vector DB Builders ---
def build_normal_db(text: str, embedder: Embeddings) -> Chroma:
    """Normal (Fixed-size Character) Chunking without recursive logic."""
    dir_path = "chroma_db_normal"
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    splitter = CharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separator=" ",
    )
    docs = splitter.split_documents([Document(page_content=text)])
    return Chroma.from_documents(
        docs,
        embedder,
        collection_name="normal_store",
        persist_directory=dir_path,
    )


def build_semantic_db(text: str, embedder: Embeddings) -> Chroma:
    """Semantic Chunking (Splits based on semantic sentence distance)."""
    dir_path = "chroma_db_semantic"
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    splitter = SemanticChunker(
        embedder,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=60,  # Lower threshold prevents over-sized chunks
    )
    docs = splitter.split_documents([Document(page_content=text)])
    return Chroma.from_documents(
        docs,
        embedder,
        collection_name="semantic_store",
        persist_directory=dir_path,
    )


# --- Retrievers ---
def retrieve_standard(vectordb: Chroma, query: str, k: int = 3):
    return vectordb.similarity_search(query, k=k)


def retrieve_with_rerank(
    vectordb: Chroma, query: str, k_fetch: int = 12, k_top: int = 3
):
    initial_docs = vectordb.similarity_search(query, k=k_fetch)
    if not initial_docs:
        return []
    doc_texts = [d.page_content for d in initial_docs]
    scores = reranker.predict([[query, t] for t in doc_texts])
    top_idx = np.argsort(scores)[::-1][:k_top]
    return [initial_docs[i] for i in top_idx]


# --- Metrics Calculation ---
def calculate_retrieval_metrics(retrieved_docs, keywords: list):
    hit = 0
    mrr = 0.0
    for rank, doc in enumerate(retrieved_docs, start=1):
        content = doc.page_content.lower()
        if any(kw.lower() in content for kw in keywords):
            hit = 1
            if mrr == 0.0:
                mrr = 1.0 / rank
    return hit, mrr


def generate_and_score_answer(retrieved_docs, question: str, ground_truth: str):
    context = "\n".join([d.page_content for d in retrieved_docs])
    prompt = f"Answer using only context:\n{context}\n\nQ: {question}\nA:"
    pred_answer = llm.invoke(prompt).content

    eval_prompt = f"""Compare the Predicted Answer with the Ground Truth.
Ground Truth: {ground_truth}
Predicted Answer: {pred_answer}

Rate factual accuracy from 0.0 (completely wrong or missing) to 1.0 (exact factual match).
Return ONLY the float number (e.g., 0.85):"""
    try:
        score_txt = llm.invoke(eval_prompt).content.strip()
        score = float(score_txt.split()[0])
    except Exception:
        score = 0.5
    return min(max(score, 0.0), 1.0)


# --- Main Execution & Plotting ---
def main():
    print("📄 Extracting text from PDF...")
    raw_text = extract_text_from_pdf(PDF_FILE_PATH)

    print("⚙️ Building Vector Databases (Normal & Semantic)...")
    normal_db = build_normal_db(raw_text, embedder)
    semantic_db = build_semantic_db(raw_text, embedder)

    pipelines = {
        "1. Normal (Baseline)": lambda q: retrieve_standard(
            normal_db, q, k=3
        ),
        "2. Semantic Chunking": lambda q: retrieve_standard(
            semantic_db, q, k=3
        ),
        "3. Normal + Reranker": lambda q: retrieve_with_rerank(
            normal_db, q, k_fetch=12, k_top=3
        ),
        "4. Semantic + Reranker": lambda q: retrieve_with_rerank(
            semantic_db, q, k_fetch=12, k_top=3
        ),
    }

    records = []
    print(f"\n🚀 Running comparison across {len(TEST_DATASET)} questions...")

    for idx, item in enumerate(TEST_DATASET, start=1):
        q = item["question"]
        gt = item["ground_truth"]
        kw = item["keywords"]

        print(f"[{idx}/{len(TEST_DATASET)}] Testing: {q[:50]}...")
        for p_name, retriever_fn in pipelines.items():
            start_t = time.time()
            docs = retriever_fn(q)
            hit, mrr = calculate_retrieval_metrics(docs, kw)
            accuracy = generate_and_score_answer(docs, q, gt)
            latency = round(time.time() - start_t, 2)

            records.append(
                {
                    "Pipeline": p_name,
                    "Question": q,
                    "Hit_Rate@3": hit,
                    "MRR": mrr,
                    "Answer_Accuracy": accuracy,
                    "Latency (s)": latency,
                }
            )

    df = pd.DataFrame(records)
    summary = (
        df.groupby("Pipeline")[
            ["Hit_Rate@3", "MRR", "Answer_Accuracy", "Latency (s)"]
        ]
        .mean()
        .reset_index()
    )

    baseline_acc = summary.loc[
        summary["Pipeline"] == "1. Normal (Baseline)", "Answer_Accuracy"
    ].values[0]
    baseline_mrr = summary.loc[
        summary["Pipeline"] == "1. Normal (Baseline)", "MRR"
    ].values[0]

    summary["Accuracy_Gain_%"] = (
        (summary["Answer_Accuracy"] - baseline_acc) / max(baseline_acc, 0.01)
    ) * 100
    summary["MRR_Gain_%"] = (
        (summary["MRR"] - baseline_mrr) / max(baseline_mrr, 0.01)
    ) * 100

    print("\n================ BENCHMARK RESULTS ================")
    print(summary.to_string(index=False))
    print("===================================================")

    # Plot Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Retrieval Metrics (Hit Rate & MRR)
    x = np.arange(len(summary))
    w = 0.35
    axes[0].bar(
        x - w / 2,
        summary["Hit_Rate@3"] * 100,
        width=w,
        label="Hit Rate@3 (%)",
        color="#1f77b4",
    )
    axes[0].bar(
        x + w / 2,
        summary["MRR"] * 100,
        width=w,
        label="MRR (x100)",
        color="#ff7f0e",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        [p.split(". ")[1] for p in summary["Pipeline"]],
        rotation=15,
        ha="right",
        fontsize=9,
    )
    axes[0].set_ylabel("Score (%)")
    axes[0].set_title(
        "Retrieval Quality: Hit Rate@3 & MRR", fontweight="bold"
    )
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend()

    # Panel 2: Percentage Improvement over Normal Baseline
    bars = axes[1].bar(
        summary["Pipeline"],
        summary["Accuracy_Gain_%"],
        color=["#7f7f7f", "#2ca02c", "#17becf", "#9467bd"],
        edgecolor="black",
    )
    axes[1].set_ylabel("Improvement over Normal Baseline (%)")
    axes[1].set_title(
        "Overall Accuracy Gain vs. Normal Baseline", fontweight="bold"
    )
    axes[1].set_xticklabels(
        [p.split(". ")[1] for p in summary["Pipeline"]],
        rotation=15,
        ha="right",
        fontsize=9,
    )
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        axes[1].annotate(
            f"{h:+.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3 if h >= 0 else -10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("normal_vs_semantic_benchmark.png", dpi=300)
    print(
        "\n✅ Comparison chart saved as 'normal_vs_semantic_benchmark.png'"
    )
    plt.show()


if __name__ == "__main__":
    main()