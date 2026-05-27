"""
Hotel Guest Feedback Analyzer — ISOM5240 Group Project
Streamlit app chaining three Hugging Face pipelines:

  Pipeline 1 : fine-tuned Cardiff RoBERTa -> 3-class sentiment (negative/neutral/positive)
  Pipeline 2a: distilbart-mnli            -> zero-shot aspect tagging (what complaints are about)
  Pipeline 2b: distilbart-cnn             -> summarization (gist of negative reviews)

Two modes:
  - Single review  : paste one review -> sentiment + aspects
  - CSV upload     : upload many reviews -> full manager report (stats + themes + summary)
"""

import io
import time
from collections import Counter

import pandas as pd
import streamlit as st
from huggingface_hub import login
from transformers import pipeline
import altair as alt

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
# CHANGE THIS to your pushed fine-tuned model repo
SENTIMENT_MODEL = "ktysu1026/tripadvisor-cardiff-3class"
ASPECT_MODEL = "valhalla/distilbart-mnli-12-1"
SUMMARY_MODEL = "sshleifer/distilbart-cnn-12-6"

# The sentiment model is PRIVATE on Hugging Face, so we log in with a token.
# On Streamlit Cloud: add HF_TOKEN under App settings -> Secrets.
# Locally: create .streamlit/secrets.toml with HF_TOKEN = "hf_...".
HF_TOKEN = st.secrets.get("HF_TOKEN", None)
if HF_TOKEN:
    login(token=HF_TOKEN)

ASPECTS = [
    "cleanliness and tidiness of the room", "staff and service", "hotel location",
    "noise and quietness", "value for money", "bed and room comfort", "food and breakfast",
]
ASPECT_THRESHOLD = 0.4
HYPOTHESIS_TEMPLATE = "This hotel review talks about {}."

st.set_page_config(page_title="Hotel Guest Feedback Analyzer", page_icon="🏨", layout="wide")


# ----------------------------------------------------------------------------
# Load models once and cache (critical for Streamlit's ~1GB memory limit)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment():
    return pipeline("text-classification", model=SENTIMENT_MODEL,
                    truncation=True, max_length=256, token=HF_TOKEN)


@st.cache_resource(show_spinner=False)
def load_aspect():
    return pipeline("zero-shot-classification", model=ASPECT_MODEL)


@st.cache_resource(show_spinner=False)
def load_summarizer():
    return pipeline("summarization", model=SUMMARY_MODEL)


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def classify_sentiment(texts):
    """Return list of lowercase labels: negative/neutral/positive."""
    clf = load_sentiment()
    preds = clf(texts, batch_size=16)
    if isinstance(preds, dict):
        preds = [preds]
    return [p["label"].lower() for p in preds]


def tag_aspects(texts):
    """Return list of sets of aspects above threshold for each text."""
    asp = load_aspect()
    out = asp(texts, candidate_labels=ASPECTS, multi_label=True,
              hypothesis_template=HYPOTHESIS_TEMPLATE)
    if isinstance(out, dict):
        out = [out]
    result = []
    for res in out:
        scores = dict(zip(res["labels"], res["scores"]))
        result.append({a for a in ASPECTS if scores[a] >= ASPECT_THRESHOLD})
    return result


def summarize_negatives(neg_texts):
    if not neg_texts:
        return "No negative reviews to summarize."
    summ = load_summarizer()
    joined = " ".join(t[:300] for t in neg_texts)[:3000]
    out = summ(joined, max_length=120, min_length=30, do_sample=False, truncation=True)
    return out[0]["summary_text"]


SENT_EMOJI = {"negative": "🔴", "neutral": "🟡", "positive": "🟢"}


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🏨 Hotel Guest Feedback Analyzer")
st.caption("Turn guest reviews into an at-a-glance manager briefing — sentiment, complaint themes, and a summary.")

mode = st.radio("Choose input mode:", ["Single review", "Upload CSV of reviews"], horizontal=True)

# ---------------------------- SINGLE REVIEW MODE ----------------------------
if mode == "Single review":
    text = st.text_area("Paste a guest review:", height=160,
                        placeholder="e.g. The room was clean but the street noise kept us up all night...")
    if st.button("Analyze", type="primary"):
        if not text.strip():
            st.warning("Please enter a review first.")
        else:
            with st.spinner("Analyzing..."):
                label = classify_sentiment([text])[0]
                aspects = tag_aspects([text])[0]
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sentiment")
                st.markdown(f"## {SENT_EMOJI.get(label, '')} {label.capitalize()}")
            with col2:
                st.subheader("Topics mentioned")
                if aspects:
                    for a in aspects:
                        st.markdown(f"- {a}")
                else:
                    st.markdown("_No specific aspect detected above threshold._")

# ------------------------------ CSV UPLOAD MODE -----------------------------
else:
    st.markdown("Upload a CSV with a column of review text. "
                "The app finds the review column automatically (looks for `review`, else uses the first text column).")
    file = st.file_uploader("Upload CSV", type=["csv"])

    sample = "review\n\"Lovely stay, the staff were wonderful and the room spotless.\"\n\"Terrible. Dirty bathroom and noisy all night.\"\n\"It was okay, nothing special for the price.\"\n"
    st.download_button("Download a sample CSV", sample, "sample_reviews.csv", "text/csv")

    if file is not None:
        df = pd.read_csv(file)

        # find the review column
        review_col = None
        for c in df.columns:
            if c.strip().lower() == "review":
                review_col = c
                break
        if review_col is None:
            # fall back to first object/text column
            text_cols = [c for c in df.columns if df[c].dtype == object]
            review_col = text_cols[0] if text_cols else df.columns[0]

        reviews = df[review_col].astype(str).tolist()
        st.success(f"Loaded {len(reviews)} reviews from column '{review_col}'.")

        # cap to keep runtime/memory sane on Streamlit free tier
        MAX_ROWS = 200
        if len(reviews) > MAX_ROWS:
            st.info(f"Analyzing the first {MAX_ROWS} reviews to stay within resource limits.")
            reviews = reviews[:MAX_ROWS]

        if st.button("Generate manager report", type="primary"):
            with st.spinner("Classifying sentiment..."):
                labels = classify_sentiment(reviews)
            counts = Counter(labels)
            total = len(labels)
            neg_reviews = [r for r, l in zip(reviews, labels) if l == "negative"]

            # ---- summary metrics ----
            st.header("📋 Guest Feedback Report")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total reviews", total)
            c2.metric("🔴 Negative", f"{counts.get('negative',0)} ({counts.get('negative',0)/total*100:.0f}%)")
            c3.metric("🟡 Neutral", f"{counts.get('neutral',0)} ({counts.get('neutral',0)/total*100:.0f}%)")
            c4.metric("🟢 Positive", f"{counts.get('positive',0)} ({counts.get('positive',0)/total*100:.0f}%)")

            # ---- sentiment bar chart ----
            chart_df = pd.DataFrame({
              "sentiment": ["negative", "neutral", "positive"],
              "count": [
                counts.get("negative", 0),
                counts.get("neutral", 0),
                counts.get("positive", 0),
              ],
            })

            chart = (
              alt.Chart(chart_df)
              .mark_bar()
              .encode(
                  x="count:Q",
                  y=alt.Y("sentiment:N", sort=None),
                  color="sentiment:N",
              )
              .properties(height=200)
            )

            st.altair_chart(chart, use_container_width=True)

            # ---- complaint themes among negatives ----
            st.subheader("⚠️ Top complaint themes (negative reviews)")
            if neg_reviews:
                with st.spinner("Tagging complaint themes..."):
                    neg_aspects = tag_aspects(neg_reviews)
                theme_counts = Counter()
                for s in neg_aspects:
                    theme_counts.update(s)
                if theme_counts:
                    theme_df = pd.DataFrame(
                        {"count": [c for _, c in theme_counts.most_common()]},
                        index=[a for a, _ in theme_counts.most_common()],
                    )
                    st.bar_chart(theme_df)
                else:
                    st.write("No specific themes detected above threshold.")
            else:
                st.write("No negative reviews — nothing to triage. 🎉")

            # ---- abstractive summary of negatives ----
            st.subheader("📝 Summary of negative feedback")
            with st.spinner("Summarizing..."):
                summary = summarize_negatives(neg_reviews)
            st.info(summary)

            # ---- downloadable per-review table ----
            st.subheader("🔎 Per-review results")
            out_df = pd.DataFrame({"review": reviews, "sentiment": labels})
            st.dataframe(out_df, use_container_width=True)
            csv_buf = io.StringIO()
            out_df.to_csv(csv_buf, index=False)
            st.download_button("Download results CSV", csv_buf.getvalue(),
                            "feedback_results.csv", "text/csv")

st.divider()
st.caption("ISOM5240 Group Project · Pipelines: fine-tuned RoBERTa sentiment + distilbart-mnli aspects + distilbart-cnn summary")
