# 🧬 GeoVectorSearch

**GeoVectorSearch** is a lightweight Python SDK and command-line tool for discovering high-quality **GEO gene expression datasets** relevant to a disease or biological condition — optimized for **differential expression (DE) analysis**.

It combines **semantic search** using sentence embeddings with optional **GPT-based filtering** to help you rapidly identify suitable datasets for your research or pipeline.

---

## 🔍 Features

* ✅ **Natural language search** for GEO datasets
* ⚡ **Fast vector search** using [FAISS](https://faiss.ai/) and prebuilt sentence embeddings
* 🧠 **Optional GPT filtering** to assess dataset quality for DE analysis
* 🧬 Supports **microarray** and **RNA-seq** datasets
* 🖥️ **Interactive CLI** for a smooth user experience
* 🧩 Easy to integrate into larger pipelines or SDKs
* 💾 **Save results locally** for downstream analysis

---

## 📦 Installation

Install using your preferred package manager:

```bash
uv pip install geo-pysearch
```

Or clone the repository and install locally:

```bash
git clone https://github.com/Tinfloz/geo-vector-search.git
cd geo-vector-search
uv pip install .
```

---

## 🧪 Example (Python SDK)

```python
from geo_pysearch.sdk import search_datasets

results = search_datasets(
    query="duchenne muscular dystrophy",
    dataset_type="microarray",
    top_k=50,
    use_gpt_filter=True,
    return_all_gpt_results=True
)

print(results.head())
```

Convenience methods:

```python
from geo_pysearch.sdk import search_microarray, search_rnaseq

search_microarray("breast cancer")
search_rnaseq("lung fibrosis", use_gpt_filter=True)
```

---

## 💻 Example (CLI)

Launch the interactive CLI:

```bash
geo-search
```

* Use the arrow keys to select dataset type and filtering options
* Enter your disease query
* Results will be saved to a local CSV file in a new directory
* Review and use the datasets for downstream DE analysis

---

## 🧠 GPT Filtering (Optional)

If enabled, the SDK uses GPT to evaluate whether each dataset is suitable for **differential gene expression analysis**. You can configure GPT behavior with:

* Adjustable confidence thresholds

## 📁 Project Structure

```
gse-pysearch/
├── geo_pysearch/
│   ├── data/                # Prebuilt FAISS index, vectors, metadata
│   ├── vector_search/
│   │   ├── vector_search.py
│   │   ├── gpt_filter.py
│   ├── sdk.py               # Main SDK interface
│   └── cli.py               # CLI implementation
├── examples/                # Example usage scripts
├── .env                     # Optional environment variables

```

---

## 🛠️ Requirements

* Python 3.12+
* `faiss-cpu`, `pandas`, `sentence-transformers`

---

## 📖 License

**GNU General Public License v3.0**

This project is licensed under the [GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html), which guarantees end users the freedom to run, study, share, and modify the software.

If you redistribute or modify this software, your contributions must also be licensed under the same terms.

---

## References

This project implements semantic query generation and evidence extraction strategies inspired by:

1. Deka, P., Jurek-Loughrey, A., & others. (2022). *Evidence Extraction to Validate Medical Claims in Fake News Detection*. International Conference on Health Information Science, pp. 3–15.

2. Deka, P., & Jurek-Loughrey, A. (2021). *Unsupervised Keyword Combination Query Generation from Online Health Related Content for Evidence-Based Fact Checking*. The 23rd International Conference on Information Integration and Web Intelligence, pp. 267–277.