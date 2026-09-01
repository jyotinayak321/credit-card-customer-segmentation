# Credit Card Customer Behaviour Segmentation
### Using Agglomerative Hierarchical Clustering

An unsupervised ML project that groups credit-card customers into behavioural
segments (e.g. low-activity, regular purchasers, high-value, cash-advance-heavy)
using **Agglomerative Clustering**.

---

## 📁 Folder Structure

```
credit_card_segmentation/
│
├── data/
│   ├── README.md                  <- where to download the real dataset from
│   └── credit_card_customers.csv  <- (you add this, OR generate synthetic data)
│
├── notebooks/
│   └── customer_segmentation.ipynb  <- full step-by-step notebook (EDA -> clusters)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py        <- loads CSV, basic inspection
│   ├── preprocessing.py      <- cleaning, missing values, scaling
│   ├── clustering.py         <- dendrogram, AgglomerativeClustering, silhouette
│   ├── visualization.py      <- plots: dendrogram, PCA scatter, cluster profile
│   └── generate_sample_data.py  <- creates a FAKE dataset so you can test the
│                                    whole pipeline before downloading the real one
│
├── outputs/                  <- auto-created: dendrogram.png, cluster_plot.png,
│                                 cluster_profile.csv (generated when you run code)
│
├── requirements.txt
└── README.md                 <- this file
```

**Why this structure?**
| Folder | Purpose | Analogy |
|---|---|---|
| `data/` | Raw input only, never edited by code | The raw ingredients in your kitchen |
| `src/` | Reusable functions (the "engine") | The recipe steps written once, used many times |
| `notebooks/` | Where you actually run things, see plots | The dining table where you eat the dish |
| `outputs/` | Saved results (images, CSVs) | Photos of the finished dish for your portfolio |

Separating `src/` from `notebooks/` is an **industry best practice**: notebooks
are for exploration, `.py` files are for reusable, testable code. In interviews,
mentioning this separation shows you think like a software engineer, not just
a script-writer.

---

## 🚀 How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get the data** — two options:
   - **Real data (recommended):** Download from Kaggle
     `https://www.kaggle.com/datasets/arjunbhasin2013/ccdata`
     and save it as `data/credit_card_customers.csv`
   - **No Kaggle account yet? Test with synthetic data:**
     ```bash
     python src/generate_sample_data.py
     ```
     This creates a realistic *fake* `data/credit_card_customers.csv` with the
     same 18 columns, so you can run the entire pipeline today and swap in
     real data later — nothing else changes.

3. **Run the notebook**
   ```bash
   jupyter notebook notebooks/customer_segmentation.ipynb
   ```
   Run cells top to bottom. It will save `dendrogram.png`, `cluster_plot.png`
   and `cluster_profile.csv` into `outputs/`.

   OR run everything as a script:
   ```bash
   python src/clustering.py
   ```

---

## ⚠️ Important Notes (from the project guide)

- There is **no target column** — this is unsupervised learning, so we don't
  evaluate with "accuracy".
- **`CUST_ID` is excluded** from clustering — it's an identifier, not behaviour.
- Cluster numbers (0,1,2,3) are **arbitrary** — always compare cluster
  *profiles* (means), not the numbers themselves.
- Don't claim a customer is "risky" or "creditworthy" just because of their
  cluster — describe clusters using neutral behavioural language only.
