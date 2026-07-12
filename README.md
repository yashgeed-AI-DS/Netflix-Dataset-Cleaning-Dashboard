# 🎬 Netflix Dataset Cleaning Dashboard

An interactive, professional **Streamlit** web application that turns a
Jupyter Notebook data-cleaning project into a full end-to-end dashboard —
from a messy, real-world Netflix movies dataset to a clean, analysis-ready
one.

Built as a portfolio project to demonstrate practical **Python + Pandas**
data-wrangling skills through a polished, deployable web app.

---

## ✨ Features

| Section | What it shows |
|---|---|
| 🏠 **Home** | Project objective and tech stack |
| 📋 **Dataset Overview** | `df.head()`, shape, `df.info()`, `df.describe()` |
| 🔍 **Missing Values Analysis** | Table, percentages, bar chart, heatmap |
| 🧹 **Data Cleaning Steps** | 16 steps — explanation, code, and measurable result for each |
| ⚖️ **Before vs After Comparison** | Rows / columns / missing values / duplicates, with charts |
| ✅ **Final Dataset** | Search, filter, and sort the cleaned data |
| ⬇️ **Download Cleaned Dataset** | One-click CSV export |
| ℹ️ **About Project** | Summary, tech stack, and author info |

---

## 🗂️ Project Structure

```
netflix_app/
├── app.py                        # Main Streamlit application
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── utils/
│   └── cleaning.py                # Cleaning pipeline (ported from the notebook)
└── assets/
    ├── style.css                  # Custom dashboard styling
    ├── netflix_movies_dirty.csv   # Bundled sample "dirty" dataset
    └── generate_sample_data.py    # Script used to generate the sample dataset
```

> 📌 The original notebook (`netflix_cleaning.ipynb`) did not ship with its
> source CSV, so a synthetic dataset reproducing the exact same data-quality
> issues (duplicate IDs, messy titles, mixed date formats, currency symbols,
> outlier ratings, etc.) is bundled in `assets/`. You can also upload your
> own CSV with the same 17 columns from the sidebar.

---

## 🧹 What Gets Cleaned

1. Duplicate `Movie_ID` rows
2. Messy `Title` values (whitespace, newlines, casing, long strings)
3. Inconsistent `Genre` formatting (commas / pipes / slashes)
4. Unrealistic `Release_Year` values
5. Mixed `Duration` formats (`"3h 13m"`, `"160mins"`, `"113 min"`)
6. `Rating` stored as text
7. Out-of-range `IMDb_Rating` values
8. Comma-formatted `Votes`
9. `Budget` with `$` signs and negative values
10. `Revenue` with commas and zero values
11. `Date_Added` in 4 different formats
12. `Country` name variants (USA / U.S. / US → United States)
13. `Language` abbreviations (Hin → Hindi, Kor → Korean, …)
14. Near-duplicate rows (same Title + Release_Year)
15. Remaining missing numeric values (filled with median)
16. Final data type conversion (nullable `Int64`)

Every step is implemented in `utils/cleaning.py`, matching the logic in the
original notebook.

---

## 🚀 Run Locally

```bash
# 1. Clone or download this project
cd netflix_app

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this project to a **public GitHub repository** (include `app.py`,
   `requirements.txt`, `utils/`, and `assets/`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **"New app"**, select your repository, branch, and set the
   main file path to `app.py`.
4. Click **"Deploy"** — Streamlit Cloud will install `requirements.txt`
   automatically and build your app.
5. Once deployed, you'll get a shareable URL like
   `https://your-app-name.streamlit.app` — perfect for your resume,
   GitHub README, or LinkedIn post.

**Tips for a smooth deployment:**
- Keep `assets/netflix_movies_dirty.csv` under GitHub's file-size limits
  (it's well under 1 MB by default).
- If you update `requirements.txt`, Streamlit Cloud will auto-rebuild on
  the next push.
- Use the **"Manage app"** menu on Streamlit Cloud to view logs if the
  app fails to start.

---

## 🛠️ Tech Stack

- **Python** — core language
- **Pandas** & **NumPy** — data manipulation and cleaning
- **Streamlit** — web application framework
- **Matplotlib** & **Seaborn** — static visualizations (missing-value heatmap)
- **Plotly** — interactive charts
- **streamlit-option-menu** — styled sidebar navigation

---

## 👨‍💻 Author

**Yash Geed**
Artificial Intelligence & Data Science Student

---

## 📄 License

This project is free to use for learning, portfolio, and educational purposes.
