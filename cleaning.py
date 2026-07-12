"""
cleaning.py
-----------
This module contains the full data-cleaning pipeline for the Netflix
movies dataset, ported directly from `netflix_cleaning.ipynb`.

Every cleaning function mirrors the logic used in the notebook. The
`run_pipeline()` function executes each step in order and records a
"before / after" snapshot for each step so the Streamlit app can display
step-by-step results, code, and outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Individual column-cleaning functions (unchanged logic from the notebook)
# --------------------------------------------------------------------------- #

def clean_title(title):
    """Fix whitespace, newlines, trailing '!', overly long titles, casing."""
    if pd.isnull(title):
        return np.nan
    title = str(title).replace("\n", " ").replace("\t", " ")
    title = " ".join(title.split())          # collapse extra/leading whitespace
    title = title.rstrip("!")                # remove trailing exclamation marks
    if "(" in title:                          # truncate overly long "(...)" suffixes
        title = title[: title.index("(")].strip()
    title = title.title()
    return title if title else np.nan


def clean_genre(genre):
    """Extract the primary genre and standardise casing/whitespace."""
    if pd.isnull(genre):
        return np.nan
    genre = str(genre).replace("\n", " ").replace("\t", " ")
    for sep in ["|", "/"]:
        genre = genre.replace(sep, ",")
    genre = genre.split(",")[0]
    genre = " ".join(genre.split())
    return genre.title()


def clean_year(year, lower=1900, upper=2025):
    """Replace unrealistic release years with NaN."""
    if pd.isnull(year):
        return np.nan
    try:
        year = float(year)
    except (TypeError, ValueError):
        return np.nan
    if year < lower or year > upper:
        return np.nan
    return year


def clean_duration(duration):
    """Normalise '3h 13m', '160mins', '113 min', '160' into total minutes."""
    if pd.isnull(duration):
        return np.nan
    duration = str(duration).strip()
    try:
        if "h" in duration:
            hours, minutes = duration.split("h")
            hours = int(hours.strip())
            minutes = minutes.replace("m", "").strip()
            minutes = int(minutes) if minutes else 0
            total = hours * 60 + minutes
        elif "min" in duration:
            total = int(duration.replace("mins", "").replace("min", "").strip())
        else:
            total = int(duration)
    except (ValueError, TypeError):
        return np.nan
    if total < 30 or total > 600:             # outlier guard
        return np.nan
    return total


def clean_rating(rating):
    """Convert Rating (0-10 scale from critics/users) to float."""
    try:
        return float(rating)
    except (TypeError, ValueError):
        return np.nan


def clean_imdb_rating(rating):
    """IMDb ratings must fall within [0, 10]."""
    if pd.isna(rating):
        return np.nan
    try:
        rating = float(rating)
    except (TypeError, ValueError):
        return np.nan
    if rating < 0 or rating > 10:
        return np.nan
    return rating


def clean_votes(votes):
    """Remove thousands-separator commas and convert to a number."""
    if pd.isnull(votes):
        return np.nan
    votes = str(votes).replace(",", "").strip()
    try:
        return int(votes)
    except ValueError:
        return np.nan


def clean_budget(budget):
    """Remove '$' / ',' and drop negative budgets."""
    if pd.isnull(budget):
        return np.nan
    budget = str(budget).replace("$", "").replace(",", "").strip()
    try:
        budget = float(budget)
    except ValueError:
        return np.nan
    if budget < 0:
        return np.nan
    return budget


def clean_revenue(revenue):
    """Remove ',' and drop zero/negative revenue."""
    if pd.isnull(revenue):
        return np.nan
    revenue = str(revenue).replace(",", "").strip()
    try:
        revenue = float(revenue)
    except ValueError:
        return np.nan
    if revenue <= 0:
        return np.nan
    return revenue


COUNTRY_MAP = {
    "U.S.": "United States", "US": "United States", "USA": "United States",
    "U.S.A": "United States", "usa": "United States",
    "U.K.": "United Kingdom", "UK": "United Kingdom", "uk": "United Kingdom",
    "Britain": "United Kingdom",
    "S. Korea": "South Korea",
}


def clean_country(value):
    if pd.isnull(value):
        return np.nan
    value = str(value).strip()
    return COUNTRY_MAP.get(value, value)


LANGUAGE_MAP = {
    "english": "English", "English": "English", "ENGLISH": "English", "Eng": "English",
    "Hin": "Hindi", "hindi": "Hindi",
    "Kor": "Korean", "korean": "Korean",
    "Ger": "German", "german": "German",
    "Fre": "French", "french": "French",
    "Jap": "Japanese", "japanese": "Japanese",
    "Spa": "Spanish", "spanish": "Spanish",
    "Ita": "Italian", "italian": "Italian",
    "Chinese": "Chinese", "chinese": "Chinese",
}


def clean_language(lang):
    if pd.isnull(lang):
        return np.nan
    lang = str(lang).strip()
    if lang in LANGUAGE_MAP:
        return LANGUAGE_MAP[lang]
    return lang.title()


def clean_date(series: pd.Series) -> pd.Series:
    """Parse a Date_Added column that mixes 4 different date formats."""
    s = series.astype(str)
    # Handle the compact "YYYYMMDD" format (8 digits, no separators) first
    compact_mask = s.str.match(r"^\d{8}$")
    s = s.where(~compact_mask, s.str.replace(
        r"^(\d{4})(\d{2})(\d{2})$", r"\1-\2-\3", regex=True))
    return pd.to_datetime(s, format="mixed", errors="coerce")


# --------------------------------------------------------------------------- #
# Step definitions used to build the "Data Cleaning Steps" page
# --------------------------------------------------------------------------- #

STEP_METADATA = [
    {
        "key": "duplicates_id",
        "name": "Step 1 — Remove Duplicate Movie IDs",
        "explanation": (
            "Every movie should have a unique Movie_ID. Rows sharing the same "
            "ID are true duplicates, so we keep only the first occurrence of "
            "each ID and drop the rest."
        ),
        "code": 'df = df.drop_duplicates(subset="Movie_ID", keep="first")',
    },
    {
        "key": "title",
        "name": "Step 2 — Clean the Title Column",
        "explanation": (
            "Titles contain stray newlines/tabs, extra whitespace, trailing "
            "'!!!' marks, ALL-CAPS text, and overly long descriptive suffixes. "
            "A custom function normalises whitespace, strips punctuation, "
            "truncates long parenthetical suffixes, and applies Title Case."
        ),
        "code": (
            "def clean_title(title):\n"
            "    if pd.isnull(title):\n"
            "        return np.nan\n"
            "    title = title.replace('\\n', ' ').replace('\\t', ' ')\n"
            "    title = ' '.join(title.split())\n"
            "    title = title.rstrip('!')\n"
            "    if '(' in title:\n"
            "        title = title[:title.index('(')].strip()\n"
            "    return title.title()\n\n"
            "df['Title'] = df['Title'].apply(clean_title)"
        ),
    },
    {
        "key": "genre",
        "name": "Step 3 — Standardise the Genre Column",
        "explanation": (
            "Genres appear in mixed casing and with multiple values joined by "
            "commas, pipes, or slashes. We extract only the primary genre and "
            "standardise its casing."
        ),
        "code": (
            "def clean_genre(genre):\n"
            "    if pd.isnull(genre):\n"
            "        return np.nan\n"
            "    genre = genre.replace('\\n', ' ').replace('\\t', ' ')\n"
            "    genre = genre.split(',')[0]\n"
            "    genre = ' '.join(genre.split())\n"
            "    return genre.title()\n\n"
            "df['Genre'] = df['Genre'].apply(clean_genre)"
        ),
    },
    {
        "key": "year",
        "name": "Step 4 — Fix Release Year Outliers",
        "explanation": (
            "Release_Year contains impossible values (e.g. 1890, 2099). Any "
            "year outside a realistic 1900–2025 window is replaced with NaN "
            "rather than guessed."
        ),
        "code": (
            "def clean_year(year):\n"
            "    if pd.isnull(year):\n"
            "        return np.nan\n"
            "    if year < 1900 or year > 2025:\n"
            "        return np.nan\n"
            "    return year\n\n"
            "df['Release_Year'] = df['Release_Year'].apply(clean_year)"
        ),
    },
    {
        "key": "duration",
        "name": "Step 5 — Normalise the Duration Column",
        "explanation": (
            "Duration values arrive in mixed formats such as '3h 13m', "
            "'160mins', and '113 min'. Each format is parsed and converted "
            "into a single number of total minutes; impossible durations are "
            "dropped."
        ),
        "code": (
            "def clean_duration(duration):\n"
            "    if pd.isnull(duration):\n"
            "        return np.nan\n"
            "    if 'h' in duration:\n"
            "        hours, minutes = duration.split('h')\n"
            "        return int(hours) * 60 + int(minutes.replace('m', ''))\n"
            "    elif 'min' in duration:\n"
            "        return int(duration.replace('mins', '').replace('min', ''))\n"
            "    return int(duration)\n\n"
            "df['Duration'] = df['Duration'].apply(clean_duration)"
        ),
    },
    {
        "key": "rating",
        "name": "Step 6 — Convert the Rating Column to Numeric",
        "explanation": (
            "Rating is stored as text. We convert it to a float so it can be "
            "used in calculations; invalid values become NaN."
        ),
        "code": "df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')",
    },
    {
        "key": "imdb_rating",
        "name": "Step 7 — Clean the IMDb_Rating Column",
        "explanation": (
            "IMDb ratings must fall between 0 and 10. Any value outside this "
            "domain rule is a data-entry error and is replaced with NaN."
        ),
        "code": (
            "def clean_imdb_rating(rating):\n"
            "    if pd.isna(rating):\n"
            "        return np.nan\n"
            "    if rating < 0 or rating > 10:\n"
            "        return np.nan\n"
            "    return rating\n\n"
            "df['IMDb_Rating'] = df['IMDb_Rating'].apply(clean_imdb_rating)"
        ),
    },
    {
        "key": "votes",
        "name": "Step 8 — Clean the Votes Column",
        "explanation": (
            "Votes are formatted with thousands-separator commas (e.g. "
            "'1,558,685'), which pandas treats as text. We strip the commas "
            "and convert to an integer."
        ),
        "code": (
            "def clean_votes(votes):\n"
            "    if pd.isnull(votes):\n"
            "        return np.nan\n"
            "    return int(votes.replace(',', ''))\n\n"
            "df['Votes'] = df['Votes'].apply(clean_votes)"
        ),
    },
    {
        "key": "budget",
        "name": "Step 9 — Clean the Budget Column",
        "explanation": (
            "Budget values include '$' signs, thousands separators, and some "
            "negative numbers. We strip formatting characters, convert to "
            "float, and set negative budgets to NaN."
        ),
        "code": (
            "def clean_budget(budget):\n"
            "    if pd.isnull(budget):\n"
            "        return np.nan\n"
            "    budget = float(budget.replace('$', '').replace(',', ''))\n"
            "    return np.nan if budget < 0 else budget\n\n"
            "df['Budget'] = df['Budget'].apply(clean_budget)"
        ),
    },
    {
        "key": "revenue",
        "name": "Step 10 — Clean the Revenue Column",
        "explanation": (
            "Revenue is formatted with commas, and some rows report 0 — an "
            "unrealistic value for a released movie. We strip the commas and "
            "treat 0 or negative revenue as missing."
        ),
        "code": (
            "def clean_revenue(revenue):\n"
            "    if pd.isnull(revenue):\n"
            "        return np.nan\n"
            "    revenue = float(revenue.replace(',', ''))\n"
            "    return np.nan if revenue <= 0 else revenue\n\n"
            "df['Revenue'] = df['Revenue'].apply(clean_revenue)"
        ),
    },
    {
        "key": "date",
        "name": "Step 11 — Standardise the Date_Added Column",
        "explanation": (
            "Dates appear in four different formats (ISO, DD/MM/YYYY, 'Month "
            "Day, Year', and compact YYYYMMDD). We normalise the compact "
            "format first, then let pandas' flexible parser handle the rest."
        ),
        "code": (
            "df['Date_Added'] = pd.to_datetime(\n"
            "    df['Date_Added'], format='mixed', errors='coerce'\n"
            ")"
        ),
    },
    {
        "key": "country",
        "name": "Step 12 — Standardise the Country Column",
        "explanation": (
            "The same country appears under many aliases (USA, U.S., US, "
            "United States). A lookup dictionary maps every known variant to "
            "one official name."
        ),
        "code": (
            "country_map = {'USA': 'United States', 'US': 'United States', ...}\n"
            "df['Country'] = df['Country'].apply(\n"
            "    lambda v: country_map.get(v, v)\n"
            ")"
        ),
    },
    {
        "key": "language",
        "name": "Step 13 — Standardise the Language Column",
        "explanation": (
            "Languages are abbreviated ('Hin', 'Kor') or inconsistently cased "
            "('english', 'ENGLISH'). A lookup dictionary maps abbreviations to "
            "full names; anything else is Title Cased."
        ),
        "code": (
            "lang_map = {'Eng': 'English', 'Hin': 'Hindi', ...}\n"
            "df['Language'] = df['Language'].apply(\n"
            "    lambda l: lang_map.get(l, l.title())\n"
            ")"
        ),
    },
    {
        "key": "near_duplicates",
        "name": "Step 14 — Remove Near-Duplicate Rows",
        "explanation": (
            "Even after removing duplicate IDs, rows can still share the same "
            "Title and Release_Year. We drop these near-duplicates, keeping "
            "the first occurrence."
        ),
        "code": 'df = df.drop_duplicates(subset=["Title", "Release_Year"], keep="first")',
    },
    {
        "key": "fillna",
        "name": "Step 15 — Fill Remaining Missing Values",
        "explanation": (
            "Numeric columns (Duration, Rating, IMDb_Rating, Votes) are "
            "filled with their median, which is robust to outliers. Budget, "
            "Revenue, Release_Year, Director, Cast, and Production_House are "
            "intentionally left as NaN — guessing these would be worse than "
            "leaving them missing."
        ),
        "code": (
            "for col in ['Duration', 'Rating', 'IMDb_Rating', 'Votes']:\n"
            "    df[col] = df[col].fillna(df[col].median())"
        ),
    },
    {
        "key": "dtypes",
        "name": "Step 16 — Final Data Type Conversion",
        "explanation": (
            "Columns such as Release_Year, Votes, and Duration should be "
            "whole numbers, but contain NaNs. We convert them to pandas' "
            "nullable 'Int64' type, which supports integers and missing "
            "values together."
        ),
        "code": (
            "df[['Release_Year', 'Votes', 'Duration']] = df[\n"
            "    ['Release_Year', 'Votes', 'Duration']\n"
            "].astype('Int64')"
        ),
    },
]


def _snapshot(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


def run_pipeline(raw_df: pd.DataFrame):
    """
    Runs every cleaning step in sequence and returns:
      - cleaned_df: the fully cleaned DataFrame
      - step_results: list of dicts with before/after snapshots per step
      - overall: {"before": snapshot, "after": snapshot}
    """
    df = raw_df.copy()
    overall_before = _snapshot(df)
    step_results = []

    def record(key_lookup, df_before, df_after, note=""):
        meta = next(s for s in STEP_METADATA if s["key"] == key_lookup)
        step_results.append({
            **meta,
            "before": _snapshot(df_before),
            "after": _snapshot(df_after),
            "note": note,
        })

    # 1. Duplicate Movie_ID removal
    before = df.copy()
    df = df.drop_duplicates(subset="Movie_ID", keep="first")
    record("duplicates_id", before, df,
           f"Removed {before.shape[0] - df.shape[0]} duplicate-ID rows.")

    # 2. Title
    before = df.copy()
    df["Title"] = df["Title"].apply(clean_title)
    record("title", before, df, "Standardised whitespace, casing, and length.")

    # 3. Genre
    before = df.copy()
    df["Genre"] = df["Genre"].apply(clean_genre)
    record("genre", before, df, "Extracted primary genre and standardised casing.")

    # 4. Release Year
    before = df.copy()
    df["Release_Year"] = pd.to_numeric(df["Release_Year"], errors="coerce")
    df["Release_Year"] = df["Release_Year"].apply(clean_year)
    record("year", before, df, "Replaced unrealistic years with NaN.")

    # 5. Duration
    before = df.copy()
    df["Duration"] = df["Duration"].apply(clean_duration)
    record("duration", before, df, "Converted all formats to total minutes.")

    # 6. Rating
    before = df.copy()
    df["Rating"] = df["Rating"].apply(clean_rating)
    record("rating", before, df, "Converted text ratings to numeric floats.")

    # 7. IMDb Rating
    before = df.copy()
    df["IMDb_Rating"] = df["IMDb_Rating"].apply(clean_imdb_rating)
    record("imdb_rating", before, df, "Removed out-of-range IMDb ratings.")

    # 8. Votes
    before = df.copy()
    df["Votes"] = df["Votes"].apply(clean_votes)
    record("votes", before, df, "Removed comma separators; converted to integer.")

    # 9. Budget
    before = df.copy()
    df["Budget"] = df["Budget"].apply(clean_budget)
    record("budget", before, df, "Removed '$'/',' and dropped negative budgets.")

    # 10. Revenue
    before = df.copy()
    df["Revenue"] = df["Revenue"].apply(clean_revenue)
    record("revenue", before, df, "Removed ',' and dropped zero/negative revenue.")

    # 11. Date_Added
    before = df.copy()
    df["Date_Added"] = clean_date(df["Date_Added"])
    record("date", before, df, "Parsed 4 mixed date formats into datetime64.")

    # 12. Country
    before = df.copy()
    df["Country"] = df["Country"].apply(clean_country)
    record("country", before, df, "Mapped country aliases to official names.")

    # 13. Language
    before = df.copy()
    df["Language"] = df["Language"].apply(clean_language)
    record("language", before, df, "Mapped abbreviations to full language names.")

    # 14. Near-duplicate rows
    before = df.copy()
    df = df.drop_duplicates(subset=["Title", "Release_Year"], keep="first")
    record("near_duplicates", before, df,
           f"Removed {before.shape[0] - df.shape[0]} near-duplicate rows.")

    # 15. Fill remaining missing numeric values with median
    before = df.copy()
    for col in ["Duration", "Rating", "IMDb_Rating", "Votes"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    df["Title"] = df["Title"].fillna("Unknown")
    record("fillna", before, df, "Filled numeric columns with their median value.")

    # 16. Final dtype conversion
    before = df.copy()
    for col in ["Release_Year", "Votes", "Duration"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    record("dtypes", before, df, "Converted whole-number columns to nullable Int64.")

    overall_after = _snapshot(df)
    return df, step_results, {"before": overall_before, "after": overall_after}
