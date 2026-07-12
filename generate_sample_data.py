"""
generate_sample_data.py
------------------------
Generates a synthetic "dirty" Netflix movies dataset that reproduces every
data-quality problem handled by netflix_cleaning.ipynb:

  - Duplicate Movie_IDs
  - Messy titles (whitespace, newlines/tabs, ALL CAPS, trailing "!!!", very long titles)
  - Inconsistent genre casing / multiple genres joined with different separators
  - Release year outliers (< 1900 or > 2025 style values)
  - Duration in mixed formats ("3h 13m", "160mins", "113 min")
  - Rating stored as text with missing values
  - IMDb rating outliers (< 0 or > 10)
  - Votes formatted with thousands-separator commas
  - Budget with "$" signs and negative values
  - Revenue with commas and zero values
  - Date_Added in four different formats
  - Country name variants (USA / U.S. / US / United States, UK / U.K. / Britain)
  - Language abbreviations and inconsistent casing
  - Random missing values scattered across most columns

Run once with `python generate_sample_data.py` to (re)create
`netflix_movies_dirty.csv` in this folder.
"""

import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

N_ROWS = 1200

TITLE_WORDS = [
    "Shadow", "Midnight", "Silent", "Golden", "Broken", "Hidden", "Last",
    "Eternal", "Crimson", "Whispering", "Lost", "Forgotten", "Wild", "Iron",
    "Velvet", "Sacred", "Distant", "Frozen", "Burning", "Secret",
]
TITLE_NOUNS = [
    "Kingdom", "Horizon", "Legacy", "Empire", "Journey", "Storm", "Garden",
    "River", "City", "Dream", "Warrior", "House", "Voyage", "Symphony",
    "Chronicles", "Path", "Flame", "Echo", "Harbor", "Tiger",
]

GENRE_POOL = [
    "action", "ACTION", "Action", "drama", "DRAMA", "Drama",
    "comedy", "Comedy", "COMEDY", "thriller", "THRILLER", "Thriller",
    "romance", "ROMANCE", "Romance", "animation", "Animation", "documentary",
    "Horror", "horror", "Sci-Fi", "sci-fi", "Fantasy", "fantasy", "Crime",
]
GENRE_SEPARATORS = [",", "|", "/"]

COUNTRY_VARIANTS = {
    "United States": ["USA", "U.S.", "US", "U.S.A", "United States", "usa"],
    "United Kingdom": ["UK", "U.K.", "Britain", "United Kingdom", "uk"],
    "India": ["India", "INDIA", "india"],
    "South Korea": ["South Korea", "S. Korea", "Korea"],
    "France": ["France", "FRANCE"],
    "Germany": ["Germany", "GERMANY"],
    "Japan": ["Japan", "JAPAN"],
    "Spain": ["Spain", "SPAIN"],
    "Canada": ["Canada", "CANADA"],
    "Italy": ["Italy", "ITALY"],
}

LANGUAGE_VARIANTS = {
    "English": ["english", "English", "ENGLISH", "Eng"],
    "Hindi": ["Hindi", "Hin", "hindi"],
    "Korean": ["Korean", "Kor", "korean"],
    "German": ["German", "Ger", "german"],
    "French": ["French", "Fre", "french"],
    "Japanese": ["Japanese", "Jap", "japanese"],
    "Spanish": ["Spanish", "Spa", "spanish"],
    "Italian": ["Italian", "Ita", "italian"],
    "Chinese": ["Chinese", "chinese"],
}

FIRST_NAMES = ["James", "Maria", "Wei", "Aiko", "Liam", "Sofia", "Noah",
               "Ines", "Carlos", "Priya", "Yusuf", "Elena", "Tom", "Anna"]
LAST_NAMES = ["Smith", "Garcia", "Chen", "Tanaka", "Miller", "Rossi",
              "Kumar", "Nowak", "Silva", "Andersson", "Dubois", "Kim"]

AGE_RATINGS = ["G", "PG", "PG-13", "R", "NC-17", "TV-MA", "TV-14", "TV-PG"]

PRODUCTION_HOUSES = [
    "Silver Screen Studios", "Bluewave Pictures", "Nova Films",
    "Horizon Entertainment", "Redwood Productions", "Lantern Studios",
    "Orbit Media House", "Cascade Films", "Ember Pictures", "Northstar Studios",
]


def random_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def random_cast(n=3):
    return ", ".join(random_name() for _ in range(n))


def dirty_title(clean):
    """Randomly inject the messiness described in the notebook."""
    variant = random.random()
    if variant < 0.12:
        return clean.upper()
    if variant < 0.22:
        return f"  {clean}   "
    if variant < 0.30:
        return clean + "\n"
    if variant < 0.38:
        return clean + "!!!"
    if variant < 0.45:
        return f"{clean} (Extended Collector's Edition with Bonus Features and Director's Commentary)"
    if variant < 0.50:
        return "   ".join(clean.split())  # extra inner spaces
    return clean


def dirty_genre():
    primary = random.choice(GENRE_POOL)
    if random.random() < 0.4:
        secondary = random.choice(GENRE_POOL)
        sep = random.choice(GENRE_SEPARATORS)
        return f"{primary}{sep} {secondary}"
    if random.random() < 0.1:
        return f"\t{primary}"
    return primary


def dirty_year():
    if random.random() < 0.06:
        return random.choice([1890, 1895, 1899, 2099, 2087, 2150])
    return random.randint(1970, 2024)


def dirty_duration():
    minutes = random.randint(70, 200)
    fmt = random.random()
    if fmt < 0.3:
        h, m = divmod(minutes, 60)
        return f"{h}h {m}m"
    if fmt < 0.55:
        return f"{minutes}mins"
    if fmt < 0.8:
        return f"{minutes} min"
    return str(minutes)


def dirty_rating():
    return round(random.uniform(1.0, 10.0), 1)


def dirty_imdb_rating():
    if random.random() < 0.05:
        return round(random.choice([-1.0, -2.5, 11.0, 12.5, 15.0]), 1)
    return round(random.uniform(1.0, 10.0), 1)


def dirty_votes():
    v = random.randint(1000, 2000000)
    return f"{v:,}"


def dirty_budget():
    b = random.randint(500_000, 250_000_000)
    if random.random() < 0.08:
        b = -b
    return f"${b:,}"


def dirty_revenue():
    if random.random() < 0.08:
        r = 0
    else:
        r = random.randint(1_000_000, 1_800_000_000)
    return f"{r:,}"


def dirty_date():
    year = random.randint(2015, 2024)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    fmt = random.random()
    if fmt < 0.25:
        return f"{year}-{month:02d}-{day:02d}"
    if fmt < 0.5:
        return f"{day:02d}/{month:02d}/{year}"
    if fmt < 0.75:
        month_name = pd.Timestamp(year=year, month=month, day=1).strftime("%B")
        return f"{month_name} {day}, {year}"
    return f"{year}{month:02d}{day:02d}"


def dirty_country():
    official, variants = random.choice(list(COUNTRY_VARIANTS.items()))
    return random.choice(variants)


def dirty_language():
    official, variants = random.choice(list(LANGUAGE_VARIANTS.items()))
    return random.choice(variants)


def maybe_missing(value, p=0.06):
    return np.nan if random.random() < p else value


rows = []
for i in range(N_ROWS):
    clean_title_str = f"{random.choice(TITLE_WORDS)} {random.choice(TITLE_NOUNS)}"
    if random.random() < 0.15:
        clean_title_str += f" Part {random.choice(['One', 'Two', 'Three', 'II', 'III'])}"

    row = {
        "Movie_ID": f"NF{random.randint(1000, 9999)}",
        "Title": maybe_missing(dirty_title(clean_title_str), 0.05),
        "Genre": dirty_genre(),
        "Release_Year": dirty_year(),
        "Duration": maybe_missing(dirty_duration(), 0.11),
        "Rating": maybe_missing(dirty_rating(), 0.08),
        "IMDb_Rating": maybe_missing(dirty_imdb_rating(), 0.11),
        "Votes": maybe_missing(dirty_votes(), 0.05),
        "Director": maybe_missing(random_name(), 0.05),
        "Country": dirty_country(),
        "Language": dirty_language(),
        "Budget": maybe_missing(dirty_budget(), 0.08),
        "Revenue": maybe_missing(dirty_revenue(), 0.10),
        "Date_Added": dirty_date(),
        "Age_Rating": random.choice(AGE_RATINGS),
        "Cast": maybe_missing(random_cast(), 0.07),
        "Production_House": maybe_missing(random.choice(PRODUCTION_HOUSES), 0.05),
    }
    rows.append(row)

df = pd.DataFrame(rows)

# Inject ~15% duplicate Movie_IDs (2-4 repeats) to mirror the notebook's finding
dup_source = df.sample(frac=0.15, random_state=42)
dup_rows = []
for _, r in dup_source.iterrows():
    for _ in range(random.randint(1, 3)):
        dup_rows.append(r.copy())
df = pd.concat([df, pd.DataFrame(dup_rows)], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("netflix_movies_dirty.csv", index=False)
print(f"Generated netflix_movies_dirty.csv with {df.shape[0]} rows and {df.shape[1]} columns")
