"""
app.py  -  CineMatch (Streamlit front end)

Run from the project root:
    python -m streamlit run app.py

Design: dark navy "AI / wireframe" look (glowing globe, sparkle stars, thin header
lines, numbered blue squares, outlined frames, italic blue sub-titles).
Optional: put .streamlit/config.toml next to app.py for matching widget colours.

Only imports and calls the team's functions:
    preprocess.load_data()                           -> movies, ratings
    content_based.get_similar_movies(title, n)       -> [(title, score), ...]
    hybrid.get_hybrid_recs(user_id, title, n, alpha) -> [(title, score), ...]
    hybrid.get_user_recs(user_id, n)                 -> [(title, score), ...]
"""
import html
import math
import os
import random
import re
from collections import Counter
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st

from content_based import get_similar_movies
from hybrid import get_hybrid_recs, get_user_recs
from preprocess import load_data

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

MASCOT = "🧸"  # change to 🤖 🐼 🦊 🍿 or any emoji you like

# Final numbers from evaluate.py / tune_alpha.py (edit here if you re-run them)
FINAL_RESULTS = pd.DataFrame({
    "Model": ["Content-Based", "Collaborative", "Hybrid (alpha 0.8)"],
    "RMSE": [1.0079, 0.9205, 0.9078],
    "MAE": [0.7705, 0.7181, 0.7116],
    "Precision@10": [0.0200, 0.1885, 0.1870],
    "Recall@10": [0.0232, 0.1710, 0.1675],
}).set_index("Model")

ALPHA_TUNING = pd.DataFrame({
    "alpha": [0.5, 0.7, 0.8, 0.9, 0.95],
    "RMSE": [0.9180, 0.9073, 0.9078, 0.9122, 0.9159],
    "MAE": [0.7216, 0.7121, 0.7116, 0.7136, 0.7153],
    "Precision@10": [0.0445, 0.1645, 0.1870, 0.1880, 0.1885],
    "Recall@10": [0.0433, 0.1521, 0.1675, 0.1691, 0.1706],
}).set_index("alpha")

# genre -> (emoji, glow colour)
GENRE_STYLE = {
    "Action": ("💥", "#ff6b6b"), "Adventure": ("🧭", "#ffa94d"),
    "Animation": ("🎨", "#ffd43b"), "Children": ("🧸", "#ff8787"),
    "Comedy": ("😂", "#fcc419"), "Crime": ("🕵️", "#adb5bd"),
    "Documentary": ("🎥", "#38d9a9"), "Drama": ("🎭", "#b197fc"),
    "Fantasy": ("🧙", "#e599f7"), "Film-Noir": ("🌃", "#868e96"),
    "Horror": ("👻", "#9775fa"), "IMAX": ("🏟️", "#4dabf7"),
    "Musical": ("🎶", "#f783ac"), "Mystery": ("🔍", "#748ffc"),
    "Romance": ("💕", "#faa2c1"), "Sci-Fi": ("🚀", "#3bc9db"),
    "Thriller": ("😱", "#ff8787"), "War": ("🪖", "#8ce99a"),
    "Western": ("🤠", "#e9b872"),
}

MOODS = {
    "Any mood": [], "😂 Laugh": ["Comedy"], "💥 Adrenaline": ["Action", "Adventure"],
    "😱 Scared": ["Horror", "Thriller"], "💕 Romantic": ["Romance"],
    "🚀 Mind-bending": ["Sci-Fi", "Mystery"], "🎭 Deep": ["Drama"],
    "🧸 Family": ["Animation", "Children"], "🔍 Whodunit": ["Crime", "Mystery"],
}


# ----------------------------------------------------------------- artwork (SVG)
def globe_svg(color="#7da2ff"):
    """Wireframe globe drawn from ellipses (latitude + longitude lines)."""
    r, c = 190, 200
    parts = [f'<circle cx="{c}" cy="{c}" r="{r}"/>']
    for k in range(-4, 5):                       # latitude rings
        lat = k * math.pi / 10
        rx = r * math.cos(lat)
        parts.append(f'<ellipse cx="{c}" cy="{c + r * math.sin(lat)}" rx="{rx:.1f}" '
                     f'ry="{rx * 0.22:.1f}"/>')
    for k in range(0, 8):                        # longitude lines
        rx = abs(r * math.cos(k * math.pi / 8))
        parts.append(f'<ellipse cx="{c}" cy="{c}" rx="{rx:.1f}" ry="{r}"/>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">'
            f'<g fill="none" stroke="{color}" stroke-width="1.1">{"".join(parts)}</g></svg>')


def net_svg(seed=3):
    """Glowing neural-network constellation with a purple core."""
    rng = random.Random(seed)
    pts = []
    while len(pts) < 70:
        x, y = rng.uniform(-1, 1), rng.uniform(-1, 1)
        if x * x + y * y <= 1:
            pts.append((200 + x * 185, 200 + y * 185))
    lines, dots = [], []
    for i, (x, y) in enumerate(pts):
        near = sorted(range(len(pts)), key=lambda j: (pts[j][0] - x) ** 2 + (pts[j][1] - y) ** 2)[1:3]
        for j in near:
            lines.append(f'<line x1="{x:.0f}" y1="{y:.0f}" x2="{pts[j][0]:.0f}" y2="{pts[j][1]:.0f}"/>')
        dots.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{rng.choice([1.8, 2.4, 3.2])}"/>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400"><defs>'
            '<radialGradient id="core"><stop offset="0" stop-color="#fff"/>'
            '<stop offset=".35" stop-color="#b197fc"/><stop offset="1" stop-color="#5f3dc4" '
            'stop-opacity="0"/></radialGradient></defs>'
            f'<g stroke="#5c7cfa" stroke-opacity=".55" stroke-width="1">{"".join(lines)}</g>'
            f'<g fill="#a5c4ff">{"".join(dots)}</g>'
            '<circle cx="200" cy="200" r="46" fill="url(#core)"/>'
            '<circle cx="200" cy="200" r="30" fill="none" stroke="#d0bfff" stroke-width="2"/></svg>')


STAR_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><defs>'
            '<linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" '
            'stop-color="#4dabf7"/><stop offset="1" stop-color="#e599f7"/></linearGradient></defs>'
            '<path fill="url(#g)" d="M12 0C13 8 16 11 24 12 16 13 13 16 12 24 11 16 8 13 0 12 '
            '8 11 11 8 12 0Z"/></svg>')


def data_uri(svg):
    return "data:image/svg+xml;utf8," + quote(svg)


def img(svg, cls, style=""):
    return f'<img class="{cls}" style="{style}" src="{data_uri(svg)}">'


GLOBE_URI = data_uri(globe_svg())

CSS = """
<style>
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp button, .stApp input, .stApp textarea, [class^="cm-"], [class*=" cm-"]
  {font-family: 'Exo 2', 'Segoe UI', sans-serif;}
.stApp {color: #e8eeff;
  background: radial-gradient(1100px 560px at 85% -8%, rgba(54,78,200,.55), rgba(54,78,200,0) 62%),
              linear-gradient(160deg,#080c2b 0%,#0a1040 55%,#070a22 100%);}
header[data-testid="stHeader"] {background: transparent;}
[data-testid="stToolbar"], #MainMenu, footer {visibility: hidden;}
section[data-testid="stSidebar"] {background: linear-gradient(180deg,#0b1245,#080c2e);
  border-right: 1px solid rgba(140,170,255,.25);}
[data-testid="stMetricValue"] {font-weight: 800; color: #fff;}
[data-testid="stMetricLabel"] {font-style: italic; color: #7c9bff;}
.stTabs [data-baseweb="tab-list"] {gap: 6px; border-bottom: 1px solid rgba(160,185,255,.35);}
.stTabs [data-baseweb="tab"] {font-weight: 700; letter-spacing: .4px; font-size: 1.02rem; padding: 10px 18px;}
.stTabs [aria-selected="true"] {color: #7da2ff;}
.stTabs [data-baseweb="tab-highlight"] {background: #7da2ff; height: 3px;}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(90deg,#4c6ef5,#9775fa); border: 0; border-radius: 999px;
  padding: .65rem 2.2rem; font-weight: 800; letter-spacing: .5px; color: #fff;
  animation: cmpulse 2.2s infinite;}
.stButton > button {border-radius: 999px;}
@keyframes cmpulse {0% {box-shadow: 0 0 0 0 rgba(120,140,255,.65);}
  70% {box-shadow: 0 0 0 16px rgba(120,140,255,0);} 100% {box-shadow: 0 0 0 0 rgba(120,140,255,0);}}
@keyframes cmfade {from {opacity: 0; transform: translateY(14px);} to {opacity: 1; transform: none;}}
@keyframes cmpopin {0% {opacity: 0; transform: translateY(24px) scale(.92);} 100% {opacity: 1; transform: none;}}
@keyframes cmslide {0% {transform: translateX(-380px) rotate(-30deg); opacity: 0;}
  60% {transform: translateX(16px) rotate(8deg); opacity: 1;} 100% {transform: none; opacity: 1;}}
@keyframes cmpop {0% {transform: scale(0); opacity: 0;} 70% {transform: scale(1.1);} 100% {transform: scale(1); opacity: 1;}}
@keyframes cmwave {0%,100% {transform: rotate(0);} 25% {transform: rotate(20deg);} 75% {transform: rotate(-14deg);}}
@keyframes cmfloat {0%,100% {transform: translateY(0);} 50% {transform: translateY(-10px);}}
@keyframes cmtwinkle {0%,100% {opacity: .35; transform: scale(.8) rotate(0);} 50% {opacity: 1; transform: scale(1.15) rotate(20deg);}}
@keyframes cmrise {0% {transform: translateY(0); opacity: 0;} 15% {opacity: .9;} 100% {transform: translateY(-460px); opacity: 0;}}
@keyframes cmspin {to {transform: rotate(360deg);}}
@keyframes cmglow {0%,100% {transform: scale(1); opacity: .9;} 50% {transform: scale(1.25); opacity: 1;}}
@keyframes cmcycle {0% {opacity: 0; transform: translateY(12px);} 6% {opacity: 1; transform: none;}
  30% {opacity: 1;} 36% {opacity: 0; transform: translateY(-10px);} 100% {opacity: 0;}}
.cm-topline {display: flex; align-items: center; gap: 14px; font-size: .95rem; opacity: .9; margin-top: 4px;}
.cm-topline i {flex: 1; height: 1px; background: rgba(160,185,255,.5);}
.cm-page-title {font-size: 2.6rem; font-weight: 800; color: #fff; line-height: 1.05; margin: 10px 0 0 0;
  text-shadow: 0 0 24px rgba(100,130,255,.45);}
.cm-italic {font-size: 1.35rem; font-style: italic; font-weight: 600; color: #5c7cfa; margin: 2px 0 12px 0;}
.cm-footer {display: flex; align-items: center; gap: 14px; font-size: .9rem; opacity: .75; margin-top: 26px;}
.cm-footer i {flex: 1; height: 1px; background: rgba(160,185,255,.4);}
.cm-hero {position: relative; overflow: hidden; border-radius: 26px; padding: 34px 48px 30px 48px;
  min-height: 430px; border: 1px solid rgba(140,170,255,.3); animation: cmfade .9s ease;
  background: radial-gradient(700px 420px at 78% 40%, rgba(70,100,255,.35), rgba(70,100,255,0) 70%),
              linear-gradient(135deg,#0a1148,#0d1b6b 60%,#091037);
  box-shadow: 0 0 70px rgba(76,110,245,.28);}
.cm-hero-globe {position: absolute; right: -70px; top: -90px; width: 560px; opacity: .6;
  animation: cmspin 140s linear infinite;}
.cm-burst {position: absolute; right: 21%; top: 42%; width: 130px; height: 130px;
  background: radial-gradient(circle, #fff 0 5%, rgba(170,195,255,.9) 14%, rgba(90,120,255,.35) 34%, transparent 64%);
  animation: cmglow 3s ease-in-out infinite;}
.cm-burst::before {content: ""; position: absolute; left: 50%; top: -35%; width: 2px; height: 170%;
  background: linear-gradient(transparent, #fff, transparent);}
.cm-burst::after {content: ""; position: absolute; top: 50%; left: -35%; height: 2px; width: 170%;
  background: linear-gradient(90deg, transparent, #fff, transparent);}
.cm-spark {position: absolute; bottom: -30px; color: #8fb0ff; pointer-events: none;
  animation: cmrise 9s linear infinite;}
.cm-hero > .cm-layer {position: relative; z-index: 2;}
.cm-small {font-size: 1.25rem; opacity: .95;}
.cm-mascot-row {display: flex; align-items: center; gap: 16px; margin: 6px 0 4px 0; flex-wrap: wrap;}
.cm-mascot {font-size: 5.2rem; line-height: 1;
  animation: cmslide 1.4s ease-out both, cmfloat 2.4s ease-in-out 1.5s infinite;}
.cm-bubble {position: relative; background: rgba(8,14,60,.85); color: #fff; font-weight: 700;
  font-size: 1.45rem; padding: 12px 24px; border-radius: 6px; border: 1.5px solid #7da2ff;
  box-shadow: 0 0 22px rgba(110,140,255,.55); animation: cmpop .6s ease-out 1.2s both;}
.cm-bubble::before {content: ""; position: absolute; left: -12px; top: 50%; margin-top: -10px;
  border-style: solid; border-width: 10px 12px 10px 0; border-color: transparent #7da2ff transparent transparent;}
.cm-wave {display: inline-block; transform-origin: 70% 70%; animation: cmwave 1.2s ease-in-out 1.6s infinite;}
.cm-hero-title {font-size: clamp(3rem, 8.4vw, 6.4rem); font-weight: 800; letter-spacing: 3px;
  color: #fff; line-height: 1; margin-top: 26px; text-shadow: 0 0 34px rgba(110,140,255,.6);}
.cm-hero-sub {font-size: 2.7rem; font-style: italic; font-weight: 600; color: #5c7cfa; line-height: 1.1;}
.cm-tagline {position: relative; height: 1.9em; margin-top: 10px; font-size: 1.15rem; color: #d0dcff;}
.cm-tagline span {position: absolute; left: 0; opacity: 0; animation: cmcycle 9s infinite;}
.cm-by {position: absolute; right: 36px; bottom: 22px; font-size: 1rem; opacity: .9; z-index: 2;}
.cm-star {width: 38px; filter: drop-shadow(0 0 8px rgba(150,140,255,.9)); animation: cmtwinkle 2.6s ease-in-out infinite;}
.cm-frame {border: 1.5px solid rgba(190,205,255,.7); padding: 16px 22px; margin: 8px 0; height: 100%;
  background: rgba(10,18,70,.45); animation: cmfade 1s ease;}
.cm-frame h4 {font-style: italic; font-weight: 600; color: #5c7cfa; margin: 0 0 6px 0; font-size: 1.25rem;}
.cm-num {display: inline-flex; width: 28px; height: 28px; align-items: center; justify-content: center;
  background: #4c6ef5; color: #fff; font-weight: 800; margin-right: 12px; flex: none;}
.cm-stepx {display: flex; align-items: flex-start; max-width: 520px; margin: 14px 0; animation: cmfade 1s ease;}
.cm-tl {display: grid; grid-template-columns: repeat(3, 1fr); gap: 22px; position: relative; margin: 10px 0 18px 0;}
.cm-tl::before {content: ""; position: absolute; top: 12px; left: 8%; right: 8%;
  border-top: 2px dotted rgba(190,205,255,.75);}
.cm-tl-item {position: relative; padding-top: 38px;}
.cm-tl-dot {position: absolute; top: 0; left: 50%; margin-left: -12px; width: 24px; height: 24px; border-radius: 50%;
  background: #dbe4ff; color: #1b2a80; text-align: center; font-weight: 900; line-height: 24px;}
.cm-tl-box {background: #4c6ef5; color: #fff; padding: 14px 16px; font-size: .92rem; min-height: 96px;}
.cm-tl-box b {font-size: 1.05rem;}
.cm-hwrap {position: relative;}
.cm-netimg {position: absolute; right: 0; top: -4px; width: 150px; opacity: .95;}
.cm-grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(195px, 1fr)); gap: 20px; margin-top: 10px;}
.cm-card {position: relative; border-radius: 6px; overflow: hidden; border: 1px solid rgba(120,150,255,.32);
  background: linear-gradient(180deg, rgba(20,34,118,.8), rgba(9,14,56,.92));
  transition: transform .25s, box-shadow .25s, border-color .25s; animation: cmpopin .6s ease both;}
.cm-card:hover {transform: translateY(-8px); border-color: #7da2ff; box-shadow: 0 12px 38px rgba(76,110,245,.5);}
.cm-rank {position: absolute; top: 10px; left: 10px; z-index: 3; background: #4c6ef5; color: #fff;
  font-weight: 800; font-size: .8rem; padding: 3px 9px;}
.cm-tile {position: relative; height: 158px; display: flex; flex-direction: column; align-items: center;
  justify-content: center; text-align: center; padding: 10px; overflow: hidden;}
.cm-tile::before {content: ""; position: absolute; inset: -10%; background: url("__GLOBE__") center/130% no-repeat;
  opacity: .2;}
.cm-emoji {position: relative; font-size: 3.1rem; line-height: 1;}
.cm-tile-title {position: relative; font-weight: 700; font-size: .95rem; margin-top: 8px; color: #fff;}
.cm-poster {width: 100%; height: 285px; object-fit: cover; display: block;}
.cm-body {padding: 10px 12px 14px 12px;}
.cm-ctitle {font-weight: 700; font-size: .95rem; margin-bottom: 2px; color: #fff;}
.cm-meta {font-size: .8rem; opacity: .85; margin-bottom: 6px;}
.cm-stars {color: #ffd43b; letter-spacing: 1px;}
.cm-chip {display: inline-block; font-size: .7rem; padding: 2px 9px; border: 1px solid rgba(160,185,255,.45);
  margin: 2px 4px 2px 0; color: #c9d6ff;}
.cm-bar {height: 6px; background: rgba(255,255,255,.12); margin-top: 8px;}
.cm-fill {height: 100%; background: linear-gradient(90deg,#4dabf7,#b197fc);}
.cm-why {font-size: .72rem; opacity: .75; margin-top: 5px; font-style: italic;}
.cm-spot {position: relative; display: flex; gap: 22px; align-items: center; flex-wrap: wrap;
  border: 1.5px solid rgba(190,205,255,.7); padding: 20px 24px; margin: 8px 0 20px 0;
  background: linear-gradient(120deg, rgba(40,60,190,.35), rgba(120,70,200,.22)); animation: cmfade .9s ease;
  box-shadow: 0 0 40px rgba(76,110,245,.25);}
.cm-spot-tile {position: relative; width: 190px; height: 190px; display: flex; align-items: center;
  justify-content: center; overflow: hidden; flex: none;}
.cm-spot-tile::before {content: ""; position: absolute; inset: -10%; background: url("__GLOBE__") center/130% no-repeat; opacity: .25;}
.cm-spot-tile .cm-emoji {font-size: 5rem;}
.cm-spot-poster {height: 270px; flex: none;}
.cm-spot-body {flex: 1; min-width: 230px;}
.cm-spot-label {font-style: italic; font-weight: 600; color: #7da2ff; font-size: 1.1rem;}
.cm-spot-title {font-size: 2rem; font-weight: 800; color: #fff; line-height: 1.1; margin: 4px 0;}
.cm-ring {width: 92px; height: 92px; border-radius: 50%; display: flex; align-items: center;
  justify-content: center; flex: none; box-shadow: 0 0 24px rgba(130,140,255,.5);}
.cm-ring span {width: 70px; height: 70px; border-radius: 50%; background: #0b1245; color: #fff;
  display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.15rem;}
</style>
"""
CSS = CSS.replace("__GLOBE__", GLOBE_URI)
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading data...")
def get_data():
    movies, ratings = load_data()
    stats = ratings.groupby("movieId")["rating"].agg(avg="mean", n_ratings="count")
    meta = movies.drop_duplicates("title").merge(
        stats, left_on="movieId", right_index=True, how="left")
    meta["avg"] = meta["avg"].fillna(0.0)
    meta["n_ratings"] = meta["n_ratings"].fillna(0).astype(int)
    meta["genre_list"] = meta["genres"].fillna("").str.split("|")
    return movies, ratings, meta.set_index("title")


def to_pairs(result):
    """Turn whatever a model returns into a list of (title, score)."""
    if result is None:
        return []
    if isinstance(result, pd.DataFrame):
        cols = list(result.columns)
        title_col = "title" if "title" in cols else cols[0]
        score_col = next((c for c in ("score", "similarity_score", "predicted_score")
                          if c in cols), None)
        scores = result[score_col] if score_col else [0.0] * len(result)
        return [(t, float(s)) for t, s in zip(result[title_col], scores)]
    pairs = []
    for item in result:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            pairs.append((item[0], float(item[1])))
        else:
            pairs.append((str(item), 0.0))
    return pairs


@st.cache_resource(show_spinner="Warming up the models (first start can take a minute)...")
def warm_up(first_user):
    get_user_recs(first_user, n=1)
    return True


@st.cache_data(show_spinner="Finding movies for you...")
def hybrid_recs(user_id, title, pool, alpha):
    return to_pairs(get_hybrid_recs(user_id, title, n=pool, alpha=alpha))


@st.cache_data(show_spinner="Finding similar movies...")
def similar_recs(title, pool):
    return to_pairs(get_similar_movies(title, n=pool))


@st.cache_data(show_spinner="Finding picks for this member...")
def user_recs(user_id, pool):
    return to_pairs(get_user_recs(user_id, n=pool))


def safe(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        st.error(f"Something went wrong: {e}")
        return []


# ----------------------------------------------------------------- helpers
def clean_title(title):
    m = re.match(r"^(.*)\s\((\d{4})\)\s*$", title)
    name, year = (m.group(1), m.group(2)) if m else (title, None)
    m2 = re.match(r"^(.*), (The|A|An)$", name)
    if m2:
        name = f"{m2.group(2)} {m2.group(1)}"
    return name, year


@st.cache_data(show_spinner=False)
def poster_url(title, key):
    if not key:
        return None
    name, year = clean_title(title)
    params = {"api_key": key, "query": name}
    if year:
        params["year"] = year
    try:
        r = requests.get("https://api.themoviedb.org/3/search/movie",
                         params=params, timeout=5)
        results = r.json().get("results", [])
        if results and results[0].get("poster_path"):
            return "https://image.tmdb.org/t/p/w342" + results[0]["poster_path"]
    except Exception:
        pass
    return None


def build_table(pairs, n, ref_genres=(), tiebreak=False, exclude=(), genres=None):
    """Add rating info + 'why' genres, apply filters, return top n rows."""
    gf = genres if genres else genre_filter
    rows = []
    for title, score in pairs:
        if title not in meta.index or title in exclude:
            continue
        info = meta.loc[title]
        glist = info["genre_list"]
        if info["n_ratings"] < min_ratings:
            continue
        if gf and not set(glist) & set(gf):
            continue
        shared = [g for g in glist if g in ref_genres]
        rows.append({
            "Title": title, "Score": float(score),
            "Avg rating": round(float(info["avg"]), 2),
            "Ratings": int(info["n_ratings"]),
            "Genres": ", ".join(glist),
            "Why": ", ".join(shared) if shared else "-",
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if tiebreak:  # equal scores -> prefer the better-known / better-rated movie
        df["_s"] = df["Score"].round(4)
        df = df.sort_values(["_s", "Ratings", "Avg rating"],
                            ascending=False).drop(columns="_s")
    df = df.head(n).reset_index(drop=True)
    df.index = range(1, len(df) + 1)
    return df


def style_for(glist):
    return next((GENRE_STYLE[g] for g in glist if g in GENRE_STYLE), ("🎬", "#748ffc"))


def stars(avg):
    full = max(0, min(5, int(round(avg))))
    return "★" * full + "☆" * (5 - full)


def pct_of(r, top):
    return int(round(max(0.0, min(1.0, r["Score"] / top if top else 0)) * 100))


def card_html(r, label, top, rank, delay):
    title = r["Title"]
    name, _ = clean_title(title)
    glist = [g for g in r["Genres"].split(", ") if g]
    emoji, accent = style_for(glist)
    poster = poster_url(title, api_key)
    if poster:
        media = f'<img class="cm-poster" src="{html.escape(poster)}">'
    else:
        media = (f'<div class="cm-tile" style="background:radial-gradient(circle at 50% 38%,'
                 f'{accent}66 0%,rgba(8,13,51,0) 68%),linear-gradient(160deg,#12236f,#080d33)">'
                 f'<div class="cm-emoji" style="text-shadow:0 0 24px {accent}">{emoji}</div>'
                 f'<div class="cm-tile-title">{html.escape(name)}</div></div>')
    pct = pct_of(r, top)
    chips = "".join(f'<span class="cm-chip">{html.escape(g)}</span>' for g in glist[:3])
    why = (f'<div class="cm-why">Because: {html.escape(r["Why"])}</div>'
           if r["Why"] not in ("-", "") else "")
    return (f'<div class="cm-card" style="animation-delay:{delay:.2f}s">'
            f'<div class="cm-rank">#{rank}</div>{media}<div class="cm-body">'
            f'<div class="cm-ctitle">{html.escape(title)}</div>'
            f'<div class="cm-meta"><span class="cm-stars">{stars(r["Avg rating"])}</span> '
            f'{r["Avg rating"]} · {r["Ratings"]:,}</div>{chips}'
            f'<div class="cm-bar"><div class="cm-fill" style="width:{pct}%"></div></div>'
            f'<div class="cm-meta" style="margin-top:4px">{pct}% {label}</div>{why}</div></div>')


def spot_html(r, label, top, spot_label):
    title = r["Title"]
    name, _ = clean_title(title)
    glist = [g for g in r["Genres"].split(", ") if g]
    emoji, accent = style_for(glist)
    poster = poster_url(title, api_key)
    if poster:
        media = f'<img class="cm-poster cm-spot-poster" src="{html.escape(poster)}">'
    else:
        media = (f'<div class="cm-spot-tile" style="background:radial-gradient(circle at 50% 40%,'
                 f'{accent}77 0%,rgba(8,13,51,0) 70%),linear-gradient(160deg,#12236f,#080d33)">'
                 f'<div class="cm-emoji" style="text-shadow:0 0 30px {accent}">{emoji}</div></div>')
    pct = pct_of(r, top)
    chips = "".join(f'<span class="cm-chip">{html.escape(g)}</span>' for g in glist)
    why = (f'<div class="cm-why">Because: {html.escape(r["Why"])}</div>'
           if r["Why"] not in ("-", "") else "")
    return (f'<div class="cm-spot">{media}<div class="cm-spot-body">'
            f'<div class="cm-spot-label">{spot_label}</div>'
            f'<div class="cm-spot-title">{html.escape(title)}</div>'
            f'<div class="cm-meta"><span class="cm-stars">{stars(r["Avg rating"])}</span> '
            f'{r["Avg rating"]} · {r["Ratings"]:,} ratings</div>{chips}{why}</div>'
            f'<div class="cm-ring" style="background:conic-gradient(#4dabf7 0%,#b197fc {pct}%,'
            f'rgba(255,255,255,.14) {pct}%)"><span>{pct}%</span></div></div>')


def show_cards(df, label="match", normalize=False, spotlight=False,
               spot_label="🏆 YOUR #1 MATCH"):
    if df.empty:
        st.info("No movies match your filters. Try lowering the minimum number of "
                "ratings or clearing the genre filter.")
        return
    top = float(df["Score"].max()) if normalize else 1.0
    rows = [r for _, r in df.iterrows()]
    rest, start = rows, 1
    if spotlight:
        st.markdown(spot_html(rows[0], label, top, spot_label), unsafe_allow_html=True)
        rest, start = rows[1:], 2
    if rest:
        cards = "".join(card_html(r, label, top, i + start, i * 0.06)
                        for i, r in enumerate(rest))
        st.markdown(f'<div class="cm-grid">{cards}</div>', unsafe_allow_html=True)


def multi_pick_recs(picks, user_id, pool, alpha):
    """Average the hybrid scores over several favourite movies."""
    totals = Counter()
    for p in picks:
        for t, s in safe(hybrid_recs, user_id, p, pool, alpha):
            if t not in picks:
                totals[t] += s
    return [(t, s / len(picks)) for t, s in totals.most_common()]


def user_profile(uid):
    r = ratings[ratings["userId"] == uid].merge(
        movies[["movieId", "title", "genres"]], on="movieId")
    liked = r[r["rating"] >= 4]
    gen = Counter(g for gs in liked["genres"] for g in gs.split("|")
                  if g != "(no genres listed)")
    top = r.sort_values("rating", ascending=False).head(5)[["title", "rating"]]
    return gen, top, len(r)


def header(label, title, subtitle="", art=""):
    sub = f'<div class="cm-italic">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(f'<div class="cm-hwrap">{art}<div class="cm-topline"><span>{html.escape(label)}'
                f'</span><i></i></div><div class="cm-page-title">{html.escape(title)}</div>{sub}</div>',
                unsafe_allow_html=True)


def footer():
    st.markdown('<div class="cm-footer"><span>CineMatch - 2026</span><i></i></div>',
                unsafe_allow_html=True)


def chart(fn, data, colors):
    try:
        fn(data, color=colors)
    except TypeError:
        fn(data)


def fun_facts():
    most = meta["n_ratings"].idxmax()
    good = meta[meta["n_ratings"] >= 100]["avg"].idxmax()
    gc = Counter(g for gl in meta["genre_list"] for g in gl
                 if g and g != "(no genres listed)")
    tg, tn = gc.most_common(1)[0]
    return [
        ("🏆 Most-rated movie", f"{most} · {int(meta.loc[most, 'n_ratings']):,} ratings"),
        ("⭐ Highest rated (100+ ratings)", f"{good} · {meta.loc[good, 'avg']:.2f} / 5"),
        ("🎭 Most common genre", f"{tg} · {tn:,} movies"),
    ]


# ----------------------------------------------------------------- state
st.session_state.setdefault("started", False)
st.session_state.setdefault("name", "")
movies, ratings, meta = get_data()
titles = sorted(meta.index)
user_ids = sorted(ratings["userId"].unique())
popular_titles = list(meta[meta["n_ratings"] >= 200].index)


def start():
    st.session_state["started"] = True
    st.session_state["just_started"] = True


def go_home():
    st.session_state["started"] = False


def surprise():
    st.session_state["picks"] = random.sample(popular_titles, 3)


# ----------------------------------------------------------------- landing page
if not st.session_state["started"]:
    name = st.session_state["name"].strip()
    hello = f"Hey {name}, welcome!" if name else "Hey, welcome!"
    rng = random.Random(7)
    sparks = "".join(
        f'<span class="cm-spark" style="left:{rng.randint(3, 96)}%;font-size:{rng.randint(14, 30)}px;'
        f'animation-delay:{rng.uniform(0, 8):.1f}s;animation-duration:{rng.uniform(7, 12):.1f}s">✦</span>'
        for _ in range(14))
    st.markdown(
        '<div class="cm-hero">'
        f'{img(globe_svg(), "cm-hero-globe")}<div class="cm-burst"></div>{sparks}'
        '<div class="cm-layer cm-small">Find your next favourite film</div>'
        '<div class="cm-layer cm-mascot-row">'
        f'<div class="cm-mascot">{MASCOT}</div>'
        f'<div class="cm-bubble">{html.escape(hello)} <span class="cm-wave">👋</span></div></div>'
        '<div class="cm-layer cm-hero-title">CINEMATCH</div>'
        '<div class="cm-layer cm-hero-sub">Movie Guide</div>'
        '<div class="cm-layer cm-tagline">'
        '<span style="animation-delay:0s">Tell us what you love and we find your next film 🎬</span>'
        '<span style="animation-delay:3s">Hybrid AI: movie content + community taste ⚡</span>'
        '<span style="animation-delay:6s">Pick 3 movies. Get 10 perfect matches ✨</span></div>'
        '<div class="cm-by">Presented by Team CineMatch</div></div>', unsafe_allow_html=True)
    st.write("")
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.text_input("What should we call you? (optional)", key="name",
                      placeholder="Type your name")
        st.button("🍿  Start exploring", type="primary", on_click=start)
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric("Members", f"{len(user_ids):,}")
    c2.metric("Movies", f"{len(movies):,}")
    c3.metric("Ratings", f"{len(ratings):,}")
    st.write("")
    f1, f2, f3 = st.columns(3)
    for col, (head, body) in zip((f1, f2, f3), fun_facts()):
        col.markdown(f'<div class="cm-frame"><h4>{head}</h4>{html.escape(body)}</div>',
                     unsafe_allow_html=True)
    st.write("")
    header("CineMatch", "How it works", "Three simple steps")
    steps = [("Pick", "Choose up to three movies you love, or hit Surprise me."),
             ("Match", "The hybrid model blends movie content with what similar viewers enjoyed."),
             ("Watch", "Get ranked picks with match scores, ratings and reasons.")]
    st.markdown("".join(
        f'<div class="cm-stepx" style="margin-left:{i * 70}px"><span class="cm-num">{i + 1}</span>'
        f'<div><b>{t}</b><br>{d}</div></div>' for i, (t, d) in enumerate(steps)),
        unsafe_allow_html=True)
    st.markdown(img(STAR_SVG, "cm-star", "animation-delay:0s") +
                img(STAR_SVG, "cm-star", "animation-delay:.5s") +
                img(STAR_SVG, "cm-star", "animation-delay:1s"), unsafe_allow_html=True)
    footer()
    st.stop()

if st.session_state.pop("just_started", False):
    st.balloons()

# ----------------------------------------------------------------- sidebar
all_genres = sorted({g for gl in meta["genre_list"] for g in gl
                     if g and g != "(no genres listed)"})
with st.sidebar:
    st.button("🏠 Back to welcome page", on_click=go_home)
    st.header("🎛 Your settings")
    new_user = st.checkbox("I'm a new viewer (no ratings yet)", value=True)
    if new_user:
        user_id = None
    else:
        user_id = int(st.number_input("Member ID", min_value=int(min(user_ids)),
                                      max_value=int(max(user_ids)),
                                      value=int(min(user_ids)), step=1))
    n = st.slider("How many recommendations?", 5, 20, 10)
    st.subheader("Filters")
    min_ratings = st.slider("Hide movies with fewer ratings than", 0, 200, 10, 5)
    genre_filter = st.multiselect("Only these genres (optional)", all_genres)
    with st.expander("Advanced"):
        alpha = st.slider("Hybrid weight on collaborative (alpha)", 0.0, 1.0, 0.8, 0.05,
                          help="1 = only collaborative, 0 = only content-based. "
                               "0.8 was best in our tuning.")
        api_key = st.text_input("TMDB API key (optional, for posters)",
                                value=os.environ.get("TMDB_API_KEY", ""), type="password")

warm_up(int(user_ids[0]))
pool = min(max(n * 4, 40), 100)
who = st.session_state["name"].strip()

tab1, tab2, tab3, tab4 = st.tabs(
    ["✨ Quick Match", "🎞 More like this", "👤 Member profile", "🔬 Under the hood"])

# ----- Quick Match
with tab1:
    header("CineMatch", "Quick Match",
           f"Hey {who}! What are we watching tonight?" if who
           else "What are we watching tonight?")
    cc1, cc2 = st.columns([4, 1])
    with cc1:
        picks = st.multiselect("Pick up to 3 movies you love (type to search)", titles,
                               max_selections=3, key="picks")
    with cc2:
        st.write("")
        st.write("")
        st.button("🎲 Surprise me", on_click=surprise)
    if hasattr(st, "pills"):
        mood = st.pills("What's your mood?", list(MOODS), default="Any mood",
                        key="mood") or "Any mood"
    else:
        mood = st.radio("What's your mood?", list(MOODS), horizontal=True, key="mood")
    mood_genres = MOODS[mood]
    if picks:
        ref = {g for p in picks for g in meta.loc[p]["genre_list"]}
        st.markdown(f'<div class="cm-italic">Because you like '
                    f'{html.escape(", ".join(clean_title(p)[0] for p in picks))}</div>',
                    unsafe_allow_html=True)
        pairs = multi_pick_recs(picks, user_id, pool, alpha)
        df = build_table(pairs, n, ref, exclude=picks, genres=mood_genres)
        show_cards(df, spotlight=True)
    else:
        st.markdown('<div class="cm-italic">🔥 Popular right now · pick favourites above '
                    'for personal recommendations</div>', unsafe_allow_html=True)
        pop = meta[(meta["n_ratings"] >= max(min_ratings, 100)) & (meta["avg"] >= 3.9)]
        pop = pop.sort_values("n_ratings", ascending=False)
        pairs = [(t, r["avg"] / 5.0) for t, r in pop.iterrows()]
        df = build_table(pairs, n, genres=mood_genres)
        show_cards(df, label="rated", spotlight=True, spot_label="🔥 TRENDING NOW")
    if not df.empty:
        st.download_button("⬇ Save my picks (CSV)",
                           df[["Title", "Avg rating", "Ratings", "Genres"]]
                           .to_csv(index=False).encode("utf-8"),
                           "my_cinematch_picks.csv", "text/csv")
    footer()

# ----- More like this
with tab2:
    header("CineMatch", "More like this", "Find movies like one you know")
    single = st.selectbox("Search a movie (type to filter)", titles, key="single")
    sel = meta.loc[single]
    sel_genres = sel["genre_list"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Average rating", f"{sel['avg']:.2f} / 5")
    m2.metric("Number of ratings", f"{int(sel['n_ratings']):,}")
    m3.metric("Genres", " • ".join(sel_genres[:3]))
    mode = st.radio("Show", ["Similar movies (content-based)", "Hybrid picks"],
                    horizontal=True)
    if mode.startswith("Similar"):
        pairs = safe(similar_recs, single, pool)
        show_cards(build_table(pairs, n, sel_genres, tiebreak=True, exclude=[single]))
    else:
        pairs = safe(hybrid_recs, user_id, single, pool, alpha)
        show_cards(build_table(pairs, n, sel_genres, exclude=[single]))
    footer()

# ----- Member profile
with tab3:
    header("CineMatch", "Member profile", "Picks for an existing member")
    if user_id is None:
        st.info("Untick 'I'm a new viewer' in the sidebar and choose a Member ID.")
    else:
        gen, top, n_rated = user_profile(user_id)
        fav = [g for g, _ in gen.most_common(3)]
        st.write(f"**Member {user_id}** rated {n_rated} movies. "
                 f"Favourite genres: {', '.join(fav) if fav else 'n/a'}.")
        left, right = st.columns(2)
        with left:
            st.caption("Top-rated movies by this member")
            st.dataframe(top.rename(columns={"title": "Title", "rating": "Rating"}))
        with right:
            if gen:
                st.caption("Liked movies (rating 4+) by genre")
                chart(st.bar_chart, pd.Series(dict(gen.most_common(8))), "#7da2ff")
        st.markdown('<div class="cm-italic">Recommended for this member '
                    '(collaborative filtering)</div>', unsafe_allow_html=True)
        show_cards(build_table(safe(user_recs, user_id, pool), n, fav),
                   normalize=True, spotlight=True)
    footer()

# ----- Under the hood
with tab4:
    header("CineMatch", "Under the hood", "How it works", art=img(net_svg(), "cm-netimg"))
    st.markdown(
        '<div class="cm-tl">'
        '<div class="cm-tl-item"><div class="cm-tl-dot">›</div><div class="cm-tl-box">'
        '<b>🎞 Content-based</b><br>Finds movies with similar genres and features to the '
        'ones you pick.</div></div>'
        '<div class="cm-tl-item"><div class="cm-tl-dot">›</div><div class="cm-tl-box">'
        '<b>🤝 Collaborative</b><br>Learns from the ratings of many viewers '
        '(matrix factorisation).</div></div>'
        '<div class="cm-tl-item"><div class="cm-tl-dot">›</div><div class="cm-tl-box">'
        '<b>⚡ Hybrid</b><br>Mixes both with a weight alpha, so it also works for new '
        'viewers.</div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="cm-italic">Same movie, three models</div>', unsafe_allow_html=True)
    st.caption(f"Movie: {single}  |  Member: "
               f"{user_id if user_id is not None else 'new viewer'}")
    sg = meta.loc[single]["genre_list"]
    lists = {
        "Content-based": build_table(safe(similar_recs, single, pool), n, sg,
                                     tiebreak=True, exclude=[single]),
        "Collaborative": (build_table(safe(user_recs, user_id, pool), n)
                          if user_id is not None else pd.DataFrame()),
        "Hybrid": build_table(safe(hybrid_recs, user_id, single, pool, alpha), n, sg,
                              exclude=[single]),
    }
    cols = st.columns(3)
    for col, (name, d) in zip(cols, lists.items()):
        col.markdown(f"**{name}**")
        if d.empty:
            col.caption("No results (new viewer or filters too strict).")
        else:
            col.dataframe(d[["Title", "Avg rating"]])
    seen = Counter(t for d in lists.values() if not d.empty for t in d["Title"])
    both = [t for t, c in seen.most_common() if c >= 2]
    st.markdown("**Picked by two or more models:** " +
                (", ".join(both) if both else "no overlap between the lists."))

    st.markdown('<div class="cm-italic">Evaluation results</div>', unsafe_allow_html=True)
    st.caption("80/20 split per user, ratings of 4+ count as 'liked', K = 10. "
               "Lower RMSE/MAE is better, higher Precision/Recall is better.")
    st.dataframe(FINAL_RESULTS)
    chart(st.bar_chart, FINAL_RESULTS[["Precision@10", "Recall@10"]], ["#4dabf7", "#b197fc"])
    st.markdown('<div class="cm-italic">Tuning the hybrid weight (alpha)</div>',
                unsafe_allow_html=True)
    l, r = st.columns(2)
    with l:
        st.caption("Ranking quality vs alpha")
        chart(st.line_chart, ALPHA_TUNING[["Precision@10", "Recall@10"]], ["#4dabf7", "#b197fc"])
    with r:
        st.caption("Rating error vs alpha")
        chart(st.line_chart, ALPHA_TUNING[["RMsSE", "MAE"]], ["#4dabf7", "#b197fc"])
    st.info("alpha = 0.8 gives the lowest MAE and almost the best ranking, "
            "so it is the default.")
    footer()