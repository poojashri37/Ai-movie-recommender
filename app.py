"""
app.py  -  CineMatch (Streamlit front end)

Run from the project root:
    python -m streamlit run app.py

Only imports and calls the team's functions:
    preprocess.load_data()                           -> movies, ratings
    content_based.get_similar_movies(title, n)       -> [(title, score), ...]
    hybrid.get_hybrid_recs(user_id, title, n, alpha) -> [(title, score), ...]
    hybrid.get_user_recs(user_id, n)                 -> [(title, score), ...]
Posters from TMDB are optional (paste a free API key in the sidebar).
"""
import html
import os
import random
import re
from collections import Counter

import pandas as pd
import requests
import streamlit as st

from content_based import get_similar_movies
from hybrid import get_hybrid_recs, get_user_recs
from preprocess import load_data

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

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

MASCOT = "🧸"  # change to 🤖 🐼 🦊 🍿 or any emoji you like

GENRE_STYLE = {
    "Action": ("💥", "#ff6b6b", "#e03131"), "Adventure": ("🧭", "#ffa94d", "#f76707"),
    "Animation": ("🎨", "#fab005", "#f08c00"), "Children": ("🧸", "#ff8787", "#fa5252"),
    "Comedy": ("😂", "#fcc419", "#f59f00"), "Crime": ("🕵️", "#868e96", "#495057"),
    "Documentary": ("🎥", "#38d9a9", "#12b886"), "Drama": ("🎭", "#9775fa", "#7048e8"),
    "Fantasy": ("🧙", "#da77f2", "#ae3ec9"), "Film-Noir": ("🌃", "#495057", "#212529"),
    "Horror": ("👻", "#5c3d99", "#2b1b4d"), "IMAX": ("🏟️", "#4dabf7", "#1c7ed6"),
    "Musical": ("🎶", "#f783ac", "#d6336c"), "Mystery": ("🔍", "#748ffc", "#4263eb"),
    "Romance": ("💕", "#faa2c1", "#e64980"), "Sci-Fi": ("🚀", "#38d9a9", "#0ca678"),
    "Thriller": ("😱", "#ff8787", "#c92a2a"), "War": ("🪖", "#69db7c", "#2f9e44"),
    "Western": ("🤠", "#e9b872", "#a0522d"),
}

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
@keyframes cmfade {from {opacity: 0; transform: translateY(14px);} to {opacity: 1; transform: none;}}
.cm-hero {background: linear-gradient(120deg,#7048e8,#e64980 55%,#fd7e14); color: #fff;
  border-radius: 24px; padding: 54px 40px; text-align: center; animation: cmfade .9s ease;
  box-shadow: 0 12px 40px rgba(112,72,232,.35);}
.cm-hero h1 {font-size: 3.2rem; margin: 6px 0 4px 0; color: #fff;}
@keyframes cmslide {0% {transform: translateX(-380px) rotate(-30deg); opacity: 0;}
  60% {transform: translateX(16px) rotate(8deg); opacity: 1;} 100% {transform: none; opacity: 1;}}
@keyframes cmpop {0% {transform: scale(0); opacity: 0;} 70% {transform: scale(1.12);}
  100% {transform: scale(1); opacity: 1;}}
@keyframes cmwave {0%,100% {transform: rotate(0);} 25% {transform: rotate(20deg);}
  75% {transform: rotate(-14deg);}}
@keyframes cmfloat {0%,100% {transform: translateY(0);} 50% {transform: translateY(-10px);}}
.cm-mascot-row {display: flex; align-items: center; justify-content: center; gap: 16px;
  margin-bottom: 6px; flex-wrap: wrap;}
.cm-mascot {font-size: 6rem; line-height: 1;
  animation: cmslide 1.4s ease-out both, cmfloat 2.4s ease-in-out 1.5s infinite;}
.cm-bubble {position: relative; background: #fff; color: #5f3dc4; font-weight: 800;
  font-size: 1.6rem; padding: 12px 24px; border-radius: 24px;
  box-shadow: 0 6px 18px rgba(0,0,0,.25); animation: cmpop .6s ease-out 1.2s both;}
.cm-bubble::before {content: ""; position: absolute; left: -13px; top: 50%; margin-top: -11px;
  border-style: solid; border-width: 11px 15px 11px 0;
  border-color: transparent #fff transparent transparent;}
.cm-wave {display: inline-block; transform-origin: 70% 70%;
  animation: cmwave 1.2s ease-in-out 1.6s infinite;}
.cm-sub {font-size: 1.15rem; opacity: .95; margin-bottom: 18px;}
.cm-pill {display: inline-block; background: rgba(255,255,255,.2); border-radius: 999px;
  padding: 6px 16px; margin: 4px; font-size: .95rem;}
.cm-band {background: linear-gradient(120deg,#7048e8,#e64980 60%,#fd7e14); color: #fff;
  border-radius: 16px; padding: 14px 24px; margin-bottom: 8px;}
.cm-band b {font-size: 1.4rem;}
.cm-step {border-radius: 18px; padding: 18px; background: rgba(127,127,127,.12); height: 100%;
  animation: cmfade 1.1s ease;}
.cm-grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 18px;
  margin-top: 8px;}
.cm-card {border-radius: 18px; overflow: hidden; background: rgba(127,127,127,.13);
  box-shadow: 0 4px 16px rgba(0,0,0,.18); transition: transform .2s, box-shadow .2s;
  animation: cmfade .7s ease;}
.cm-card:hover {transform: translateY(-6px); box-shadow: 0 12px 28px rgba(0,0,0,.35);}
.cm-tile {height: 170px; display: flex; flex-direction: column; align-items: center;
  justify-content: center; color: #fff; text-align: center; padding: 10px;}
.cm-emoji {font-size: 3rem; line-height: 1;}
.cm-tile-title {font-weight: 700; font-size: .95rem; margin-top: 8px; text-shadow: 0 1px 4px rgba(0,0,0,.4);}
.cm-poster {width: 100%; height: 280px; object-fit: cover; display: block;}
.cm-body {padding: 10px 12px 14px 12px;}
.cm-title {font-weight: 700; font-size: .95rem; margin-bottom: 2px;}
.cm-meta {font-size: .8rem; opacity: .8; margin-bottom: 6px;}
.cm-chip {display: inline-block; font-size: .7rem; padding: 2px 9px; border-radius: 999px;
  background: rgba(127,127,127,.28); margin: 2px 4px 2px 0;}
.cm-bar {height: 7px; border-radius: 5px; background: rgba(127,127,127,.28); margin-top: 8px;}
.cm-fill {height: 100%; border-radius: 5px; background: linear-gradient(90deg,#f06595,#fcc419);}
.cm-why {font-size: .72rem; opacity: .75; margin-top: 5px;}
</style>
"""
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


def build_table(pairs, n, ref_genres=(), tiebreak=False, exclude=()):
    """Add rating info + 'why' genres, apply sidebar filters, return top n rows."""
    rows = []
    for title, score in pairs:
        if title not in meta.index or title in exclude:
            continue
        info = meta.loc[title]
        glist = info["genre_list"]
        if info["n_ratings"] < min_ratings:
            continue
        if genre_filter and not set(glist) & set(genre_filter):
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


def card_html(r, label, top_score):
    title = r["Title"]
    name, _ = clean_title(title)
    glist = [g for g in r["Genres"].split(", ") if g]
    emoji, c1, c2 = next((GENRE_STYLE[g] for g in glist if g in GENRE_STYLE),
                         ("🎬", "#7950f2", "#4c6ef5"))
    poster = poster_url(title, api_key)
    if poster:
        top = f'<img class="cm-poster" src="{html.escape(poster)}">'
    else:
        top = (f'<div class="cm-tile" style="background:linear-gradient(135deg,{c1},{c2})">'
               f'<div class="cm-emoji">{emoji}</div>'
               f'<div class="cm-tile-title">{html.escape(name)}</div></div>')
    pct = int(round(max(0.0, min(1.0, r["Score"] / top_score if top_score else 0)) * 100))
    chips = "".join(f'<span class="cm-chip">{html.escape(g)}</span>' for g in glist[:3])
    why = (f'<div class="cm-why">Because: {html.escape(r["Why"])}</div>'
           if r["Why"] not in ("-", "") else "")
    return (f'<div class="cm-card">{top}<div class="cm-body">'
            f'<div class="cm-title">{html.escape(title)}</div>'
            f'<div class="cm-meta">⭐ {r["Avg rating"]} · {r["Ratings"]:,} ratings</div>'
            f'{chips}<div class="cm-bar"><div class="cm-fill" style="width:{pct}%"></div></div>'
            f'<div class="cm-meta" style="margin-top:4px">{pct}% {label}</div>'
            f'{why}</div></div>')


def show_cards(df, label="match", normalize=False):
    if df.empty:
        st.info("No movies match your filters. Try lowering the minimum number of "
                "ratings or clearing the genre filter.")
        return
    top = float(df["Score"].max()) if normalize else 1.0
    cards = "".join(card_html(r, label, top) for _, r in df.iterrows())
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
    st.markdown(
        '<div class="cm-hero"><div class="cm-mascot-row">'
        f'<div class="cm-mascot">{MASCOT}</div>'
        f'<div class="cm-bubble">{html.escape(hello)} <span class="cm-wave">👋</span></div>'
        '</div><h1>🎬 CineMatch</h1>'
        '<div class="cm-sub">Your AI movie guide. Tell us what you love and we '
        'will find your next favourite film.</div>'
        '<span class="cm-pill">🎞 Content-based</span>'
        '<span class="cm-pill">🤝 Collaborative</span>'
        '<span class="cm-pill">⚡ Hybrid</span></div>', unsafe_allow_html=True)
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
    s1, s2, s3 = st.columns(3)
    s1.markdown('<div class="cm-step"><h3>1️⃣ Pick</h3>Choose up to three movies you '
                'love, or hit Surprise me.</div>', unsafe_allow_html=True)
    s2.markdown('<div class="cm-step"><h3>2️⃣ Match</h3>Our hybrid model blends movie '
                'content with what similar viewers enjoyed.</div>', unsafe_allow_html=True)
    s3.markdown('<div class="cm-step"><h3>3️⃣ Watch</h3>Get ranked picks with match '
                'scores, ratings and reasons.</div>', unsafe_allow_html=True)
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
st.markdown(
    f'<div class="cm-band"><b>🎬 CineMatch</b> &nbsp; Hey'
    f'{" " + html.escape(who) if who else ""}! What are we watching tonight?</div>',
    unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["✨ Quick Match", "🎞 More like this", "👤 Member profile", "🔬 Under the hood"])

# ----- Quick Match
with tab1:
    st.markdown("### ✨ Tell us what you love")
    cc1, cc2 = st.columns([4, 1])
    with cc1:
        picks = st.multiselect("Pick up to 3 movies you love (type to search)", titles,
                               max_selections=3, key="picks")
    with cc2:
        st.write("")
        st.write("")
        st.button("🎲 Surprise me", on_click=surprise)
    if picks:
        ref = {g for p in picks for g in meta.loc[p]["genre_list"]}
        st.markdown(f"### 🍿 Because you like {', '.join(clean_title(p)[0] for p in picks)}")
        pairs = multi_pick_recs(picks, user_id, pool, alpha)
        show_cards(build_table(pairs, n, ref, exclude=picks))
    else:
        st.markdown("### 🔥 Popular right now")
        st.caption("Pick a few favourites above to get personal recommendations.")
        pop = meta[(meta["n_ratings"] >= max(min_ratings, 100)) & (meta["avg"] >= 3.9)]
        pop = pop.sort_values("n_ratings", ascending=False).head(n)
        pairs = [(t, r["avg"] / 5.0) for t, r in pop.iterrows()]
        show_cards(build_table(pairs, n), label="rated")

# ----- More like this
with tab2:
    st.markdown("### 🎞 Find movies like one you know")
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

# ----- Member profile
with tab3:
    st.markdown("### 👤 Picks for an existing member")
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
                st.bar_chart(pd.Series(dict(gen.most_common(8))))
        st.markdown("#### 🎯 Recommended for this member (collaborative filtering)")
        show_cards(build_table(safe(user_recs, user_id, pool), n, fav),
                   normalize=True)

# ----- Under the hood
with tab4:
    st.markdown("### 🔬 How it works")
    h1, h2, h3 = st.columns(3)
    h1.markdown('<div class="cm-step"><h4>🎞 Content-based</h4>Finds movies with '
                'similar genres and features to the ones you pick.</div>',
                unsafe_allow_html=True)
    h2.markdown('<div class="cm-step"><h4>🤝 Collaborative</h4>Learns from the '
                'ratings of many viewers (matrix factorisation).</div>',
                unsafe_allow_html=True)
    h3.markdown('<div class="cm-step"><h4>⚡ Hybrid</h4>Mixes both with a weight '
                'alpha, so it also works for new viewers.</div>', unsafe_allow_html=True)

    st.markdown("### ⚖ Same movie, three models")
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

    st.markdown("### 📊 Evaluation results")
    st.caption("80/20 split per user, ratings of 4+ count as 'liked', K = 10. "
               "Lower RMSE/MAE is better, higher Precision/Recall is better.")
    st.dataframe(FINAL_RESULTS)
    st.bar_chart(FINAL_RESULTS[["Precision@10", "Recall@10"]])
    st.markdown("**Tuning the hybrid weight (alpha)**")
    l, r = st.columns(2)
    with l:
        st.caption("Ranking quality vs alpha")
        st.line_chart(ALPHA_TUNING[["Precision@10", "Recall@10"]])
    with r:
        st.caption("Rating error vs alpha")
        st.line_chart(ALPHA_TUNING[["RMSE", "MAE"]])
    st.info("alpha = 0.8 gives the lowest MAE and almost the best ranking, "
            "so it is the default.")