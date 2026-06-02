import json
import os
import base64
from datetime import datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import finnhub
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from openai import OpenAI
from plotly.subplots import make_subplots
from xgboost import XGBRegressor

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:
    YFRateLimitError = Exception


# ---------- App Setup: page metadata, API clients, shared constants ----------

st.set_page_config(
    page_title="Lupa AI Stock Terminal",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

FINNHUB_API_KEY = st.secrets["FINNHUB_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

BIG_TECHS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOGL", "AMD"]
PERIOD_OPTIONS = ["3mo", "6mo", "1y", "2y", "5y"]
FORECAST_STATE_KEY = "forecast_result"
US_MARKET_TZ = ZoneInfo("America/New_York")
FINBERT_MIN_AVAILABLE_MB = 900
MARKET_CLOSE_STABILIZATION_HOURS = 2
MAX_DYNAMIC_BLEND_AGE_DAYS = 30
PREDICTION_LOG_PATH = os.path.join(os.path.dirname(__file__), "llm_prediction_log.csv")
PREDICTION_LOG_COLUMNS = [
    "ticker",
    "created_at",
    "target_date",
    "reference_close_date",
    "reference_close_price",
    "xgb_pred_price",
    "llm_pred_price",
    "llm_conf",
    "ensemble_price",
    "weight_xgb_used",
    "weight_llm_used",
    "actual_close",
    "xgb_abs_error",
    "llm_abs_error",
    "ensemble_abs_error",
    "status",
]


# ---------- Theme: light/dark colors and global CSS ----------


def empty_prediction_log_df():
    return pd.DataFrame(columns=PREDICTION_LOG_COLUMNS)

def get_theme(is_dark_mode):
    if is_dark_mode:
        return {
            "bg_style": (
                "radial-gradient(circle at 50% 30%, rgba(255,255,255,0.05), transparent 60%), "
                "radial-gradient(circle at center, #1e293b 0%, #020617 100%)"
            ),
            "sidebar_bg": "#020617",
            "text_color": "#ffffff",
            "muted_text_color": "#cbd5e1",
            "metric_bg": "rgba(255,255,255,0.05)",
            "card_bg": "rgba(255,255,255,0.06)",
            "card_border": "1px solid rgba(255,255,255,0.10)",
            "input_bg": "#0f172a",
            "input_border": "1px solid rgba(255,255,255,0.12)",
            "dropdown_bg": "#0f172a",
            "dropdown_hover_bg": "#1e293b",
            "dropdown_selected_bg": "#273449",
            "plotly_template": "plotly_dark",
            "grid_color": "rgba(255,255,255,0.10)",
        }

    return {
        "bg_style": (
            "radial-gradient(circle at 50% 30%, rgba(0,0,0,0.12), transparent 55%), "
            "radial-gradient(circle at center, #ffffff 0%, #cbd5e1 100%)"
        ),
        "sidebar_bg": "#ffffff",
        "text_color": "#000000",
        "muted_text_color": "#334155",
        "metric_bg": "#ffffff",
        "card_bg": "rgba(255,255,255,0.92)",
        "card_border": "1px solid rgba(15,23,42,0.08)",
        "input_bg": "#ffffff",
        "input_border": "1px solid rgba(15,23,42,0.16)",
        "dropdown_bg": "#ffffff",
        "dropdown_hover_bg": "#f1f5f9",
        "dropdown_selected_bg": "#e2e8f0",
        "plotly_template": "plotly_white",
        "grid_color": "rgba(0,0,0,0.10)",
    }


def apply_theme(theme):
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: {theme["bg_style"]} !important;
        }}

        [data-testid="stSidebar"] {{
            background-color: {theme["sidebar_bg"]};
        }}

        .block-container {{
            padding-top: 2rem;
        }}

        [data-testid="stMetric"] {{
            background: {theme["metric_bg"]};
            padding: 15px;
            border-radius: 10px;
        }}

        h1, h2, h3, h4, h5, p, label, span, div {{
            color: {theme["text_color"]};
        }}

        [data-testid="stSidebar"] *,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {{
            color: {theme["text_color"]} !important;
        }}

        [data-testid="stMetricValue"] div,
        button[data-baseweb="tab"] div {{
            color: {theme["text_color"]} !important;
        }}

        .stTextInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        .stSelectbox input {{
            background: {theme["input_bg"]} !important;
            border: {theme["input_border"]} !important;
            color: {theme["text_color"]} !important;
            -webkit-text-fill-color: {theme["text_color"]} !important;
        }}

        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stSelectbox input {{
            background: {theme["input_bg"]} !important;
            color: {theme["text_color"]} !important;
            -webkit-text-fill-color: {theme["text_color"]} !important;
        }}

        div[data-baseweb="popover"] ul {{
            background: {theme["dropdown_bg"]} !important;
            border: {theme["input_border"]} !important;
        }}

        div[data-baseweb="popover"] li {{
            background: {theme["dropdown_bg"]} !important;
            color: {theme["text_color"]} !important;
        }}

        div[data-baseweb="popover"] li:hover {{
            background: {theme["dropdown_hover_bg"]} !important;
        }}

        div[data-baseweb="popover"] li[aria-selected="true"] {{
            background: {theme["dropdown_selected_bg"]} !important;
            color: {theme["text_color"]} !important;
        }}

        .stButton > button p {{
            color: white !important;
            font-weight: 700 !important;
        }}

        .themed-card {{
            background: {theme["card_bg"]};
            border: {theme["card_border"]};
            border-radius: 15px;
        }}

        .signal-card {{
            background: {theme["card_bg"]};
            border: {theme["card_border"]};
            border-radius: 15px;
            text-align: center;
        }}

        .signal-card-title {{
            margin: 0;
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.2;
        }}

        .signal-card-meta {{
            margin-top: 10px;
            font-size: 1rem;
            color: {theme["text_color"]};
        }}

        .signal-buy {{
            color: #22c55e !important;
        }}

        .signal-sell {{
            color: #ef4444 !important;
        }}

        .app-header {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 18px;
            margin: 0 0 24px 0;
        }}

        .app-header-logo {{
            width: 86px;
            height: 86px;
            object-fit: contain;
            display: block;
        }}

        .app-header-title {{
            margin: 0;
            font-size: 3.2rem;
            font-weight: 800;
            line-height: 1;
            color: {theme["text_color"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------- Session State: initialize and reset derived forecast output ----------

def clear_forecast_state():
    st.session_state.pop(FORECAST_STATE_KEY, None)


def initialize_session_state():
    st.session_state.setdefault("ticker", "AAPL")
    st.session_state.setdefault("bigtech", "AAPL")


def on_ticker_changed():
    ticker = st.session_state.ticker.upper()
    st.session_state.ticker = ticker
    if ticker in BIG_TECHS:
        st.session_state.bigtech = ticker
    else:
        st.session_state.bigtech = None
    clear_forecast_state()


def on_bigtech_changed():
    if st.session_state.bigtech:
        st.session_state.ticker = st.session_state.bigtech
    clear_forecast_state()


def get_supabase_config():
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY")
    if not url or not key:
        return None
    return {"url": url.rstrip("/"), "key": key}


def supabase_request(method, path, payload=None):
    config = get_supabase_config()
    if config is None:
        return None

    request = Request(
        f'{config["url"]}{path}',
        method=method,
        headers={
            "apikey": config["key"],
            "Authorization": f'Bearer {config["key"]}',
            "Content-Type": "application/json",
        },
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
    )
    with urlopen(request, timeout=15) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else None


def load_prediction_log():
    supabase_config = get_supabase_config()
    if supabase_config is not None:
        records = supabase_request(
            "GET",
            "/rest/v1/prediction_log?select=*&order=created_at.desc",
        )
        if not records:
            return empty_prediction_log_df()
        return pd.DataFrame(records)

    if not os.path.exists(PREDICTION_LOG_PATH):
        return empty_prediction_log_df()
    return pd.read_csv(PREDICTION_LOG_PATH)


def prediction_record_exists(ticker, target_date, reference_close_date):
    supabase_config = get_supabase_config()
    if supabase_config is not None:
        existing = supabase_request(
            "GET",
            "/rest/v1/prediction_log?select=id"
            f"&ticker=eq.{quote(ticker)}"
            f"&target_date=eq.{quote(target_date)}"
            f"&reference_close_date=eq.{quote(reference_close_date)}"
            "&limit=1",
        )
        return bool(existing)

    log_df = load_prediction_log()
    if log_df.empty:
        return False

    existing = log_df[
        (log_df["ticker"] == ticker)
        & (log_df["target_date"] == target_date)
        & (log_df["reference_close_date"] == reference_close_date)
    ]
    return not existing.empty


def append_prediction_log_record(ticker, reference_close_price, forecast_result):
    target_date = forecast_result["predicted_date"]
    reference_close_date = forecast_result["reference_close_date"]

    if prediction_record_exists(ticker, target_date, reference_close_date):
        return "duplicate", None

    row_dict = {
        "ticker": ticker,
        "created_at": datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y-%m-%d %H:%M:%S"),
        "target_date": target_date,
        "reference_close_date": reference_close_date,
        "reference_close_price": float(reference_close_price),
        "xgb_pred_price": float(forecast_result["pred_price"]),
        "llm_pred_price": float(forecast_result["llm_price"]),
        "llm_conf": float(forecast_result["llm_conf"]),
        "ensemble_price": float(forecast_result["ensemble_price"]),
        "weight_xgb_used": float(forecast_result["weight_xgb"]),
        "weight_llm_used": float(forecast_result["weight_llm"]),
        "actual_close": None,
        "xgb_abs_error": None,
        "llm_abs_error": None,
        "ensemble_abs_error": None,
        "status": "pending",
    }

    supabase_config = get_supabase_config()
    if supabase_config is not None:
        try:
            supabase_request("POST", "/rest/v1/prediction_log", [row_dict])
            return "supabase", None
        except HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8")
            except Exception:
                error_body = exc.reason
            return "csv", f"Supabase insert failed ({exc.code}): {error_body}"
        except (URLError, TimeoutError, ValueError) as exc:
            return "csv", f"Supabase insert failed: {exc}"

    row = pd.DataFrame([row_dict], columns=PREDICTION_LOG_COLUMNS)
    if os.path.exists(PREDICTION_LOG_PATH):
        row.to_csv(PREDICTION_LOG_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(PREDICTION_LOG_PATH, index=False)
    return "csv", None


def normalize_download_history(df, ticker):
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=-1)
        except Exception:
            df.columns = df.columns.get_level_values(0)

    return df


def fetch_actual_close_map_for_ticker(ticker, target_dates):
    if not target_dates:
        return {}

    parsed_dates = [
        datetime.strptime(str(target_date), "%Y-%m-%d").date()
        for target_date in target_dates
    ]
    start = (min(parsed_dates) - timedelta(days=2)).strftime("%Y-%m-%d")
    end = (max(parsed_dates) + timedelta(days=3)).strftime("%Y-%m-%d")

    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    except YFRateLimitError:
        return {}
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        try:
            df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        except YFRateLimitError:
            return {}
        except Exception:
            return {}

    df = normalize_download_history(df, ticker)
    if df.empty:
        return {}

    close_map = {}
    for ts, row in df.iterrows():
        close_map[pd.Timestamp(ts).date().isoformat()] = float(row["Close"])

    return close_map


def fetch_actual_close_for_target_date(ticker, target_date_str):
    return fetch_actual_close_map_for_ticker(ticker, [target_date_str]).get(target_date_str)


def update_actual_closes_in_log():
    log_df = load_prediction_log()
    if log_df.empty:
        return {"updated": 0, "skipped": 0}

    supabase_config = get_supabase_config()
    pending_df = log_df[log_df["status"].fillna("pending") != "completed"].copy()
    updated = 0
    skipped = 0

    for ticker, ticker_rows in pending_df.groupby("ticker"):
        target_dates = [str(target_date) for target_date in ticker_rows["target_date"].tolist()]
        close_map = fetch_actual_close_map_for_ticker(ticker, target_dates)

        for _, row in ticker_rows.iterrows():
            record_id = row.get("id")
            target_date = str(row["target_date"])
            actual_close = close_map.get(target_date)
            if actual_close is None:
                skipped += 1
                continue

            xgb_abs_error = abs(float(row["xgb_pred_price"]) - actual_close)
            llm_abs_error = abs(float(row["llm_pred_price"]) - actual_close)
            ensemble_abs_error = abs(float(row["ensemble_price"]) - actual_close)

            update_payload = {
                "actual_close": actual_close,
                "xgb_abs_error": xgb_abs_error,
                "llm_abs_error": llm_abs_error,
                "ensemble_abs_error": ensemble_abs_error,
                "status": "completed",
            }

            if supabase_config is not None and pd.notna(record_id):
                supabase_request(
                    "PATCH",
                    f"/rest/v1/prediction_log?id=eq.{int(record_id)}",
                    update_payload,
                )
            else:
                mask = (
                    (log_df["ticker"] == ticker)
                    & (log_df["target_date"] == target_date)
                    & (log_df["reference_close_date"] == row["reference_close_date"])
                )
                for key, value in update_payload.items():
                    log_df.loc[mask, key] = value
            updated += 1

    if supabase_config is None and updated > 0:
        log_df.to_csv(PREDICTION_LOG_PATH, index=False)

    return {"updated": updated, "skipped": skipped}


def get_dynamic_blend_weights(ticker, fallback_llm_weight, min_samples=5, window=10):
    fallback_llm_weight = min(max(float(fallback_llm_weight), 0.2), 0.8)
    fallback_xgb_weight = 1 - fallback_llm_weight
    default_result = {
        "weight_xgb": fallback_xgb_weight,
        "weight_llm": fallback_llm_weight,
        "source": "default",
        "sample_count": 0,
        "mae_xgb": None,
        "mae_llm": None,
    }

    log_df = load_prediction_log()
    if log_df.empty:
        return default_result

    ticker_df = log_df[log_df["ticker"] == ticker].copy()
    if ticker_df.empty:
        return default_result

    ticker_df = ticker_df[ticker_df["status"].fillna("pending") == "completed"].copy()
    if ticker_df.empty:
        return default_result

    ticker_df["target_date"] = pd.to_datetime(ticker_df["target_date"], errors="coerce")
    ticker_df["created_at"] = pd.to_datetime(ticker_df["created_at"], errors="coerce")
    ticker_df["xgb_abs_error"] = pd.to_numeric(ticker_df["xgb_abs_error"], errors="coerce")
    ticker_df["llm_abs_error"] = pd.to_numeric(ticker_df["llm_abs_error"], errors="coerce")
    cutoff_date = pd.Timestamp(datetime.now(ZoneInfo("Asia/Singapore")).date() - timedelta(days=MAX_DYNAMIC_BLEND_AGE_DAYS))
    ticker_df = ticker_df.dropna(subset=["target_date", "xgb_abs_error", "llm_abs_error"])
    ticker_df = ticker_df[ticker_df["target_date"] >= cutoff_date].sort_values("target_date")
    if ticker_df.empty:
        return default_result

    recent_df = ticker_df.tail(window)
    sample_count = len(recent_df)

    if sample_count < min_samples:
        return {
            "weight_xgb": fallback_xgb_weight,
            "weight_llm": fallback_llm_weight,
            "source": "default",
            "sample_count": sample_count,
            "mae_xgb": None,
            "mae_llm": None,
        }

    mae_xgb = float(recent_df["xgb_abs_error"].mean())
    mae_llm = float(recent_df["llm_abs_error"].mean())

    raw_xgb = 1 / (mae_xgb + 1e-6)
    raw_llm = 1 / (mae_llm + 1e-6)
    total = raw_xgb + raw_llm
    weight_xgb = raw_xgb / total
    weight_llm = raw_llm / total

    weight_xgb = max(0.2, min(0.8, weight_xgb))
    weight_llm = 1 - weight_xgb

    return {
        "weight_xgb": weight_xgb,
        "weight_llm": weight_llm,
        "source": "dynamic",
        "sample_count": sample_count,
        "mae_xgb": mae_xgb,
        "mae_llm": mae_llm,
    }


# ---------- Shared Helpers: small reusable calculations and formatters ----------

def get_signal_style(value, reference):
    if value > reference:
        return "Bullish", "#22c55e"
    return "Bearish", "#ef4444"


def nth_weekday_of_month(year, month, weekday, occurrence):
    first_day = datetime(year, month, 1).date()
    offset = (weekday - first_day.weekday()) % 7
    return first_day + timedelta(days=offset + (occurrence - 1) * 7)


def last_weekday_of_month(year, month, weekday):
    if month == 12:
        next_month = datetime(year + 1, 1, 1).date()
    else:
        next_month = datetime(year, month + 1, 1).date()
    current = next_month - timedelta(days=1)
    while current.weekday() != weekday:
        current -= timedelta(days=1)
    return current


def observed_holiday(holiday_date):
    if holiday_date.weekday() == 5:
        return holiday_date - timedelta(days=1)
    if holiday_date.weekday() == 6:
        return holiday_date + timedelta(days=1)
    return holiday_date


def calculate_easter_date(year):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime(year, month, day).date()


@st.cache_data(ttl=86400)
def get_us_market_holidays_for_year(year):
    holidays = {
        observed_holiday(datetime(year, 1, 1).date()),
        nth_weekday_of_month(year, 1, 0, 3),   # MLK Day
        nth_weekday_of_month(year, 2, 0, 3),   # Presidents' Day
        calculate_easter_date(year) - timedelta(days=2),  # Good Friday
        last_weekday_of_month(year, 5, 0),     # Memorial Day
        observed_holiday(datetime(year, 6, 19).date()),   # Juneteenth
        observed_holiday(datetime(year, 7, 4).date()),    # Independence Day
        nth_weekday_of_month(year, 9, 0, 1),   # Labor Day
        nth_weekday_of_month(year, 11, 3, 4),  # Thanksgiving
        observed_holiday(datetime(year, 12, 25).date()),  # Christmas
    }
    return {holiday.isoformat() for holiday in holidays}


def is_us_trading_day(check_date_str):
    check_date = datetime.strptime(check_date_str, "%Y-%m-%d").date()
    if check_date.weekday() >= 5:
        return False

    return check_date.isoformat() not in get_us_market_holidays_for_year(check_date.year)


def get_next_trading_day(base_date):
    next_day = base_date + timedelta(days=1)
    while not is_us_trading_day(next_day.strftime("%Y-%m-%d")):
        next_day += timedelta(days=1)
    return next_day


def get_next_market_open(reference_time):
    candidate_date = reference_time.date()

    if not is_us_trading_day(candidate_date.strftime("%Y-%m-%d")):
        while not is_us_trading_day(candidate_date.strftime("%Y-%m-%d")):
            candidate_date += timedelta(days=1)
    elif reference_time.time() >= datetime.min.replace(hour=16).time():
        candidate_date = get_next_trading_day(candidate_date)

    return datetime(
        candidate_date.year,
        candidate_date.month,
        candidate_date.day,
        9,
        30,
        tzinfo=US_MARKET_TZ,
    )


def format_time_delta(delta):
    total_seconds = max(int(delta.total_seconds()), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes = remainder // 60
    return f"{hours}h {minutes}m"


def has_stable_completed_close(market_now):
    market_close = market_now.replace(hour=16, minute=0, second=0, microsecond=0)
    stable_after = market_close + timedelta(hours=MARKET_CLOSE_STABILIZATION_HOURS)
    return market_now >= stable_after


def get_prediction_target_context(latest_trading_timestamp):
    latest_date = pd.Timestamp(latest_trading_timestamp).date()
    market_now = datetime.now(US_MARKET_TZ)
    market_date = market_now.date()
    market_open = market_now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = market_now.replace(hour=16, minute=0, second=0, microsecond=0)
    market_is_open_day = is_us_trading_day(market_date.strftime("%Y-%m-%d"))
    stable_completed_close = has_stable_completed_close(market_now)

    if market_is_open_day and market_open <= market_now < market_close:
        target_date = market_date if market_date > latest_date else latest_date
        target_label = f"US market open; closes in {format_time_delta(market_close - market_now)}"
    elif market_is_open_day and market_now < market_open:
        target_date = market_date if market_date > latest_date else latest_date
        target_label = f"US market not open yet; opens in {format_time_delta(market_open - market_now)}"
    elif market_is_open_day and not stable_completed_close:
        target_date = market_date if market_date > latest_date else latest_date
        stable_after = market_close + timedelta(hours=MARKET_CLOSE_STABILIZATION_HOURS)
        target_label = f"US market closed; waiting {format_time_delta(stable_after - market_now)} for stable close data"
    else:
        reference_date = market_date if market_date > latest_date else latest_date
        target_date = get_next_trading_day(reference_date)
        next_open = datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            9,
            30,
            tzinfo=US_MARKET_TZ,
        )
        target_label = f"US market closed; next session opens in {format_time_delta(next_open - market_now)}"

    return {
        "latest_date": latest_date,
        "target_date": target_date,
        "target_label": target_label,
        "market_now": market_now,
    }


def keep_completed_market_data(df):
    if df.empty:
        return df

    market_now = datetime.now(US_MARKET_TZ)
    market_date = market_now.date()
    last_row_date = pd.Timestamp(df.index[-1]).date()
    stable_completed_close = has_stable_completed_close(market_now)

    if market_date.weekday() < 5 and not stable_completed_close and last_row_date == market_date:
        completed_df = df.iloc[:-1]
        if not completed_df.empty:
            return completed_df

    return df


def coerce_series(values):
    if isinstance(values, pd.DataFrame):
        return values.iloc[:, 0]
    return values


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def render_app_header(logo_path, title, theme):
    with open(logo_path, "rb") as image_file:
        encoded_logo = base64.b64encode(image_file.read()).decode("utf-8")

    st.markdown(
        f"""
        <div class="app-header">
            <img class="app-header-logo" src="data:image/png;base64,{encoded_logo}" alt="Lupa logo">
            <h1 class="app-header-title">{title}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def has_enough_memory_for_finbert(min_available_mb=FINBERT_MIN_AVAILABLE_MB):
    if psutil is None:
        return True

    available_mb = psutil.virtual_memory().available / 1024 / 1024
    return available_mb >= min_available_mb


# ---------- Data Layer: market history, news, and seasonality inputs ----------

@st.cache_data(ttl=180)
def download_single_ticker_history(symbol, period):
    df = yf.download(symbol, period=period, progress=False, auto_adjust=False)

    if not df.empty:
        return df

    try:
        fallback_df = yf.Ticker(symbol).history(period=period, auto_adjust=False)
        return fallback_df
    except Exception:
        return df


@st.cache_data(ttl=900)
def download_multi_ticker_history(tickers, period):
    return yf.download(list(tickers), period=period, progress=False, auto_adjust=False, group_by="ticker")


@st.cache_data(ttl=900)
def load_price_data(symbol, period):
    df = download_single_ticker_history(symbol, period)

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.get_level_values(0):
            df = df[symbol]
        elif symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, axis=1, level=-1)
        else:
            df.columns = df.columns.get_level_values(0)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["MA20"] = df["Close"].rolling(20).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(20).std() * np.sqrt(252)

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

    df["BB_std"] = df["Close"].rolling(20).std()
    df["BB_upper"] = df["MA20"] + 2 * df["BB_std"]
    df["BB_lower"] = df["MA20"] - 2 * df["BB_std"]

    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_momentum"] = df["Volume"] / df["Volume_MA20"]
    return df


@st.cache_data(ttl=600)
def get_news(symbol):
    today = datetime.today().strftime("%Y-%m-%d")
    last_week = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        return finnhub_client.company_news(symbol, _from=last_week, to=today)
    except Exception:
        return []


@st.cache_data(ttl=3600)
def get_almanac_signals():
    spy = download_single_ticker_history("SPY", "2y")

    if spy.empty:
        return {
            "jan_signal": "Neutral",
            "five_signal": "Neutral",
            "best6": "Neutral Season",
            "pres": "Unknown",
        }

    jan = spy[spy.index.month == 1]

    jan_signal = "Neutral"
    if len(jan) > 5:
        jan_close = coerce_series(jan["Close"])
        jan_return = float((jan_close.iloc[-1] / jan_close.iloc[0]) - 1)
        jan_signal = "Bullish" if jan_return > 0 else "Bearish"

    five_signal = "Neutral"
    jan5 = jan.head(5)
    if len(jan5) == 5:
        jan5_close = coerce_series(jan5["Close"])
        jan5_return = float((jan5_close.iloc[-1] / jan5_close.iloc[0]) - 1)
        five_signal = "Bullish" if jan5_return > 0 else "Bearish"

    current_month = datetime.now().month
    best6 = "Bullish Season" if current_month in [11, 12, 1, 2, 3, 4] else "Weak Season"

    year = datetime.now().year
    cycle = year % 4
    if cycle == 0:
        pres = "Election Year"
    elif cycle == 1:
        pres = "Post Election"
    elif cycle == 2:
        pres = "Midterm Weakness"
    else:
        pres = "Pre Election Bullish"

    return {
        "jan_signal": jan_signal,
        "five_signal": five_signal,
        "best6": best6,
        "pres": pres,
    }


@st.cache_resource
def load_finbert():
    if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
        return None, None

    if not has_enough_memory_for_finbert():
        return None, None

    try:
        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        return tokenizer, model
    except Exception:
        return None, None


def score_text_with_finbert(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0].tolist()

    labels = ["positive", "negative", "neutral"]
    scores = dict(zip(labels, probabilities))
    compound = scores["positive"] - scores["negative"]

    return {
        "label": max(scores, key=scores.get),
        "scores": scores,
        "compound": compound,
    }


@st.cache_data(ttl=1800, show_spinner=False)
def score_news_with_finbert(headlines):
    tokenizer, model = load_finbert()
    if tokenizer is None or model is None:
        return []

    scored_items = []
    for headline in headlines:
        stripped = headline.strip()
        if not stripped:
            continue

        sentiment = score_text_with_finbert(stripped, tokenizer, model)
        scored_items.append(
            {
                "headline": stripped,
                "label": sentiment["label"],
                "compound": float(sentiment["compound"]),
            }
        )

    return scored_items


def calculate_technical_sentiment(df):
    price = df["Close"].iloc[-1]
    ma20 = df["MA20"].iloc[-1]
    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    macd_signal = df["MACD_signal"].iloc[-1]
    ret_5d = df["Close"].pct_change(5).iloc[-1]
    volume_momentum = df["Volume_momentum"].iloc[-1]

    score = 50.0
    score += 10 if price > ma20 else -10
    score += 10 if rsi > 60 else -10 if rsi < 40 else 0
    score += 10 if macd > macd_signal else -10
    score += clamp(ret_5d * 100, -10, 10) if pd.notna(ret_5d) else 0
    score += 5 if volume_momentum > 1.1 else -5 if volume_momentum < 0.9 else 0

    return clamp(score, 0, 100)


def aggregate_news_sentiment(scored_news):
    if not scored_news:
        return 50.0

    average_compound = sum(item["compound"] for item in scored_news) / len(scored_news)
    return clamp((average_compound + 1) * 50, 0, 100)


def build_news_sentiment_summary(scored_news):
    if not scored_news:
        if psutil is not None and not has_enough_memory_for_finbert():
            return "FinBERT disabled due to memory limits; using technical signals only."
        return "FinBERT unavailable; using headlines without structured sentiment score."

    average_compound = sum(item["compound"] for item in scored_news) / len(scored_news)
    overall_label = "Positive" if average_compound > 0.1 else "Negative" if average_compound < -0.1 else "Neutral"
    return f"{overall_label} ({average_compound:+.2f} compound from recent headlines)"


def build_prompt_news_summary(news_items, scored_news):
    headlines = []

    latest_headline = next((item.get("headline", "").strip() for item in news_items if item.get("headline")), "")
    if latest_headline:
        headlines.append(latest_headline[:120])

    if scored_news:
        strongest_item = max(scored_news, key=lambda item: abs(item["compound"]))
        strongest_headline = strongest_item["headline"].strip()
        if strongest_headline and strongest_headline != latest_headline:
            headlines.append(strongest_headline[:120])
    else:
        fallback_headline = next(
            (
                item.get("headline", "").strip()
                for item in news_items[1:]
                if item.get("headline") and item.get("headline").strip() != latest_headline
            ),
            "",
        )
        if fallback_headline:
            headlines.append(fallback_headline[:120])

    return " | ".join(headlines) if headlines else "No significant recent news."


# ---------- Forecasting: XGBoost model, LLM prompt, and ensemble output ----------

def train_model(X, y):
    model = XGBRegressor(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=1,
    )
    model.fit(X, y)
    return model


def price_forecast(df, window=20):
    feature_columns = [
        "Close",
        "MA20",
        "RSI",
        "Returns",
        "Volatility",
        "MACD",
        "MACD_signal",
        "BB_upper",
        "BB_lower",
        "Volume_momentum",
    ]

    training_df = df.tail(350).dropna()
    if len(training_df) <= window:
        return float(df["Close"].iloc[-1])

    feature_matrix = training_df[feature_columns].values

    X = []
    y = []
    for index in range(window, len(feature_matrix)):
        X.append(feature_matrix[index - window : index].flatten())
        y.append(feature_matrix[index][0])

    X = np.array(X)
    y = np.array(y)

    model = train_model(X, y)
    last_window = feature_matrix[-window:].flatten().reshape(1, -1)
    return float(model.predict(last_window)[0])



def build_llm_prompt(symbol, price, trend, df, news_summary, news_sentiment_summary, almanac, target_context):
    reference_close_date = target_context["latest_date"]
    target_date = target_context["target_date"]
    market_status = target_context["target_label"]

    return f"""
    You are a professional quantitative hedge fund analyst.

    [DATA]
    Stock: {symbol}
    Reference close date: {reference_close_date}
    Reference close price: {price}
    Target close date: {target_date}
    US market status: {market_status}

    RSI: {df['RSI'].iloc[-1]:.2f}
    Volatility: {df['Volatility'].iloc[-1]:.2%}
    Trend (MA20): {trend}

    Recent News Headlines:
    {news_summary}

    News Sentiment:
    {news_sentiment_summary}

    Almanac and seasonality signals (weak context only):
    - January Barometer: {almanac["jan_signal"]} (January direction signal)
    - First 5 Trading Days: {almanac["five_signal"]} (early-year momentum signal)
    - Best 6 Months: {almanac["best6"]} (seasonal strength signal)
    - Presidential Cycle: {almanac["pres"]} (election-cycle context)

    [INSTRUCTIONS]

    1. Predict the CLOSE price for the target US trading session.

    - Reference close date: {reference_close_date}
    - Reference close price: {price}
    - Target close date: {target_date}
    - target_price MUST be the closing price of the target date

    2. Provide:
    - target_price: realistic closing price (within +/-10%)
    - confidence: 0 to 1

    3. Use:
    - technical indicators
    - news sentiment
    - almanac/seasonality only as weak supporting context

    4. Rules:
    - Do NOT predict intraday high/low
    - Do NOT output a price range
    - Output ONE single closing price

    5. Be decisive

    [OUTPUT FORMAT - JSON ONLY]
    {{
      "target_price": 210.5,
      "confidence": 0.72,
      "reason": "max 15 sentences"
    }}
    """


@st.cache_data(ttl=600)
def run_llm(prompt):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def parse_llm_response(llm_text, fallback_price):
    try:
        llm_data = json.loads(llm_text)
        llm_price = float(llm_data.get("target_price", fallback_price) or fallback_price)
        llm_conf = float(llm_data.get("confidence", 0.5) or 0.5)
        llm_conf = min(max(llm_conf, 0), 1)
        llm_reason = (llm_data.get("reason") or "No reasoning provided")[:2000]
        return llm_price, llm_conf, llm_reason, None
    except Exception:
        return fallback_price, 0.5, "No analysis available", llm_text


def build_forecast_result(ticker, current_price, pred_price, llm_price, llm_conf, llm_reason, target_context):
    llm_conf = min(max(llm_conf, 0.2), 0.8)
    blend_info = get_dynamic_blend_weights(ticker, llm_conf)
    weight_xgb = blend_info["weight_xgb"]
    weight_llm = blend_info["weight_llm"]
    ensemble_price = (pred_price * weight_xgb) + (llm_price * weight_llm)

    return {
        "ensemble_price": ensemble_price,
        "llm_price": llm_price,
        "pred_price": pred_price,
        "llm_reason": llm_reason,
        "llm_conf": llm_conf,
        "weight_xgb": weight_xgb,
        "weight_llm": weight_llm,
        "weight_source": blend_info["source"],
        "weight_sample_count": blend_info["sample_count"],
        "mae_xgb": blend_info["mae_xgb"],
        "mae_llm": blend_info["mae_llm"],
        "signal_text": "BUY" if ensemble_price > current_price else "SELL",
        "predicted_change_pct": ((ensemble_price - current_price) / current_price) * 100,
        "reference_close_date": target_context["latest_date"].strftime("%Y-%m-%d"),
        "predicted_date": target_context["target_date"].strftime("%Y-%m-%d"),
        "predicted_label": target_context["target_label"],
    }


# ---------- Charts: reusable Plotly figures for dashboard sections ----------

def build_sentiment_gauge(sentiment, theme):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=sentiment,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Market Sentiment", "font": {"color": theme["text_color"]}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": theme["text_color"],
                    "tickfont": {"color": theme["text_color"]},
                },
                "bar": {"color": "#3b82f6"},
                "steps": [
                    {"range": [0, 40], "color": "#ef4444"},
                    {"range": [40, 60], "color": "#facc15"},
                    {"range": [60, 100], "color": "#22c55e"},
                ],
            },
        )
    )
    fig.update_layout(
        template=theme["plotly_template"],
        autosize=False,
        width=760,
        height=420,
        margin={"l": 40, "r": 40, "t": 60, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": theme["text_color"]},
    )
    fig.update_traces(number={"font": {"color": theme["text_color"]}})
    return fig


def build_price_chart(df, theme):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MA20"],
            line=dict(color="#60a5fa", width=2),
            name="MA20",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color="rgba(120,160,255,0.3)",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=650,
        hovermode="x unified",
        dragmode="pan",
        template=theme["plotly_template"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": theme["text_color"]},
        legend=dict(
            bgcolor=theme["card_bg"],
            bordercolor=theme["grid_color"],
            borderwidth=1,
            font={"color": theme["text_color"]},
        ),
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
    )
    fig.update_xaxes(tickfont=dict(color=theme["text_color"]), gridcolor=theme["grid_color"])
    fig.update_yaxes(tickfont=dict(color=theme["text_color"]), gridcolor=theme["grid_color"])
    return fig


def build_heatmap_chart(heatmap_df, theme):
    fig = px.bar(
        heatmap_df,
        x="Ticker",
        y="Change",
        color="Change",
        text="Change",
        color_continuous_scale="RdYlGn",
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(
        height=450,
        template=theme["plotly_template"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": theme["text_color"]},
    )
    fig.update_xaxes(tickfont=dict(color=theme["text_color"]))
    fig.update_yaxes(tickfont=dict(color=theme["text_color"]))
    return fig


# ---------- UI Helpers: repeated cards and tab-specific render blocks ----------

def render_value_card(title, value, signal_text, signal_color, theme, extra_text=None):
    st.markdown(
        f"""
        <div class="themed-card" style="padding: 20px; margin-top: 10px;">
            <p style="color:{theme["muted_text_color"]};">{title}</p>
            <h2>${value:.2f}</h2>
            <span style="color:{signal_color}; font-weight:600;">{signal_text}</span>
            {f'<p style="color:{theme["muted_text_color"]}; margin-top: 10px;">{extra_text}</p>' if extra_text else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_card(pred_price, current_price, theme):
    signal_text, signal_color = get_signal_style(pred_price, current_price)
    st.markdown(
        f"""
        <div class="themed-card" style="padding: 20px;">
            <p style="color:{theme["muted_text_color"]};">Predicted Price</p>
            <h2>${pred_price:.2f}</h2>
            <span style="color:{signal_color}; font-weight:600;">{signal_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_card(forecast_result):
    signal_class = "signal-buy" if forecast_result["signal_text"] == "BUY" else "signal-sell"
    arrow = "\u2191" if forecast_result["signal_text"] == "BUY" else "\u2193"
    reference_close_date = forecast_result.get("reference_close_date", "last completed")
    predicted_label = forecast_result.get("predicted_label", "")

    st.markdown(
        f"""
        <div class="signal-card" style="padding: 25px; margin-bottom: 10px;">
            <div class="signal-card-title {signal_class}">{arrow} {forecast_result["signal_text"]}</div>
            <div class="signal-card-meta">
                Forecast for {forecast_result["predicted_date"]} |
                {forecast_result["predicted_change_pct"]:+.2f}% vs {reference_close_date} close
            </div>
            <div class="signal-card-meta">{predicted_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_news_tab(symbol, news_items, scored_news, theme):
    st.subheader(f"{symbol} News")

    scored_lookup = {item["headline"]: item for item in scored_news}

    for news_item in news_items[:10]:
        headline = news_item.get("headline", "No title")
        url = news_item.get("url", "#")
        summary = news_item.get("summary", "")
        date = datetime.fromtimestamp(news_item.get("datetime", 0)).strftime("%Y-%m-%d")
        st.markdown(f"**[{headline}]({url})**")
        if headline in scored_lookup:
            sentiment = scored_lookup[headline]
            st.caption(
                f'FinBERT: {sentiment["label"].title()} | compound {sentiment["compound"]:+.2f}'
            )
        st.markdown(
            f'<div style="color:{theme["text_color"]}; white-space: pre-wrap;">{summary}</div>',
            unsafe_allow_html=True,
        )
        st.caption(date)
        st.divider()


def render_almanac_tab(almanac):
    st.header("Market Seasonality (Stock Trader's Almanac)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("January Barometer", almanac["jan_signal"])
    with col2:
        st.metric("First Five Days", almanac["five_signal"])
    with col3:
        st.metric("Best Six Months", almanac["best6"])
    st.subheader("Presidential Cycle")
    st.info(almanac["pres"])


def load_heatmap_data():
    batch_data = download_multi_ticker_history(tuple(BIG_TECHS), "5d")
    if batch_data.empty:
        return pd.DataFrame()

    rows = []
    for ticker in BIG_TECHS:
        try:
            if isinstance(batch_data.columns, pd.MultiIndex):
                ticker_frame = batch_data[ticker]
                close = coerce_series(ticker_frame["Close"])
            else:
                close = coerce_series(batch_data["Close"])
            change = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100
            rows.append({"Ticker": ticker, "Change": float(change)})
        except Exception:
            continue
    return pd.DataFrame(rows)


# ---------- Main Page: assemble sidebar, load data, and render dashboard ----------

initialize_session_state()

dark_mode = st.sidebar.toggle("Night Mode", value=True)
theme = get_theme(dark_mode)
apply_theme(theme)

logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
render_app_header(logo_path, "Lupa AI Stock Terminal", theme)

st.sidebar.text_input("Ticker", key="ticker", on_change=on_ticker_changed)
st.sidebar.radio("Big Tech", BIG_TECHS, key="bigtech", index=None, on_change=on_bigtech_changed)
period = st.sidebar.selectbox("Analysis Window", PERIOD_OPTIONS, index=2)

symbol = st.session_state.ticker.upper()
raw_df = load_price_data(symbol, period)
df = keep_completed_market_data(raw_df)

if df.empty:
    st.error("Ticker not found or latest completed close is not available yet")
    st.stop()

news = get_news(symbol)
almanac = get_almanac_signals()
headline_list = [item.get("headline", "") for item in news[:10] if item.get("headline")]
scored_news = score_news_with_finbert(headline_list)

price = df["Close"].iloc[-1]
ret = df["Returns"].iloc[-1]
trend = "Bullish" if price > df["MA20"].iloc[-1] else "Bearish"
technical_sentiment = calculate_technical_sentiment(df)
news_sentiment = aggregate_news_sentiment(scored_news)
sentiment = 0.7 * technical_sentiment + 0.3 * news_sentiment
pred_price = price_forecast(df)
target_context = get_prediction_target_context(df.index[-1])

news_sentiment_summary = build_news_sentiment_summary(scored_news)
news_summary = build_prompt_news_summary(news, scored_news)

st.markdown(f"## {symbol} Market Overview")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
with metric_col1:
    st.metric("Price", f"${price:.2f}", f"{ret:.2%}")
with metric_col2:
    st.metric("Trend", trend)
with metric_col3:
    st.metric("Volatility", f"{df['Volatility'].iloc[-1]:.2%}")
with metric_col4:
    st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")

_, sentiment_center, _ = st.columns([1, 2, 1])
with sentiment_center:
    st.plotly_chart(build_sentiment_gauge(sentiment, theme), width="content")
    with st.expander("How Market Sentiment Is Calculated"):
        st.markdown(
            f"""
            `Market Sentiment` is a composite 0-100 score:

            - `70%` Technical sentiment
            - `30%` FinBERT news sentiment

            Current breakdown:

            - Technical sentiment: `{technical_sentiment:.1f}`
            - News sentiment: `{news_sentiment:.1f}`
            - Final score: `{sentiment:.1f}`

            Technical sentiment is derived from:
            - Price vs `MA20`
            - `RSI`
            - `MACD` vs signal line
            - 5-day return
            - Volume momentum

            News sentiment is derived from recent headlines scored by `FinBERT`.
            """
        )

tab_chart, tab_ai, tab_almanac, tab_heat, tab_news = st.tabs(
    ["Chart", "AI Forecast", "Almanac", "Heatmap", "News"]
)

with tab_chart:
    st.plotly_chart(
        build_price_chart(df, theme),
        width="stretch",
        config={"scrollZoom": True},
    )

with tab_ai:
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("XGBoost Prediction")
        render_prediction_card(pred_price, price, theme)

    llm_prompt = build_llm_prompt(
        symbol,
        price,
        trend,
        df,
        news_summary,
        news_sentiment_summary,
        almanac,
        target_context,
    )
    run_llm_clicked = False

    with right_col:
        st.markdown('<div style="height: 150px;"></div>', unsafe_allow_html=True)
        _, button_col, _ = st.columns([1, 2, 1])
        with button_col:
            run_llm_clicked = st.button(
                "Run LLM Analysis",
                key="llm_button",
                width="stretch",
            )

    if run_llm_clicked:
        try:
            update_actual_closes_in_log()
        except YFRateLimitError:
            pass
        llm_text = run_llm(llm_prompt)
        llm_price, llm_conf, llm_reason, llm_parse_error = parse_llm_response(llm_text, price)

        if llm_parse_error is not None:
            st.error("LLM parsing failed")
            st.write(llm_parse_error)

        st.session_state[FORECAST_STATE_KEY] = build_forecast_result(
            ticker=symbol,
            current_price=price,
            pred_price=pred_price,
            llm_price=llm_price,
            llm_conf=llm_conf,
            llm_reason=llm_reason,
            target_context=target_context,
        )
        record_status, record_error = append_prediction_log_record(
            ticker=symbol,
            reference_close_price=price,
            forecast_result=st.session_state[FORECAST_STATE_KEY],
        )
        if record_status == "supabase":
            st.caption("Prediction logged to Supabase for dynamic weighting.")
        elif record_status == "csv":
            st.caption("Prediction logged locally for dynamic weighting.")
            if record_error:
                st.caption(record_error)
        else:
            st.caption("Prediction for this ticker and target date is already logged.")

    forecast_result = st.session_state.get(FORECAST_STATE_KEY)
    if forecast_result:
        forecast_result.setdefault("reference_close_date", "last completed")
        forecast_result.setdefault("predicted_label", "")
        st.markdown("### LLM Analysis")
        st.markdown(
            f"""
            <div class="themed-card" style="
                padding: 15px;
                border-radius: 10px;
                font-size: 15px;
                line-height: 1.6;
                margin-bottom: 10px;
            ">
                {forecast_result["llm_reason"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_signal_card(forecast_result)

        for title, value in [
            ("Ensemble Price", forecast_result["ensemble_price"]),
            ("LLM Price", forecast_result["llm_price"]),
            ("XGBoost Price", forecast_result["pred_price"]),
        ]:
            signal_text, signal_color = get_signal_style(value, price)
            extra_text = None
            if title == "Ensemble Price":
                blend_label = "Dynamic blend" if forecast_result.get("weight_source") == "dynamic" else "Default blend"
                extra_text = (
                    f'{blend_label}: XGBoost {forecast_result["weight_xgb"]:.0%} '
                    f'+ LLM {forecast_result["weight_llm"]:.0%}'
                )
                if forecast_result.get("weight_source") == "dynamic":
                    extra_text += f' | based on {forecast_result.get("weight_sample_count", 0)} completed runs'
                else:
                    extra_text += f' | waiting for {max(0, 5 - forecast_result.get("weight_sample_count", 0))} more completed runs'
                if forecast_result.get("mae_xgb") is not None and forecast_result.get("mae_llm") is not None:
                    extra_text += (
                        f' | XGB MAE: ${forecast_result["mae_xgb"]:.2f}'
                        f' | LLM MAE: ${forecast_result["mae_llm"]:.2f}'
                    )
            if title == "LLM Price":
                extra_text = f'Confidence: {forecast_result["llm_conf"]:.0%}'
            render_value_card(title, value, signal_text, signal_color, theme, extra_text=extra_text)

with tab_heat:
    heatmap_df = load_heatmap_data()
    if not heatmap_df.empty:
        st.plotly_chart(build_heatmap_chart(heatmap_df, theme), width="stretch")
    else:
        st.info("Heatmap data is temporarily unavailable.")

with tab_news:
    render_news_tab(symbol, news, scored_news, theme)

with tab_almanac:
    render_almanac_tab(almanac)
