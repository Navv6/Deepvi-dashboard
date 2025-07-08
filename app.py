import streamlit as st
import requests
import json
import os
import sys
import urllib.request
import re
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading
import plotly.graph_objects as go
from urllib.parse import urlencode
import html
from streamlit.components.v1 import html as components_html
import requests
from streamlit_searchbox import st_searchbox
from io import BytesIO
import base64
import time
import pytz
import os

# 페이지 설정 (더 빠른 렌더링을 위해)
st.set_page_config(
    page_title="DeepVi",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.session_state['prev_page'] = 'main'

# 현재 시간 표시 (서울 시간)
def get_current_time():
    seoul_tz = pytz.timezone('Asia/Seoul')
    return datetime.now(seoul_tz).strftime('%Y년-%m월-%d일 %H시%M분')

# 실시간 시계 컴포넌트 (st.fragment 사용)
@st.fragment(run_every=1)  # 1초마다 실행
def real_time_clock():
    current_time = get_current_time()
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 1rem;">
        <p style="color: #6B7280; font-size: 0.9rem; margin: 0;">
            🕐 {current_time}
        </p>
    </div>
    """, unsafe_allow_html=True)

# 실시간 시계 실행
real_time_clock()

#제목
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/pretendard@1.3.6/dist/web/static/pretendard.css" rel="stylesheet" />
<style>
    html, body, [class*="css"]  {
        font-family: 'Pretendard', sans-serif;
    }

    .deepview-card {
        background: #ffffff;
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin: 2rem auto;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.06);
        text-align: center;
        max-width: 900px;
        border: 1px solid #e5e7eb;
    }

    .deepview-title {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.6rem;
        font-size: 2.3rem;
        font-weight: 800;
        color: #1f2937;  /* Gray-800 */
    }

    .deepview-subtitle {
        color: #3B82F6;  /* Blue-500 */
        font-weight: 600;
        font-size: 1.05rem;
        margin-top: 0.5rem;
    }

    .deepview-desc {
        color: #6B7280;  /* Gray-500 */
        font-size: 0.95rem;
        margin-top: 0.6rem;
        line-height: 1.5;
    }

</style>

<div class="deepview-card">
    <div class="deepview-title">
        📊 <span>DeepVi: 기업을 깊이 읽다</span>
    </div>
    <div class="deepview-subtitle">
        AI가 읽어주는 기업 이야기, <b>DeepVi</b>에서 시작됩니다.
    </div>
    <div class="deepview-desc">
        데이터를 통해 기업의 본질을 바라보는 새로운 시선.
    </div>
</div>
""", unsafe_allow_html=True)

# 줄 바꿈
st.markdown("<br>", unsafe_allow_html=True)

# 캐시된 데이터 가져오기 함수 (TTL 설정으로 적절한 갱신)
@st.cache_data(ttl=600, show_spinner=False)  # 10분 캐시
def get_index_value_cached(code, date):
    try:
        data = fdr.DataReader(code, end=date)
        if data.empty:
            return None
        return float(data['Close'].values[-1])
    except Exception as e:
        return None

# 병렬로 모든 지수 데이터 가져오기
@st.cache_data(ttl=300, show_spinner=False)
def get_all_market_data():
    sysdate = datetime.today().date()
    yesterday = sysdate - timedelta(days=1)
    
    codes = ["KS11", "IXIC", "US500", "DJI", "USD/KRW", "KQ11"]
    
    def fetch_data(code):
        today_val = get_index_value_cached(code, sysdate)
        yesterday_val = get_index_value_cached(code, yesterday)
        return code, today_val, yesterday_val
    
    # 병렬 처리로 모든 데이터 가져오기
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(fetch_data, codes))
    
    return dict((code, (today, yesterday)) for code, today, yesterday in results)

# 뉴스 데이터 캐시
@st.cache_data(ttl=300, show_spinner=False)  # 5분 캐시
def get_naver_news_cached(query):
    naver_id = st.secrets["naver"]["client_id"]
    naver_secret = st.secrets["naver"]["client_secret"]
    
    try:
        encText = urllib.parse.quote(query)
        url = f"https://openapi.naver.com/v1/search/news.json?query={encText}&display=10"
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", st.secrets["naver"]["client_id"])
        request.add_header("X-Naver-Client-Secret", st.secrets["naver"]["client_secret"])
        
        with urllib.request.urlopen(request, timeout=5) as response:  # 타임아웃 설정
            if response.getcode() == 200:
                return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        st.error(f"뉴스 로딩 중 오류: {str(e)}")
    return None

# 세션 상태 초기화
if 'market_data_loaded' not in st.session_state:
    st.session_state.market_data_loaded = False
if 'has_run_default_search' not in st.session_state:
    st.session_state.has_run_default_search = False

# 로딩 상태 표시
loading_placeholder = st.empty()

# 백그라운드에서 시장 데이터 로드
if not st.session_state.market_data_loaded:
    with loading_placeholder.container():
        with st.spinner("시장 데이터 로딩 중..."):
            market_data = get_all_market_data()
            st.session_state.market_data = market_data
            st.session_state.market_data_loaded = True

loading_placeholder.empty()

# 시장 데이터 표시
market_data = st.session_state.market_data

FASTAPI_URL = "http://218.50.44.25:8000"
# 후보 함수: 입력값을 받아 FastAPI에서 후보 받아옴
def search_company(q):
    if not q:
        return []
    resp = requests.get(f"{FASTAPI_URL}/autocomplete", params={"q": q})
    if resp.status_code == 200:
        # 기업명만 리스트로 반환!
        return [item["stock_name"] for item in resp.json()]
    return []

st.caption("🔍 검색 시 해당 기업의 분석페이지로 이동합니다.")
selected = st_searchbox(
    search_company,
    key="search_company",
    placeholder="기업명을 입력하세요",
    clear_on_submit=True,
)

if selected:
    encoded = urllib.parse.quote(selected)
    st.session_state.company = selected  # 세션 상태에 인코딩된 값 저장
    st.switch_page("pages/stream.py")

# 상태 초기화
if "selected_index" not in st.session_state:
    st.session_state.selected_index = None

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-top: -0.5rem;">
    <p style="color: #6B7280; font-size: 0.9rem; margin: 0;">
        📈 코스피부터 나스닥까지, 글로벌 금융시장의 핵심 지표를 실시간으로 확인하세요.
    </p>
</div>
""", unsafe_allow_html=True)
# 카드 스타일 CSS
st.markdown("""
<style>
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    text-align: center;
    transition: all 0.3s ease;
    height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    color: #1F2937;
    overflow: hidden;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
}

.metric-card.highlighted {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: 2px solid #667eea;
}

.metric-title {
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 0.3rem;
    color: inherit;
}

.metric-value {
    font-size: 18px;
    font-weight: bold;
    margin: 0.3rem 0;
    line-height: 1.2;
}

.metric-change {
    font-size: 15px;
    font-weight: 500;
    margin: 0.2rem 0;
    line-height: 1.2;
}

.metric-date {
    font-size: 15px;
    opacity: 0.7;
    margin-top: 0.2rem;
    line-height: 1.1;
}
</style>
""", unsafe_allow_html=True)

# 카드에서 사용할 지수 목록
metric_list = [
    {"key": "KS11",    "title": "코스피",   "highlighted": False,  "inv_color": False, "format": "{:.2f}"},
    {"key": "KQ11",    "title": "코스닥", "highlighted": False, "inv_color": False,  "format": "{:.2f}"},
    {"key": "USD/KRW", "title": "USD/KRW", "highlighted": False, "inv_color": True,  "format": "{:.2f}"},
    {"key": "IXIC",    "title": "나스닥",   "highlighted": False, "inv_color": False, "format": "{:.2f}"},
    {"key": "DJI",     "title": "다우존스", "highlighted": False, "inv_color": False, "format": "{:.2f}"},
    {"key": "US500",   "title": "S&P500",  "highlighted": False, "inv_color": False, "format": "{:.2f}"},
]


# 카드 HTML 생성
card_data_html = ""
for metric in metric_list:
    key = metric["key"]
    title = metric["title"]
    inv_color = metric.get("inv_color", False)
    fmt = metric.get("format", "{:.2f}")

    value, prev_value = market_data.get(key, (None, None))
    if value is None or prev_value is None:
        value_html = "로딩중..."
        change_html = "-"
        percent_html = "-"
        color = "#6B7280"
    else:
        change = value - prev_value
        change_percent = (change / prev_value) * 100 if prev_value else 0
        arrow = "▲" if change_percent > 0 else "▼" if change_percent < 0 else ""
        if inv_color:
            color = "#EF4444" if change_percent > 0 else "#10B981" if change_percent < 0 else "#6B7280"
        else:
            color = "#10B981" if change_percent > 0 else "#EF4444" if change_percent < 0 else "#6B7280"

        value_html = fmt.format(value)
        change_html = f"{arrow} {change:+.2f}"
        percent_html = f"({change_percent:+.2f}%)"
    
    


    card_data_html += f"""
        <div class="metric-card-wrapper">
             <div class="metric-card">
             <div class="metric-title">{title}</div>
            <div class="metric-value" style="color:{color}">{value_html}</div>
            <div class="metric-change" style="color:{color}">{change_html}</div>
         <div class="metric-date">{percent_html}</div>
        </div>
    </div>
    """

# CSS 포함 전체 HTML 렌더링
responsive_cards_html = f"""
<style>
    .card-grid {{
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        justify-content: center;
        margin-bottom: 2rem;
    }}

    .metric-card-wrapper {{
        position: relative;
        display: inline-flex;
        flex-direction: column;
        align-items: center;
        max-width: 100%;
        width: calc(16.66% - 1rem);  /* 카드 크기 고정 */
        overflow: visible;  /* 카드 자체는 overflow 숨기지 않음 */
    }}

    .metric-card {{
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        transition: all 0.3s ease;
        color: #1F2937;
        width: 80%;
    }}

    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }}

    .metric-title {{
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }}

    .metric-value {{
        font-size: 1.5rem;
        font-weight: bold;
        margin: 0.3rem 0;
    }}

    .metric-change {{
        font-size: 1rem;
        margin-bottom: 0.2rem;
    }}

    .metric-date {{
        font-size: 0.9rem;
        opacity: 0.75;
    }}

    /* 📱 반응형 (모바일 대응) */
    @media screen and (max-width: 768px) {{
        .metric-card-wrapper {{
            width: calc(33.33% - 1rem);
        }}
        .metric-title {{
            font-size: 0.95rem;
        }}
        .metric-value {{
            font-size: 1.2rem;
        }}
        .metric-change {{
            font-size: 0.9rem;
        }}
        .metric-date {{
            font-size: 0.85rem;
        }}
    }}
</style>

<div class="card-grid">
    {card_data_html}
</div>
"""

components_html(responsive_cards_html, height=300)




# 성능 최적화를 위한 추가 CSS
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
    
    /* 전체 페이지 스타일링 */
    .main > div {
        padding-top: 1rem;
    }
    
    /* 카드 그리드 간격 조정 */
    .stColumn {
        padding: 0 5px;
    }
</style>
""", unsafe_allow_html=True)
