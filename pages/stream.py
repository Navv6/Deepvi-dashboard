# ------------------------ 기본 라이브러리 ------------------------
from langchain_pinecone import PineconeVectorStore
import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta, date
from typing import Generator
from streamlit_searchbox import st_searchbox
# ------------------------ 시각화 ------------------------
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch.version
from wordcloud import WordCloud
import matplotlib.font_manager as fm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
# ------------------------ 요청 및 기타 ------------------------
import requests
from streamlit.components.v1 import html
import streamlit.runtime.scriptrunner.script_runner as script_runner
import ast  # 문자열 → 리스트/딕셔너리 변환용
# ------------------------ 임베딩 및 LLM ------------------------
import torch
from sentence_transformers import SentenceTransformer
# ------------------------ Pinecone & LangChain ------------------------
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
# ------------------------ Google Gemini ------------------------
import google.generativeai as genai
from google.generativeai import GenerativeModel  # ✅ 핵심
from google.generativeai import types
# ------------------------ OpenAI (직접 사용 시) ------------------------
from openai import OpenAI  
import warnings
import matplotlib.font_manager as fm
# ------------------------ 스트림릿 추가 라이브러리
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')

# 우분투에서 사용 가능한 폰트 경로 찾기
font_path = os.path.join(os.path.dirname(__file__), "../fonts/NanumGothic.ttf")  # pages 폴더 내부일 경우
fm.fontManager.addfont(font_path)
plt.rc('font', family='NanumGothic')
# 단위설정
def format_korean_number_for_dashboard(col, value, decimals=1, show_unit=True):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value) if value is not None else ""
    
    # 시가총액은 '원' 단위 (예: 44.4조, 1.2조, 3,400억)
    if col == "시가총액":
        if abs(v) >= 1e12:
            s = f"{v/1e12:.{decimals}f}"
            return s + ("조" if show_unit else "")
        elif abs(v) >= 1e8:
            s = f"{v/1e8:.{decimals}f}"
            return s + ("억" if show_unit else "")
        else:
            s = f"{v:,.0f}"
            return s + ("원" if show_unit else "")
    
    # 나머지(매출액, 영업이익, 당기순이익 등)는 억 단위 (예: 1,074,488억, 12.3조)
    elif col in ["매출액", "영업이익", "당기순이익", "자산", "부채", "자본","현금"]:
        if abs(v) >= 10000:  # 만 억 = 1조
            s = f"{v/10000:.{decimals}f}"
            return s + ("조" if show_unit else "")
        elif abs(v) >= 1:
            s = f"{v:.{decimals}f}"
            return s + ("억" if show_unit else "")
        elif abs(v) >= 0.01:
            s = f"{v*10000:.{decimals}f}"
            return s + ("만" if show_unit else "")
        else:
            s = f"{v*1e8:,.0f}"
            return s + ("원" if show_unit else "")
    
    # 비율/퍼센트
    elif any(kw in col for kw in ["비율","률", "율", "ROE", "ROA"]):
        return f"{v:.1f}%"
    else:
        return value

# 페이지설정
st.set_page_config(
    page_title="DeepVI",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# FastAPI 서버 URL
FASTAPI_URL = "http://218.50.44.25:8000"

# --------------------------
# FastAPI 호출 함수 정의 + 캐싱 적용
# --------------------------
@st.cache_data(show_spinner="📡 기업 데이터를 불러오는 중입니다...", ttl=600)
def fetch_company_data(company):
    params = {"company": company}
    response = requests.get(f"{FASTAPI_URL}/company_data", params=params)
    if response.status_code != 200:
        raise Exception("FastAPI에서 데이터를 받아오는 데 실패했습니다.")
    return response.json()
# 🔍 FastAPI 검색 후보 함수
def search_company(q):
    if not q:
        return []
    resp = requests.get(f"{FASTAPI_URL}/autocomplete", params={"q": q})
    if resp.status_code == 200:
        return [item["stock_name"] for item in resp.json()]
    return []

# --------------------------
# 기업명 세션 확인 및 호출
# --------------------------
company_name = st.session_state.get("company")
if not company_name:
    st.warning("기업명을 선택해 주세요!")
    st.stop()

try:
    data = fetch_company_data(company_name)
except Exception as e:
    st.error(f"데이터 호출 실패: {e}")
    st.stop()

# 수평 레이아웃 구성 (왼쪽 col1 비움)
col1, col2 = st.columns([3, 2])

with col1:
    st.write("DeepVi: 데이터 기반의 AI 기업 분석 플랫폼")

with col2:
    # 🔍 오른쪽 상단 검색창
    selected = st_searchbox(
        search_company,
        key="search_company",
        placeholder="🔍 다른 기업을 검색하세요",
        label="",
        clear_on_submit=True,
    )
# 📢 아래쪽 기업명 타이틀
import textwrap
st.markdown(textwrap.dedent(f"""
    <div style="
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;">
        <h1 style="
            font-size: 2.2rem;
            font-weight: 700;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.12);
            margin: 0;
            border-left: 6px solid #4A90E2;
            padding-left: 1rem;
            color: #2c3e50;">
            {company_name}
        </h1>
    </div>
"""), unsafe_allow_html=True)

# ✅ 기업 선택 시 리렌더링
if selected and selected != st.session_state.get("company"):
    st.session_state["company"] = selected
    st.rerun()

# 메인페이지로 돌아가기
st.markdown("""
<style>
.back-btn {
    position: fixed;
    bottom: 30px;
    right: 30px;
    background-color: #f4f4f5;
    border: 1px solid #ccc;
    padding: 10px 14px;
    border-radius: 25px;
    font-size: 0.9rem;
    color: #444;
    text-decoration: none;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    z-index: 999;
}
.back-btn:hover {
    background-color: #e2e2e2;
}
</style>
<a href="/" class="back-btn">← 뒤로 가기</a>
""", unsafe_allow_html=True)

# ---------------------- 상단: 기업 정보 + ESG + 분석 코멘트 ----------------------
industry = data.get("market_cap", {}).get('업종', 'N/A')
market_cap = format_korean_number_for_dashboard("시가총액", data.get("market_cap", {}).get('시총', 'N/A'))
esg = data.get("esg", {}).get('grade', 'N/A')
invest_type = data['invest_info'].get("투자유형", "N/A")
invest_type_des = data['invest_info'].get("description", "N/A")
esg_description = """
ESG는 <b>환경(E)</b>, <b>사회(S)</b>, <b>지배구조(G)</b>의 세 분야를 기준으로 평가되며,<br>
본 등급은 <b>KCGS (한국ESG기준원)</b>의 평가 데이터를 기반으로 산출된 종합 등급입니다.
"""
# ------------------ 카드 스타일 + HTML ------------------
st.markdown(f"""
<style>
.metrics-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 1rem 1.3rem;
    justify-content: center;
    margin-bottom: 1.8rem;
}}

.metric-card2, .metric-card3 {{
    background: #fff;
    border-radius: 18px;
    box-shadow: 0 2px 10px rgba(80,90,150,0.07);
    border: 1.3px solid #f0f1f5;
    padding: 1.1rem 1.1rem 0.7rem 1.1rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1 1 calc(70%  rem);  /* 두 칸으로 유지 */
    min-width: 140px;
    max-width: 220px;
    box-sizing: border-box;
}}

.metric-card3 {{
    background: #e9f0fc;
}}

.metric-label2 {{
    font-size: 1.05rem;
    color: #73787e;
    font-weight: 500;
    margin-bottom: .17rem;
    letter-spacing: 0.01em;
    position: relative;
    display: inline-block;
    text-align: center;
}}

.tooltip-icon {{
    font-size: 0.7rem;
    position: absolute;
    top: -4px;
    right: -14px;
    color: #999;
    cursor: help;
}}

.tooltip-icon .tooltiptext {{
    visibility: hidden;
    width: 220px;
    background-color: #333;
    color: #fff;
    text-align: left;
    border-radius: 6px;
    padding: 8px;
    position: absolute;
    z-index: 1;
    bottom: 120%;
    left: 50%;
    transform: translateX(-50%);
    opacity: 0;
    transition: opacity 0.3s;
    font-size: 0.7rem;
    line-height: 1.4;
}}

.tooltip-icon:hover .tooltiptext {{
    visibility: visible;
    opacity: 1;
}}

.metric-value2 {{
    font-size: 1.15rem;
    font-weight: 700;
    color: #31353b;
    margin-bottom: .18rem;
    letter-spacing: -0.5px;
    word-break: keep-all;
    white-space: nowrap;
}}

@media (max-width: 600px) {{
    .metric-card2, .metric-card3 {{
        flex: 1 1 calc(50% - 1.3rem);
        max-width: 100%;
    }}
}}
</style>

<div class="metrics-row">
  <div class="metric-card2 industry-card">
    <div class="metric-label2">업종</div>
    <div class="metric-value2">{industry}</div>
  </div>
  <div class="metric-card2">
    <div class="metric-label2">시가총액</div>
    <div class="metric-value2">{market_cap}</div>
  </div>
  <div class="metric-card2">
    <div class="metric-label2">
      ESG
      <span class="tooltip-icon">🛈
        <span class="tooltiptext">{esg_description}</span>
      </span>
    </div>
    <div class="metric-value2">{esg}</div>
  </div>
  <div class="metric-card3">
    <div class="metric-label2">
      주식유형
      <span class="tooltip-icon">🛈
        <span class="tooltiptext">{invest_type_des}</span>
      </span>
    </div>
    <div class="metric-value2">{invest_type}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# 기업명과 분석 코멘트 가져오기
analysis = data.get("analysis", {})
summary = analysis.get("summary_comment", "").strip()
detail = analysis.get("detail_comment", "").strip()

# 줄바꿈 처리 함수
def format_comment(text):
    return "<br>".join(text.split("\n")) if text else "(분석 내용 없음)"

# --- 상단: 기업명 + 요약 코멘트 박스
st.caption("🧬 기업의 재무, 주가, 뉴스, 거시지표 등 다양한 정보를 기반으로 AI가 종합적으로 작성한 코멘트입니다.")
st.markdown(f"""
    <style>
    .company-title {{
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }}
    .comment-box {{
        background-color: #e9f0fc;
        padding: 1.1rem 1.4rem;
        border-radius: 10px;
        font-size: 0.9rem;
        line-height: 1.6rem;
        color: #333;
        margin-bottom: 1.5rem;
    }}
    summary {{
        cursor: pointer;
        font-weight: 600;
        margin-top: 1rem;
        color: #003366;
    }}
    </style>

    <div class="comment-box">
        💡 <strong>AI 코멘트:</strong><br>
        {format_comment(summary)}
        <details>
            <summary>🔎 상세 분석</summary>
            <div style="margin-top: 0.7rem;">
                {format_comment(detail)}
            </div>
        </details>
    </div>
""", unsafe_allow_html=True)

# ---------------------- 중단: 분석 탭 ----------------------
st.subheader("📈 Financial")

tab1, tab2, tab3, tab4 = st.tabs(["건전성평가", "재무요약", "주가", "경쟁사 비교"])

# 탭1: 레이더차트 
with tab1:
    st.markdown("""
    <span style="font-size:13px; color:gray;">
    📌 본 분석에서는 <b>재무 안정성과 직접적으로 관련된 핵심 지표를 선별</b>하여 기업의 건전성을 평가하였습니다.<br>
    <small>※ 2021년은 기준 연도로 제외되며, 2022년부터 레이더차트에 반영됩니다.<br>
    </span>
    """, unsafe_allow_html=True)

    # ① 데이터 병합 ------------------------------------------------------
    df_value = pd.DataFrame.from_dict(data.get("value_metrics", {}), orient="index")
    df_value.index = df_value.index.str[:4]

    df_ratio = pd.DataFrame(data.get("financial_ratios", []))
    df_ratio["year"] = pd.to_datetime(df_ratio["year"]).dt.year.astype(str)
    df_ratio_pivot = df_ratio.pivot(index="year", columns="metric", values="value")

    df_radar = (
        pd.concat([df_value, df_ratio_pivot], axis=1)
          .loc[lambda x: x.index.astype(int) <= 2024]
          .sort_index()
    )

    amount_vars = ["EBITDA", "FCFF"]
    good_ratio  = ["이자보상배율"]                 # ↑ 클수록 좋음
    bad_ratio   = ["부채비율", "자기자본비율"]     # ↓ 작을수록 좋음
    radar_fields = amount_vars + good_ratio + bad_ratio
    df_radar = df_radar[radar_fields]

    if df_radar.empty:
        st.warning("필수 재무 데이터가 없습니다.")
        st.stop()

    # ② 금액 지표 Base-100 지수 ------------------------------------------
    def to_index(series):
        base_idx = series[series > 0].index.min()
        if pd.isna(base_idx):
            return pd.Series(np.nan, index=series.index)
        base = series.at[base_idx]
        return series.div(base).mul(100)

    df_idx = df_radar.copy()
    df_idx[amount_vars] = df_radar[amount_vars].apply(to_index, axis=0)

    # --- 숫자+단위 포맷 함수 (단위 통합) ---
    def fmt_num(val, col_name, int_fmt="{:,.0f}", float_fmt="{:,.1f}"):
        try:
            if pd.isna(val):
                return "—"
            # 비율, 배율, 금액별 단위 구분
            if col_name in ["부채비율", "자기자본비율"]:
                return float_fmt.format(float(val)) + " %"
            elif col_name == "이자보상배율":
                return float_fmt.format(float(val)) + " 배"
            elif col_name in ["EBITDA", "FCFF"]:
                return int_fmt.format(float(val)) + " 억원"
            else:
                return int_fmt.format(float(val))
        except Exception:
            return str(val)

    # --- 1. 21년은 탭에서 제외 ---
    years = [y for y in df_radar.index if int(y) > 2021]
    year_tabs = st.tabs(years)

    # --- 2. EBITDA/FCFF 변화율(%) 계산만 레이더값용으로만 사용 ---
    def calc_pct_change(series):
        vals = series.values.astype(float)
        pct_changes = [np.nan]  # 첫 해는 None
        for i in range(1, len(vals)):
            prev, curr = vals[i-1], vals[i]
            if np.isnan(prev) or prev == 0 or np.isnan(curr):
                pct_changes.append(np.nan)
            else:
                pct = ((curr - prev) / abs(prev)) * 100
                pct = np.clip(pct, -100, 100)
                pct_changes.append(pct)
        return pd.Series(pct_changes, index=series.index)

    ebitda_chg = calc_pct_change(df_radar["EBITDA"])
    fcff_chg   = calc_pct_change(df_radar["FCFF"])

    # --- CSS 스타일 ---
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e3e7ed;
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #2A72E8, #4A90E2);
    }
    .cards-container {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 15px;
        margin: 20px 0;
    }
    @media (max-width: 1200px) {
        .cards-container { grid-template-columns: repeat(3, 1fr); }
    }
    @media (max-width: 768px) {
        .cards-container { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 480px) {
        .cards-container { grid-template-columns: 1fr; }
    }
    /* 📱 모바일 폰트 및 쉐도우 축소 */
    @media (max-width: 768px) {
        .metric-card {
            padding: 12px;
            box-shadow: none;
            min-height: 140px;
        }
        .metric-card div {
            font-size: 13px !important;
        }
        .tooltip-metric, .tooltip-status {
            display: none;
        }
    }    
    .explanation-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
    }
    .explanation-title {
        font-size: 16px;
        font-weight: 700;
        color: #495057;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .explanation-content {
        font-size: 14px;
        color: #6c757d;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

    for i, yr in enumerate(years):
        with year_tabs[i]:
            prev_yr = str(int(yr)-1)
            radar_vals, cards_data = [], []

            for col in radar_fields:
                raw = df_radar.at[yr, col]
                idx = df_idx.at[yr, col]

                # --- 레이더차트용 값만 증감률로 사용 ---
                if col == "EBITDA":
                    chg = ebitda_chg[yr]
                    v_plot = 0.5 if pd.isna(chg) else min(max((chg + 100)/200, 0), 1)
                elif col == "FCFF":
                    chg = fcff_chg[yr]
                    v_plot = 0.5 if pd.isna(chg) else min(max((chg + 100)/200, 0), 1)
                elif col == "이자보상배율":
                    if pd.isna(raw): v_plot = np.nan
                    elif raw >= 10: v_plot = 1.0
                    elif raw >= 3: v_plot = 0.8
                    elif raw >= 1: v_plot = 0.6
                    else: v_plot = 0.4
                elif col == "부채비율":
                    if pd.isna(raw): v_plot = np.nan
                    elif raw <= 100: v_plot = 1.0
                    elif raw <= 150: v_plot = 0.8
                    elif raw <= 200: v_plot = 0.6
                    else: v_plot = 0.4
                elif col == "자기자본비율":
                    if pd.isna(raw): v_plot = np.nan
                    elif raw >= 60: v_plot = 1.0
                    elif raw >= 50: v_plot = 0.8
                    elif raw >= 40: v_plot = 0.6
                    else: v_plot = 0.4
                else:
                    v_plot = np.nan
                radar_vals.append(0 if pd.isna(v_plot) else v_plot)

                # === 카드 데이터 생성 ===
                if prev_yr in df_radar.index and not pd.isna(df_radar.at[prev_yr, col]):
                    prev_val = df_radar.at[prev_yr, col]
                    if not pd.isna(raw) and not pd.isna(prev_val):
                        diff_val = raw - prev_val
                        diff_pct = (diff_val / abs(prev_val)) * 100 if prev_val != 0 else 0
                        if col in ["부채비율", "자기자본비율", "이자보상배율"]:
                            diff_str = f"{diff_val:+,.1f}"
                        else:
                            diff_str = f"{diff_val:+,.0f}"
                        pct_str = f"{diff_pct:+.1f}%"
                        if col in ["EBITDA", "FCFF"]:
                            diff_str_fmt = f"{diff_str} 억원"
                        elif col == "이자보상배율":
                            diff_str_fmt = f"{diff_str} 배"
                        elif col in ["부채비율", "자기자본비율"]:
                            diff_str_fmt = f"{diff_str} %"
                        else:
                            diff_str_fmt = diff_str
                        # 변화 상태 결정
                        if diff_pct > 0:
                            change_class = "positive"
                        elif diff_pct < 0:
                            change_class = "negative"
                        else:
                            change_class = "neutral"
                    else:
                        diff_str_fmt, pct_str = "—", "—"
                        change_class = "neutral"
                else:
                    diff_str_fmt, pct_str = "—", "—"
                    change_class = "neutral"

                # 상태 및 아이콘 결정
                if col == "부채비율":
                    if raw <= 50:
                        verdict, status_class, icon = "안정", "status-excellent", "🟢"
                    elif raw <= 100:
                        verdict, status_class, icon = "양호", "status-good", "🔵"
                    elif raw <= 150:
                        verdict, status_class, icon = "주의", "status-warning", "🟡"
                    else:
                        verdict, status_class, icon = "위험", "status-danger", "🔴"
                elif col == "이자보상배율":
                    if raw >= 10:
                        verdict, status_class, icon = "매우 우수", "status-excellent", "🟢"
                    elif raw >= 3:
                        verdict, status_class, icon = "우수", "status-good", "🔵"
                    elif raw >= 1:
                        verdict, status_class, icon = "경계", "status-warning", "🟡"
                    else:
                        verdict, status_class, icon = "위험", "status-danger", "🔴"
                elif col == "자기자본비율":
                    if raw >= 60:
                        verdict, status_class, icon = "탄탄", "status-excellent", "🟢"
                    elif raw >= 50:
                        verdict, status_class, icon = "양호", "status-good", "🔵"
                    elif raw >= 40:
                        verdict, status_class, icon = "보통", "status-warning", "🟡"
                    else:
                        verdict, status_class, icon = "취약", "status-danger", "🔴"
                else:
                    verdict, status_class, icon = "", "status-good", "💰"

                cards_data.append({
                    "지표": col,
                    "값": fmt_num(raw, col),
                    "변화값": diff_str_fmt,
                    "변화율": pct_str,
                    "변화_클래스": change_class,
                    "상태": verdict,
                    "상태_클래스": status_class,
                    "아이콘": icon
                })

            # --- 레이더차트 ---
            radar_vals_closed = radar_vals + [radar_vals[0]]
            radar_fields_closed = radar_fields + [radar_fields[0]]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=radar_vals_closed,
                theta=radar_fields_closed,
                mode='lines',
                line=dict(color='#2A72E8', width=2),
                fill='toself',
                fillcolor='rgba(110,180,250,0.18)',
                hoverinfo='skip',
                hovertemplate=None,
                showlegend=False
            ))
            fig.add_trace(go.Scatterpolar(
                r=[0.5]*len(radar_fields_closed),
                theta=radar_fields_closed,
                mode='lines',
                line=dict(color='#B0B0B0', width=1.3, dash='dot'),
                hoverinfo='skip',
                showlegend=False
            ))
            fig.update_layout(
                polar=dict(
                    bgcolor="white",
                    radialaxis=dict(
                        visible=True,
                        showticklabels=False,
                        range=[0,1],
                        gridcolor="#E3E7ED",
                        gridwidth=1.2,
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=13, family="NanumGothic, Malgun Gothic, Arial", color='#2c3e50'),
                        rotation=90,
                        direction="clockwise",
                    )
                ),
                margin=dict(l=40, r=40, t=20, b=20),
                font=dict(family="NanumGothic, Malgun Gothic, Arial", size=13),
                autosize=True,
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- 카드 형태 지표 표시 ---
            tooltip_map = {
                "EBITDA": "영업이익 + 감가상각비. 기업의 현금창출력을 의미합니다.",
                "FCFF": "기업이 영업활동 후 자유롭게 사용할 수 있는 현금흐름입니다.",
                "이자보상배율": "영업이익이 이자비용의 몇 배인지. 높을수록 안정적입니다.",
                "부채비율": "총부채/자기자본 비율. 낮을수록 재무안정성이 우수합니다.",
                "자기자본비율": "자기자본/총자본 비율. 높을수록 재무구조가 건전합니다.",
                "부채비율해석": "🟢 안정(≤50%) > 🔵 양호(≤100%) > 🟡 주의(≤150%) > 🔴 위험(>150%)",
                "이자보상배율해석": "🟢 매우 우수(≥10배) > 🔵 우수(≥3배) > 🟡 경계(≥1배) > 🔴 위험(<1배)",
                "자기자본비율해석": "🟢 탄탄(≥60%) > 🔵 양호(≥50%) > 🟡 보통(≥40%) > 🔴 취약(<40%)",
            }
            # ✅ 여기에 붙이세요 (카드 반복문 위나 아래)
            st.markdown("""
            <style>
            @media (max-width: 768px) {
                .metric-card {
                    margin-bottom: 16px !important;
                }
            }
            </style>
            """, unsafe_allow_html=True)
            cols = st.columns(5)
            for i, card in enumerate(cards_data):
                with cols[i]:
                    tooltip = tooltip_map.get(card["지표"], "")
                    is_growth_metric = card["지표"] in ["EBITDA", "FCFF"]
                    arrow = "▲" if card["변화_클래스"] == "positive" else "▼" if card["변화_클래스"] == "negative" else "■"

                    # 🔸 변화 블록
                    change_block = (
                        f'<div style="font-size:12px;font-weight:600;margin-bottom:6px;line-height:1.2;'
                        f'color:{"#27ae60" if card["변화_클래스"] == "positive" else "#e74c3c" if card["변화_클래스"] == "negative" else "#95a5a6"};">'
                        f'<span style="display:block;">{arrow}{card["변화값"]}</span>'
                        f'</div>'
                    )

                    # 🔸 해석 기준 텍스트 (툴팁용)
                    interpret_key = card["지표"] + "해석"
                    interpretation = tooltip_map.get(interpret_key, "")

                    # 🔸 하단 상태 또는 변화율 뱃지 + 해석툴팁
                    if is_growth_metric:
                        card_bottom = (
                            f'<div style="display:inline-block;padding:6px 12px;border-radius:20px;'
                            f'font-size:14px;font-weight:600;background:#f0f0f0;color:'
                            f'{"#27ae60" if card["변화_클래스"] == "positive" else "#e74c3c"};">'
                            f'{arrow} {card["변화율"]}</div>'
                        )
                    else:
                        color_bg = (
                            "#d4edda" if card["상태_클래스"] == "status-excellent" else
                            "#cce5ff" if card["상태_클래스"] == "status-good" else
                            "#fff3cd" if card["상태_클래스"] == "status-warning" else
                            "#f8d7da"
                        )
                        color_fg = (
                            "#155724" if card["상태_클래스"] == "status-excellent" else
                            "#004085" if card["상태_클래스"] == "status-good" else
                            "#856404" if card["상태_클래스"] == "status-warning" else
                            "#721c24"
                        )
                        card_bottom = (
                            f'<div style="display:flex;align-items:center;justify-content:space-between;'
                            f'padding:6px 12px;border-radius:20px;'
                            f'font-size:14px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;'
                            f'background:{color_bg};color:{color_fg};">'
                            f'<span>{card["상태"]}</span>'
                            f'<span class="tooltip-status">ⓘ<span class="tooltiptext">{interpretation}</span></span>'
                            f'</div>'
                        )

                    # 🔸 카드 구성
                    card_style = (
                        f'<div class="metric-card" style="background:linear-gradient(135deg,#ffffff 0%,#f8f9fa 100%);'
                        f'border:1px solid #e3e7ed;border-radius:12px;padding:12px;margin:-1px 0;'
                        f'box-shadow:0 2px 6px rgba(0,0,0,0.06);min-height:140px;display:flex;flex-direction:column;'
                        f'justify-content:space-between;position:relative;overflow:visible;line-height:1.2;">'
                        
                        f'<div style="content:\'\';position:absolute;top:0;left:0;right:0;height:4px;'
                        f'background:linear-gradient(90deg,#2A72E8,#4A90E2);"></div>'

                        f'<div style="font-size:16px;font-weight:700;color:#2c3e50;margin-bottom:4px;'
                        f'display:flex;align-items:center;gap:4px;white-space:nowrap;text-align:center;">'
                        f'{card["지표"]}'
                        f'<div class="tooltip-metric">ⓘ<span class="tooltiptext">{tooltip}</span></div>'
                        f'</div>'

                        f'<div style="font-size:16px;font-weight:800;color:#34495e;margin-bottom:-1px;white-space:nowrap">'
                        f'{card["값"]}</div>'

                        f'{change_block}'
                        f'{card_bottom}'
                        f'</div>'
                    )
                        
                    st.markdown(card_style, unsafe_allow_html=True)
            
            # --- 툴팁용 CSS (분리된 스타일) ---
            st.markdown("""
            <style>
            .tooltip-metric {
            position: relative;
            display: inline-block;
            cursor: help;
            font-size: 0.7rem;
            top: -5px;
            margin-left: -3px;
            line-height: 1;
            }
            .tooltip-metric .tooltiptext {
            visibility: hidden;
            width: fit-content;
            background-color: #333;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 8px;
            position: absolute;
            z-index: 1;
            bottom: 150%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 11px;
            white-space: nowrap;
            }
            .tooltip-metric:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
            }

            .tooltip-status {
            position: relative;
            display: inline-block;
            cursor: help;
            font-size: 0.65rem;
            margin-left: 6px;
            line-height: 1;
            }
            .tooltip-status .tooltiptext {
            visibility: hidden;
            width: fit-content;
            background-color: #222;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 6px;
            position: absolute;
            z-index: 1;
            top: 200%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 11px;
            white-space: nowrap;
            }
            .tooltip-status:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
            }
            </style>
            """, unsafe_allow_html=True)

    st.markdown(
        """
        <div style="margin-top:12px; font-size:11px; color:gray; text-align:right;">
            ※ 위 기준은 일반적으로 사용되는 재무 안정성 평가 기준을 참고하여 작성되었습니다.
        </div>
        """,
        unsafe_allow_html=True
    )
# --- 탭2: 재무 분석
with tab2:
    st.caption("📌 기업의 핵심 재무지표 중 수익성·안정성·성장성을 대표하는 항목들을 비교할 수 있도록 구성했습니다.")
    selected_metrics = ["부채비율", "EPS증가율", "이자보상배율", "영업이익증가율", "매출액증가율"]

    # 데이터 준비
    df_ratio = pd.DataFrame(data["financial_ratios"])
    df_ratio['year'] = pd.to_datetime(df_ratio['year']).dt.year.astype(str)
    df_filtered = df_ratio[df_ratio['metric'].isin(selected_metrics)]

    # 피벗: 지표별 연도별 값 구조로
    df_pivot = df_filtered.pivot(index='year', columns='metric', values='value').sort_index()

    # 그래프 생성
    fig = go.Figure()

    for metric in selected_metrics:
        fig.add_trace(go.Bar(
            x=df_pivot.index,             # 연도
            y=df_pivot[metric],           # 지표값
            name=metric,                   # 범례
            hovertemplate='%{x}년 ' + metric + ': %{y:.1f}%' + '<extra></extra>'
        ))

    fig.update_layout(
        barmode='group',  # 그룹 바차트
        title="연도별 주요 재무비율 비교",
        xaxis_title="연도",
        yaxis_title="값",
        legend_title="지표명",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

# --- 탭3: 주가 분석 (v1: rangeselector만 사용)
with tab3:
    today = date(2025, 7, 2)
    default_start = date(2021, 1, 1)
    range_3m = today - timedelta(days=90)
    range_6m = today - timedelta(days=180)
    range_1y = today - timedelta(days=365)
    range_2y = today - timedelta(days=730)
    range_all = date(2021, 1, 1)

    st.caption("📌 주가 데이터는 2025년 6월 29일까지 수집된 기준입니다.")
    try:
        # ✅ 전체 주가 데이터 호출
        response = requests.get(f"{FASTAPI_URL}/stocks/{company_name}", params={"frequency": "week"})
        response.raise_for_status()
        stock_df = pd.DataFrame(response.json())

        # 한글 컬럼명으로 변경
        stock_df.rename(columns={
            "open": "시작", "high": "고가", "low": "저가", "close": "종가"
        }, inplace=True)

        if stock_df.empty:
            st.warning("해당 기업의 주가 데이터가 없습니다.")
        else:
            stock_df["date"] = pd.to_datetime(stock_df["date"])
            stock_df["날짜"] = stock_df["date"].dt.strftime("%Y-%m-%d")
            x_range = [default_start, today]

            # 등락률 계산 및 이모지 부여
            stock_df["등락률"] = stock_df["종가"] / stock_df["시작"] - 1
            stock_df["등락기호"] = stock_df["등락률"].apply(lambda x: "▲" if x > 0 else ("▼" if x < 0 else "-"))
            stock_df["종가_표시"] = stock_df.apply(lambda row: f"{int(row['종가']):,}원 {row['등락기호']}", axis=1)

            # 상승/하락 색상
            colors = np.where(stock_df["종가"] >= stock_df["시작"], "#FF3B30", "#007AFF")

            # 📊 차트 생성
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3],
                subplot_titles=(f"{company_name} 주가", "거래량")
            )

            # customdata로 정보 전달
            custom_data = np.stack([
                stock_df["날짜"], stock_df["시작"],
                stock_df["고가"], stock_df["저가"],
                stock_df["종가_표시"]
            ], axis=-1)

            # 📌 캔들 차트 추가
            fig.add_trace(go.Candlestick(
                x=stock_df["date"],
                open=stock_df["시작"],
                high=stock_df["고가"],
                low=stock_df["저가"],
                close=stock_df["종가"],
                increasing_line_color="#FF3B30",
                decreasing_line_color="#007AFF",
                name="",
                customdata=custom_data,
                text=[
                    f"날짜: {d[0]}<br>시가: {int(d[1]):,}원<br>고가: {int(d[2]):,}원<br>저가: {int(d[3]):,}원<br>종가: {d[4]}"
                    for d in custom_data
                ],
                hoverinfo="text"
            ), row=1, col=1)

            # 📌 거래량 바 차트 추가
            fig.add_trace(go.Bar(
                x=stock_df["date"],
                y=stock_df["volume"],
                marker_color=colors,
                name="거래량",
                opacity=0.6
            ), row=2, col=1)

            # --- Layout 설정
            fig.update_layout(
                autosize=True,
                height=600,
                dragmode='pan',
                hovermode="x unified",
                showlegend=False,
                margin=dict(t=50, b=40),
                xaxis=dict(
                    range=x_range,
                    autorange=False,
                    rangeslider=dict(visible=False),
                    rangeselector=dict(
                        buttons=list([
                            dict(count=3, label="3M", step="month", stepmode="backward"),
                            dict(count=6, label="6M", step="month", stepmode="backward"),
                            dict(count=1, label="1Y", step="year", stepmode="backward"),
                            dict(count=2, label="2Y", step="year", stepmode="backward"),
                        ]),
                        x=0,
                        y=1.15,
                        xanchor="left",
                        bgcolor="rgba(240,240,240,0.7)",
                        activecolor="#FF3B30",
                        font=dict(size=12)
                    )
                ),
                xaxis2=dict(range=x_range, autorange=False)
            )

            fig.update_xaxes(fixedrange=False)
            fig.update_yaxes(
                fixedrange=False,
                tickformat=",d"
            )

            # 📌 렌더링
            fig_html = fig.to_html(config={"scrollZoom": True})
            html(fig_html, height=650)

    except Exception as e:
        st.warning(f"데이터 로딩 오류: {e}")

# --- 탭4: 경쟁사 비교 ---
with tab4:
    st.caption("📌 경쟁사는 동일 업종 내에서 시가총액 규모가 유사한 기업들을 기준으로 선정되었습니다.")
    competitors = data.get("competitors", {})
    company_name = data["invest_info"]["stock_name"]

    try:
        company_metrics = {
            "시가총액": float(data["market_cap"]["시총"]),
            "매출액": float(data["invest_comp"]["매출액"]),
            "영업이익": float(data["invest_comp"]["영업이익"]),
            "ROE": float(data["invest_comp"]["ROE"])
        }
    except Exception as e:
        st.error(f"자사 재무 데이터를 불러오지 못했습니다: {e}")
        st.stop()

    if not competitors or not competitors.get("기업명"):
        st.warning("경쟁사 데이터를 불러올 수 없습니다.")
    else:
        # ROE 값 float 변환
        def safe_float(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return None

        competitors["ROE"] = [safe_float(r) for r in competitors.get("ROE", [])]

        # 데이터프레임 생성
        df_comp = pd.DataFrame({
            "시가총액": [company_metrics["시가총액"]] + competitors.get("시가총액", []),
            "매출액": [company_metrics["매출액"]] + competitors.get("매출액", []),
            "영업이익": [company_metrics["영업이익"]] + competitors.get("영업이익", []),
            "ROE": [company_metrics["ROE"]] + competitors["ROE"]
        }, index=[company_name] + competitors.get("기업명", []))

        df_comp = df_comp.apply(pd.to_numeric, errors='coerce')

        selected_metric = st.selectbox("비교할 항목 선택", df_comp.columns.tolist())
        df_sorted = df_comp.sort_values(by=selected_metric, ascending=False)

        # ✅ 차트 라벨
        if selected_metric == "ROE":
            text_labels = [f"{v:.1f}%" if pd.notnull(v) else "N/A" for v in df_sorted[selected_metric]]
        else:
            text_labels = [format_korean_number_for_dashboard(selected_metric, v) for v in df_sorted[selected_metric]]

        fig = go.Figure(data=[go.Bar(
            x=df_sorted.index,
            y=df_sorted[selected_metric],
            text=text_labels,
            textposition='auto',
            marker_color='lightskyblue'
        )])
        fig.update_layout(
            title=f"📊 {selected_metric} 기준 자사 vs 경쟁사 비교",
            yaxis_title=selected_metric
        )
        st.plotly_chart(fig, use_container_width=True)

        # 표 스타일 포맷 함수
        def get_format_func(col):
            if col == "ROE":
                return lambda v: f"{v:.1f}%" if pd.notnull(v) else "N/A"
            else:
                return lambda v: format_korean_number_for_dashboard(col, v)

        st.dataframe(
            df_sorted.style.format({col: get_format_func(col) for col in df_sorted.columns}),
            use_container_width=True
        )

# --- 하단: 뉴스 탭 ----------------------
st.markdown("")
st.markdown("")
st.subheader("📰 뉴스 분석")

news_tab1, news_tab2, news_tab3 = st.tabs(["키워드", "분석", "뉴스목록"])

# --- 공통 데이터프레임 생성
df_news = pd.DataFrame(data.get("news", []))

# --- 키워드 워드클라우드

with news_tab1:
    st.subheader("📌 뉴스 키워드 워드클라우드")

    df_news = pd.DataFrame(data.get("news", []))

    if not df_news.empty and "news_name" in df_news.columns:

        query = data.get("invest_info", {}).get("stock_name", "")

        # 🔹 한글 매핑
        aspect_map = {
            'financial': '재무',
            'esg': 'ESG',
            'investment_ma': '투자·인수합병',
            'risk_issue': '위험이슈',
            'strategy': '전략',
            'product_service': '제품·서비스',
            'general': '일반',
            'partnership': '제휴·협력',
            'economy': '경제'
        }
        df_news["aspect_kor"] = df_news["news_aspect"].map(aspect_map).fillna("기타")

        # 🔹 탭 구성
        category_list = df_news["aspect_kor"].dropna().unique().tolist()
        tab_labels = ["전체"] + category_list
        tabs = st.tabs(tab_labels)

        # 🔹 공통: 감정 색상
        sentiment_colors = {
            "positive": "dodgerblue",
            "negative": "tomato",
            "neutral": "lightgray"
        }
        
        def color_func_factory(word_sentiment):
            def color_func(word, *args, **kwargs):
                sentiment = word_sentiment.get(word, "neutral")
                return sentiment_colors.get(sentiment, "gray")
            return color_func
        st.markdown("""
        <div style="text-align: right; margin-top: -0.5rem;">
            <p style="color: #6B7280; font-size: 0.9rem; margin: 0;">
                <span style="color:dodgerblue;">🔵 긍정</span> <span style="color:Tomato;">🔴 부정</span> <span style="color:lightgray;">⚪ 중립</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        # 🔸 [전체] 탭
        with tabs[0]:

            word_sentiment_pairs = []

            for _, row in df_news.iterrows():
                sentiment = row["overall_sentiment"]
                title = str(row["news_name"])
                words = re.findall(r'\b[가-힣A-Za-z0-9]{2,}\b', title)
                for word in words:
                    if word != query:
                        word_sentiment_pairs.append((word, sentiment))

            if word_sentiment_pairs:
                df_sent = pd.DataFrame(word_sentiment_pairs, columns=["word", "sentiment"])
                word_counts = df_sent["word"].value_counts().to_dict()
                word_sentiment = df_sent.groupby("word")["sentiment"].agg(lambda x: x.value_counts().idxmax()).to_dict()

                wc = WordCloud(
                    font_path=font_path,
                    width=600,
                    height=400,
                    background_color="white",
                    color_func=color_func_factory(word_sentiment),
                    max_words=100
                ).generate_from_frequencies(word_counts)

                fig, ax = plt.subplots(figsize=(9,4.5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig,use_container_width=False)
            else:
                st.warning("전체 뉴스에서 단어를 추출할 수 없습니다.")

        # 🔸 카테고리별 탭
        for i, cat_kor in enumerate(category_list):
            with tabs[i + 1]:

                sub_df = df_news[df_news["aspect_kor"] == cat_kor]

                word_sentiment_pairs = []

                for _, row in sub_df.iterrows():
                    sentiment = row["overall_sentiment"]
                    title = str(row["news_name"])
                    words = re.findall(r'\b[가-힣A-Za-z0-9]{2,}\b', title)
                    for word in words:
                        if word != query:
                            word_sentiment_pairs.append((word, sentiment))

                if word_sentiment_pairs:
                    df_sent = pd.DataFrame(word_sentiment_pairs, columns=["word", "sentiment"])
                    word_counts = df_sent["word"].value_counts().to_dict()
                    word_sentiment = df_sent.groupby("word")["sentiment"].agg(lambda x: x.value_counts().idxmax()).to_dict()

                    wc = WordCloud(
                        font_path=font_path,
                        width=800,
                        height=400,
                        background_color="white",
                        color_func=color_func_factory(word_sentiment),
                        max_words=50
                    ).generate_from_frequencies(word_counts)

                    fig, ax = plt.subplots(figsize=(9,4.5))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig,use_container_width=False)
                else:
                    st.warning(f"{cat_kor} 뉴스에서 단어를 추출할 수 없습니다.")

    else:
        st.warning("뉴스 제목 데이터가 없습니다.")

# --- 뉴스 감정분석 시각화
with news_tab2:
    st.subheader("📰 최근 뉴스 카테고리별 감정 분포")

    if not df_news.empty:
        # ① 한글 매핑 사전 정의
        aspect_map = {
            'financial': '재무',
            'esg': 'ESG',
            'investment_ma': '투자·인수합병',
            'risk_issue': '위험이슈',
            'strategy': '전략',
            'product_service': '제품·서비스',
            'general': '일반',
            'partnership': '제휴·협력',
            'economy': '경제'
        }
        sentiment_map = {
            'positive': '긍정',
            'neutral': '중립',
            'negative': '부정'
        }
        # ② news_aspect를 한글로 변환한 컬럼 추가
        df_news['aspect_kor'] = df_news['news_aspect'].map(aspect_map).fillna('기타')
        df_news['sentiment_kor'] = df_news['overall_sentiment'].map(sentiment_map).fillna('기타')

        # ③ 감성 피벗 테이블 생성 (한글 기준)
        pivot = df_news.pivot_table(
            index='aspect_kor',
            columns='sentiment_kor',
            aggfunc='size',
            fill_value=0)

        # ④ Plotly로 바 차트 생성
        fig = go.Figure()

        # 색상 팔레트 지정 (Pastel1 색상)
        colors = {'부정': 'tomato', '중립': 'lightgray', '긍정': "dodgerblue"}  # 부정, 중립, 긍정

        # 감성별 바 차트 추가
        for sentiment in ['부정', '중립', '긍정']:
            if sentiment in pivot.columns:
                fig.add_trace(go.Bar(
                    x=pivot.index,
                    y=pivot[sentiment],
                    name=sentiment,
                    marker=dict(color=colors[sentiment]),
                    hovertemplate='(%{x}, %{y}건, ' + sentiment + ')<extra></extra>'
                ))

        # ⑤ 축 설정
        fig.update_layout(
            barmode='stack',
            yaxis_title="기사 수",
            xaxis=dict(tickmode='array', tickvals=list(range(len(pivot.index))), ticktext=pivot.index),
            font=dict(family="NanumGothic", size=12),
            showlegend=False,
            template="plotly_white"
        )

        # ⑥ 출력
        st.plotly_chart(fig)
    else:
        st.info("뉴스 감정 데이터를 불러올 수 없습니다.")

# --- 원문 뉴스 리스트
with news_tab3:
    st.subheader("뉴스 요약 리스트")

    if not df_news.empty:
        news_df = df_news.copy()
        news_df = news_df.drop_duplicates(subset=['news_name'], keep='first')
    
        # 뉴스 카테고리 한글 변환
        aspect_map = {
            'financial': '재무',
            'esg': 'ESG',
            'investment_ma': '투자·인수합병',
            'risk_issue': '리스크',
            'strategy': '전략',
            'product_service': '제품·서비스',
            'general': '일반',
            'partnership': '제휴·협력',
            'economy': '경제'
        }
        news_df['category'] = news_df['news_aspect'].map(aspect_map).fillna('기타')
        
        # 감정 한글 매핑
        sentiment_map = {
            'positive': '긍정',
            'negative': '부정',
            'neutral': '중립'
        }
        news_df['overall_sentiment'] = news_df['overall_sentiment'].map(sentiment_map).fillna('-')
        
        # 날짜 형식 변환 및 정렬 (최신순)
        news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
        news_df = news_df.dropna(subset=['date'])
        news_df = news_df.sort_values(by='date', ascending=False).reset_index(drop=True)
        news_df['date'] = news_df['date'].dt.strftime('%Y-%m-%d')
        st.markdown("### 🔍 필터")
        col_filter1, col_filter2 = st.columns(2)

        category_options = ['전체'] + sorted(news_df['category'].dropna().unique().tolist())
        sentiment_options = ['전체'] + sorted(news_df['overall_sentiment'].dropna().unique().tolist())

        with col_filter1:
            selected_category = st.selectbox("카테고리 선택", options=category_options)
        with col_filter2:
            selected_sentiment = st.selectbox("감정 선택", options=sentiment_options)

        if selected_category != '전체':
            news_df = news_df[news_df['category'] == selected_category]

        if selected_sentiment != '전체':
            news_df = news_df[news_df['overall_sentiment'] == selected_sentiment]
        # 페이지네이션 설정
        news_per_page = 4
        total_pages = max(1, (len(news_df) - 1) // news_per_page + 1)

        # 세션 상태 초기화 및 데이터 변경 감지
        if "news_page" not in st.session_state:
            st.session_state.news_page = 1
        
        # 데이터 변경 감지를 위한 해시값 생성
        current_data_hash = hash(tuple(news_df['news_name'].tolist()))
        
        if "news_data_hash" not in st.session_state:
            st.session_state.news_data_hash = current_data_hash
            st.session_state.news_page = 1
        elif st.session_state.news_data_hash != current_data_hash:
            # 데이터가 변경되었으면 첫 페이지로 리셋
            st.session_state.news_data_hash = current_data_hash
            st.session_state.news_page = 1

        # 유효 범위 보정
        st.session_state.news_page = max(1, min(st.session_state.news_page, total_pages))

        start = (st.session_state.news_page - 1) * news_per_page
        end = start + news_per_page
        page_news = news_df.iloc[start:end]

        # 디버깅 정보 (배포 시 제거)
        #st.write(f"Debug: 총 {len(news_df)}개 뉴스, 현재 페이지: {st.session_state.news_page}, 표시 범위: {start}-{end}")
        #st.write(f"Debug: 실제 표시되는 뉴스 개수: {len(page_news)}")
        #if len(page_news) > 0:
        #    st.write(f"Debug: 첫 번째 뉴스 제목: {page_news.iloc[0]['news_name'][:30]}...")

        # HTML 테이블 렌더링
        html = """
        <style>
        table {
            width: 100%;
            table-layout: fixed;
            text-align: center;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        th, td {
            padding: 0.6em;
            text-align: center;
            border-bottom: 1px solid #444;
            white-space: nowrap;
        }
        td {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        th:nth-child(1), td:nth-child(1) {
            width: 60%;
            text-align: left;
        }
        th:nth-child(2), td:nth-child(2) {
            width: 15%;
        }
        th:nth-child(3), td:nth-child(3) {
            width: 10%;
        }
        th:nth-child(4), td:nth-child(4) {
            width: 17%;
        }
        a {
            text-decoration: none;
            color: #1a73e8;
            font-weight: bold;
        }
        </style>
        <table>
        <thead>
            <tr>
                <th>제목</th>
                <th>카테고리</th>
                <th>감정</th>
                <th>날짜</th>
            </tr>
        </thead>
        <tbody>
        """

        for _, row in page_news.iterrows():
            title = row.get('news_name', '-')
            category = row.get('category', '-')
            overall_sentiment = row.get('overall_sentiment', '-')
            date = row.get('date', '-')
            link = row.get('link', '#')
            title_link = f'<a href="{link}" target="_blank">{title}</a>'
            html += f"<tr><td>{title_link}</td><td>{category}</td><td>{overall_sentiment}</td><td>{date}</td></tr>"

        html += "</tbody></table>"
        st.markdown(html, unsafe_allow_html=True)

        # 페이지네이션 하단 버튼 (대안 방법)
        col1, col2, col3 = st.columns([1, 6, 1])
        
        # 버튼 상태를 직접 확인
        prev_clicked = False
        next_clicked = False
        
        with col1:
            if st.button("◀ 이전", key="prev_news"):
                prev_clicked = True
        with col3:
            if st.button("다음 ▶", key="next_news"):
                next_clicked = True
                
        # 버튼 클릭 처리
        if prev_clicked and st.session_state.news_page > 1:
            st.session_state.news_page -= 1
            st.rerun()
        elif next_clicked and st.session_state.news_page < total_pages:
            st.session_state.news_page += 1
            st.rerun()
        with col2:
            st.markdown(
                f"<div style='text-align:center; font-size: 1rem;'> {st.session_state.news_page} / {total_pages}</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("뉴스 리스트를 불러올 수 없습니다.")

#LLM------------------------------------------
import logging
import openai
# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------- 디바이스 확인 및 설정 ----------------------
def setup_device():
    """디바이스 설정 및 정보 출력"""
    print("=" * 50)
    print("🔧 시스템 정보")
    print("=" * 50)
    print(f"PyTorch 버전: {torch.__version__}")
    
    # ROCm 환경 확인
    hip_version = getattr(torch.version, 'hip', None)
    if hip_version:
        print(f"HIP 버전: {hip_version}")
    else:
        print("HIP 지원: ❌ 없음")
    
    # GPU 사용 가능 여부 확인
    if torch.cuda.is_available():
        print("GPU 사용 가능: ✅")
        device_count = torch.cuda.device_count()
        print(f"사용 가능한 GPU 수: {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"디바이스 {i}: {device_name}")
            
            # GPU 메모리 정보 (ROCm에서도 동작)
            try:
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  메모리 사용량: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
            except:
                print("  메모리 정보 확인 불가")
        
        device = "cuda"
    else:
        print("GPU 사용 가능: ❌ (CPU 사용)")
        device = "cpu"
    
    print("=" * 50)
    return device

# 디바이스 설정
DEVICE = setup_device()

# ---------------------- API 키 및 설정 ----------------------
try:
    # 🔐 API Key
    PINECONE_TEAM_API_KEY = st.secrets["team_pinecone"]["api_key"]
    PINECONE_MY_API_KEY = st.secrets["my_pinecone"]["api_key"]
    OPENAI_API_KEY = st.secrets["my_pinecone"]["openai_api_key"]
    GEMINI_API_KEY = st.secrets["team_pinecone"]["gemini_api_key"]
    
    # 📌 인덱스 이름
    TEAM_INDEX_NAME = st.secrets["team_pinecone"]["index_name"]
    COMPANY_INDEX_NAME = st.secrets["my_pinecone"]["index_company"]
    META_INDEX_NAME = st.secrets["my_pinecone"]["index_meta"]
    
    logger.info("✅ API 키 및 설정 로드 완료")
    
except Exception as e:
    logger.error(f"❌ 설정 로드 실패: {e}")
    st.error("설정 파일을 확인해주세요.")
    st.stop()

# ---------------------- Gemini 초기화 ----------------------
try:
    GEMINI_MODEL = "gemini-2.0-flash-exp"
    genai.configure(api_key=GEMINI_API_KEY)
    gemini = genai.GenerativeModel(GEMINI_MODEL)
    logger.info("✅ Gemini 초기화 완료")
except Exception as e:
    logger.error(f"❌ Gemini 초기화 실패: {e}")
    st.error("Gemini 초기화에 실패했습니다.")

# ---------------------- Pinecone 초기화 ----------------------
try:
    pc_team = Pinecone(api_key=PINECONE_TEAM_API_KEY)
    pc_my = Pinecone(api_key=PINECONE_MY_API_KEY)
    logger.info("✅ Pinecone 클라이언트 초기화 완료")
except Exception as e:
    logger.error(f"❌ Pinecone 초기화 실패: {e}")
    st.error("Pinecone 초기화에 실패했습니다.")

# ---------------------- 임베딩 모델 초기화 ----------------------
@st.cache_resource
def load_embedding_model():
    """임베딩 모델 로드 (개선된 버전)"""
    try:
        logger.info("🔄 임베딩 모델 로드 중...")
        
        # GPU 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 모델 로드 옵션들 (우선순위 순)
        model_options = [
            {
                "model_name": "intfloat/multilingual-e5-large",
                "model_kwargs": {
                    "device": DEVICE,
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32,
                },
                "encode_kwargs": {"normalize_embeddings": True, "batch_size": 32}
            },
            {
                "model_name": "intfloat/multilingual-e5-base",  # 더 작은 모델
                "model_kwargs": {
                    "device": DEVICE,
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32,
                },
                "encode_kwargs": {"normalize_embeddings": True, "batch_size": 64}
            },
            {
                "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "model_kwargs": {
                    "device": DEVICE,
                    "trust_remote_code": True,
                },
                "encode_kwargs": {"normalize_embeddings": True, "batch_size": 128}
            }
        ]
        
        # 각 모델 옵션을 순차적으로 시도
        for i, options in enumerate(model_options):
            try:
                logger.info(f"📥 모델 로드 시도 {i+1}/3: {options['model_name']}")
                
                embedding_model = HuggingFaceEmbeddings(**options)
                
                # 테스트 임베딩 생성
                test_embedding = embedding_model.embed_query("테스트")
                
                if len(test_embedding) > 100:  # 임베딩 차원 확인
                    logger.info(f"✅ 임베딩 모델 로드 성공: {options['model_name']}")
                    logger.info(f"   임베딩 차원: {len(test_embedding)}")
                    logger.info(f"   디바이스: {DEVICE}")
                    return embedding_model
                else:
                    logger.warning(f"⚠️ 임베딩 차원이 너무 작음: {len(test_embedding)}")
                    
            except Exception as e:
                logger.warning(f"⚠️ 모델 로드 실패: {options['model_name']} - {str(e)}")
                # GPU 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        # 모든 모델 로드 실패
        logger.error("❌ 모든 임베딩 모델 로드 실패")
        
        # 마지막 시도: CPU로 강제 설정
        if DEVICE == "cuda":
            logger.info("🔄 CPU로 모델 로드 재시도...")
            try:
                embedding_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    model_kwargs={"device": "cpu", "trust_remote_code": True},
                    encode_kwargs={"normalize_embeddings": True}
                )
                
                test_embedding = embedding_model.embed_query("테스트")
                if len(test_embedding) > 100:
                    logger.info("✅ CPU 모드로 임베딩 모델 로드 성공")
                    return embedding_model
                    
            except Exception as e:
                logger.error(f"❌ CPU 모드 로드도 실패: {e}")
        
        return None
        
    except Exception as e:
        logger.error(f"❌ 임베딩 모델 로드 중 예상치 못한 오류: {e}")
        return None

# ---------------------- 벡터스토어 초기화 안전화 ----------------------
@st.cache_resource
def initialize_vectorstores():
    """벡터스토어 초기화 (안전화된 버전)"""
    try:
        # 임베딩 모델 재로드
        embedding_e5 = load_embedding_model()
        
        if embedding_e5 is None:
            logger.error("❌ 임베딩 모델이 로드되지 않았습니다.")
            return None, None, None
        
        logger.info("🔄 벡터스토어 초기화 시작...")
        
        vectorstore_news = None
        vectorstore_company = None
        vectorstore_meta = None
        
        # 뉴스 벡터스토어
        try:
            vectorstore_news = PineconeVectorStore(
                index=pc_team.Index(TEAM_INDEX_NAME),
                embedding=embedding_e5,
                text_key="summary",
                namespace="news-ns"
            )
            logger.info("✅ 뉴스 벡터스토어 초기화 성공")
        except Exception as e:
            logger.error(f"❌ 뉴스 벡터스토어 초기화 실패: {e}")
        
        # 회사 벡터스토어
        try:
            vectorstore_company = PineconeVectorStore(
                index=pc_my.Index(COMPANY_INDEX_NAME),
                embedding=embedding_e5,
                text_key="summary_comment"
            )
            logger.info("✅ 회사 벡터스토어 초기화 성공")
        except Exception as e:
            logger.error(f"❌ 회사 벡터스토어 초기화 실패: {e}")
        
        # 메타 벡터스토어
        try:
            vectorstore_meta = PineconeVectorStore(
                index=pc_my.Index(META_INDEX_NAME),
                embedding=embedding_e5,
                text_key="description"
            )
            logger.info("✅ 메타 벡터스토어 초기화 성공")
        except Exception as e:
            logger.error(f"❌ 메타 벡터스토어 초기화 실패: {e}")
        
        # 성공한 벡터스토어 개수 확인
        success_count = sum([
            vectorstore_news is not None,
            vectorstore_company is not None,
            vectorstore_meta is not None
        ])
        
        logger.info(f"✅ 벡터스토어 초기화 완료 ({success_count}/3 성공)")
        return vectorstore_news, vectorstore_company, vectorstore_meta
        
    except Exception as e:
        logger.error(f"❌ 벡터스토어 초기화 실패: {e}")
        return None, None, None

# ---------------------- Retriever 안전한 초기화 ----------------------
def initialize_retrievers():
    """Retriever 안전한 초기화"""
    try:
        vectorstore_news, vectorstore_company, vectorstore_meta = initialize_vectorstores()
        if not all([vectorstore_news, vectorstore_company, vectorstore_meta]):
            logger.error("벡터스토어가 초기화되지 않았습니다.")
            return None, None, None
            
        retriever_news = vectorstore_news.as_retriever(
            search_kwargs={"k": 5, "score_threshold": 0.7}
        )
        retriever_company = vectorstore_company.as_retriever(
            search_kwargs={"k": 5, "score_threshold": 0.7}
        )
        retriever_meta = vectorstore_meta.as_retriever(
            search_kwargs={"k": 3, "score_threshold": 0.7}
        )
        
        logger.info("✅ Retriever 설정 완료")
        return retriever_news, retriever_company, retriever_meta
        
    except Exception as e:
        logger.error(f"❌ Retriever 초기화 실패: {e}")
        return None, None, None

# Retriever 초기화 (전역 변수로 설정)
retriever_news, retriever_company, retriever_meta = initialize_retrievers()

# ---------------------- LLM 설정 ----------------------
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.3
MAX_TOKENS = 2000

try:
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
        max_tokens=MAX_TOKENS,
        streaming=True  # 스트리밍 활성화
    )
    logger.info("✅ GPT LLM 초기화 완료")
except Exception as e:
    logger.error(f"❌ LLM 초기화 실패: {e}")
    st.error("LLM 초기화에 실패했습니다.")
    llm = None

# ---------------------- OpenAI 클라이언트 초기화 ----------------------
try:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    logger.info("✅ OpenAI 클라이언트 초기화 완료")
except Exception as e:
    logger.error(f"❌ OpenAI 클라이언트 초기화 실패: {e}")
    openai_client = None

# ---------------------- QA 체인 안전한 초기화 ----------------------
def initialize_qa_chains():
    """QA 체인 안전한 초기화"""
    qa_chain_news = None
    qa_chain_meta = None
    
    try:
        if llm is None:
            logger.error("LLM이 초기화되지 않았습니다.")
            return None, None
            
        # 뉴스용 QA 체인
        if retriever_news is not None:
            qa_chain_news = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever_news,
                return_source_documents=True
            )
            logger.info("✅ 뉴스 QA 체인 초기화 완료")
        else:
            logger.warning("⚠️ 뉴스 retriever가 없어 QA 체인을 생성할 수 없습니다.")
        
        # 메타 분석용 QA 체인
        if retriever_meta is not None:
            qa_chain_meta = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever_meta,
                return_source_documents=True
            )
            logger.info("✅ 메타 QA 체인 초기화 완료")
        else:
            logger.warning("⚠️ 메타 retriever가 없어 QA 체인을 생성할 수 없습니다.")
            
        return qa_chain_news, qa_chain_meta
        
    except Exception as e:
        logger.error(f"❌ QA 체인 초기화 실패: {e}")
        return None, None

# QA 체인 초기화
qa_chain_news, qa_chain_meta = initialize_qa_chains()

# ---------------------- 유틸리티 함수 ----------------------
def clear_gpu_cache():
    """GPU 캐시 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("🧹 GPU 캐시 정리 완료")

def get_gpu_memory_info():
    """GPU 메모리 사용량 정보 반환"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU 메모리: {allocated:.2f}GB / {reserved:.2f}GB"
    return "GPU 사용 불가"

def get_system_status():
    """시스템 상태 확인"""
    # 임베딩 모델 및 벡터스토어는 함수 호출로 가져옴 (캐시 활용)
    embedding_e5 = load_embedding_model()
    vectorstore_news, vectorstore_company, vectorstore_meta = initialize_vectorstores()
    status = {
        "device": DEVICE,
        "embedding_model": embedding_e5 is not None,
        "vectorstore_news": vectorstore_news is not None,
        "vectorstore_company": vectorstore_company is not None,
        "vectorstore_meta": vectorstore_meta is not None,
        "retriever_news": retriever_news is not None,
        "retriever_company": retriever_company is not None,
        "retriever_meta": retriever_meta is not None,
        "llm": llm is not None,
        "openai_client": openai_client is not None,
        "qa_chain_news": qa_chain_news is not None,
        "qa_chain_meta": qa_chain_meta is not None,
    }
    return status

# ---------------------- 안전한 사용 함수들 ----------------------
def safe_retrieve(retriever, query, fallback_text="정보를 찾을 수 없습니다."):
    """안전한 retriever 사용"""
    try:
        if retriever is None:
            logger.warning(f"Retriever가 None입니다. 쿼리: {query}")
            return [{"page_content": fallback_text, "metadata": {}}]
        
        results = retriever.invoke(query)
        if not results:
            logger.info(f"검색 결과가 없습니다. 쿼리: {query}")
            return [{"page_content": fallback_text, "metadata": {}}]
        
        return results
    except Exception as e:
        logger.error(f"검색 중 오류 발생: {e}")
        return [{"page_content": fallback_text, "metadata": {}}]

def safe_qa_query(qa_chain, query, fallback_text="답변을 생성할 수 없습니다."):
    """안전한 QA 체인 사용"""
    try:
        if qa_chain is None:
            logger.warning(f"QA 체인이 None입니다. 쿼리: {query}")
            return {"result": fallback_text, "source_documents": []}
        
        result = qa_chain.invoke({"query": query})
        return result
    except Exception as e:
        logger.error(f"QA 체인 실행 중 오류 발생: {e}")
        return {"result": fallback_text, "source_documents": []}

# ✅ 재무 데이터 전처리 (공통 함수)
@st.cache_data(ttl=300)
def process_financial_data(financial_data: list) -> pd.DataFrame:
    """재무 데이터 처리 최적화"""
    if not financial_data:
        return pd.DataFrame()
    
    # 스키마 검증
    required_columns = ['amount', 'fiscal_date', 'account_name']
    
    try:
        df = pd.DataFrame(financial_data)
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"누락된 컬럼: {missing_cols}")
            return pd.DataFrame()
        
        # 데이터 타입 최적화
        df = df.copy()
        df["amount"] = pd.to_numeric(df["amount"], errors='coerce')
        df["fiscal_date"] = pd.to_datetime(df["fiscal_date"], errors='coerce')
        
        # 유효한 데이터만 필터링
        df = df.dropna(subset=['amount', 'fiscal_date'])
        
        return df
        
    except Exception as e:
        logger.error(f"재무 데이터 처리 오류: {e}")
        return pd.DataFrame()

# ✅ 재무 데이터 포맷팅 (공통 함수)
def format_financial_summary(df: pd.DataFrame) -> str:
    """재무 데이터를 요약 텍스트로 포맷팅"""
    if df.empty:
        return "재무 데이터 없음"
    
    grouped = df.groupby("fiscal_date")
    lines = []
    for date, group in grouped:
        line = f"{date} 기준\n"
        for _, row in group.iterrows():
            formatted_value = format_korean_number_for_dashboard(
                row["account_name"], row["amount"]
            )
            line += f"- {row['account_name']}: {formatted_value}\n"
        lines.append(line)
    
    return "\n".join(lines)

# ✅ GPT 메시지 생성 (개선됨)
def generate_gpt4o_response_from_history_stream(system_prompt: str = None):
    """세션 메시지 히스토리를 기반으로 GPT 응답 생성"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    for msg in st.session_state.message:
        messages.append({"role": msg["role"], "content": msg["text"]})

    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        stream=True,
    )

    answer = ""
    for chunk in response:
        content_part = chunk.choices[0].delta.content or ""
        answer += content_part
        yield answer + "▌"
    yield answer

# 질문 판별 (개선됨)
def classify_question(text: str) -> str:
    """질문을 카테고리별로 분류"""
    text = text.lower()

    category_patterns = {
        "news": r"(뉴스|보도|이슈|최근.*소식|기사|언론|근황)",
        "meta": r"(매출|이익|자산|부채|재무비율|현금흐름|ROE|PER|PBR|FCF|EPS|재무|실적|변화)",
        "macro": r"(금리|환율|물가|GDP|경기|종합|분석|평가)"
    }

    for category in ["meta", "news", "macro"]:  # 우선순위 순서
        if re.search(category_patterns[category], text):
            return category
    return "general"

# 질문 카테고리에 따라 알맞은 RAG QA 체인을 반환하는 함수
def get_retrieval_chain_by_question(text: str):
    """질문을 분류하여 해당 카테고리와 QA 체인을 함께 반환"""
    category = classify_question(text)
    chain = {
        "news": qa_chain_news,
        "meta": qa_chain_meta,
    }.get(category, qa_chain_news)

    return category, chain

# ✅ 뉴스 소스 링크 포맷팅 (분리된 함수)
def format_news_sources(news_items: list, max_items: int = 5) -> str:
    """뉴스 아이템들을 출처 링크 형태로 포맷팅"""
    source_links = []
    for i, n in enumerate(news_items[:max_items], 1):
        if n.get('link') and n.get('date') and n.get('news_name'):
            title = n.get('news_name')
            if len(title) > 60:
                title = title[:57] + "..."
            
            date_str = n.get('date', '').replace('(', '').replace(')', '')
            source_links.append(f"• [{title}]({n.get('link')}) ({date_str})")
    
    return "\n\n".join(source_links) if source_links else "• 참고할 수 있는 뉴스 링크가 없습니다."

# 뉴스 관련 답변 (개선됨)
def ask_from_news_summary(question: str, news_items: list) -> Generator[str, None, None]:
    """최근 뉴스 요약 데이터를 기반으로 사용자의 질문에 답변을 생성합니다."""
    
    # 🔹 한글 번역 맵
    aspect_map = {
        "financial": "재무", "esg": "ESG", "investment_ma": "투자·인수합병",
        "risk_issue": "리스크", "strategy": "전략", "product_service": "제품·서비스",
        "general": "일반", "partnership": "협력", "economy": "경제"
    }
    
    sentiment_map = {
        "positive": "긍정", "neutral": "중립", "negative": "부정"
    }
    
    # 🔹 뉴스 요약 생성
    summary = "\n".join([
        f"- {n.get('news_name')} ({aspect_map.get(n.get('news_aspect'), '기타')} / {sentiment_map.get(n.get('overall_sentiment'), '감정 없음')})"
        for n in news_items[:5]
    ])
    
    if not summary.strip():
        summary = "❌ 관련 뉴스가 존재하지 않습니다."
    
    # 🔹 출처 링크 생성
    sources_text = format_news_sources(news_items)
    
    # 🔹 프롬프트 구성
    system_prompt = f"""
당신은 초보 투자자에게 뉴스를 쉽게 해석해주는 **금융 뉴스 해설 전문가**입니다.

## 🧠 분석 원칙:
1. **기사에 명시된 내용만 사용**합니다. 추정하거나 유추하지 마세요.
2. **기사에서 언급된 수치와 키워드만** 기반으로 분석하세요.
3. **명확하고 구조화된 답변**을 제공하세요.
4. 전체 답변은 다음 순서로 구성하세요:
   **실적 요약** (수치 중심)
   **사업부별 주요 이슈** (AI 반도체, 시스템 반도체 등)
   **향후 전망** (기업이 언급한 전략)
   **기업의 대응 전략 또는 영향**
5. **답변 마지막에 출처를 다음과 같이 추가**하세요:

**출처:**

{sources_text}

## [사용자 질문]
{question}

## [최근 뉴스 요약]
{summary}

사용자의 질문에 대해 위 기사들을 참고하여, 초보 투자자가 이해할 수 있도록 친절하고 명확하게 설명해 주세요.
답변 마지막에는 반드시 위에 제시된 출처 형식을 그대로 포함해주세요.
"""
    
    return generate_gpt4o_response_from_history_stream(system_prompt)

# ✅ 재무 기반 GPT 답변 (개선됨)
def generate_financial_based_answer_stream(question: str, financial_data: list):
    """재무제표 데이터를 바탕으로 GPT가 쉽게 해석한 설명을 생성합니다."""
    
    # 🔹 데이터 전처리 (캐시된 함수 사용)
    df = process_financial_data(financial_data)
    formatted = format_financial_summary(df)

    # 🔹 GPT system 프롬프트
    system_prompt = f"""
당신은 재무제표를 쉽게 설명해 주는 **기업 분석가**입니다.

사용자의 질문에 대해 아래 지침에 따라 응답해 주세요:

- **회계 비전문가도 이해할 수 있게** 용어를 풀어 설명해 주세요.
- **수치보다는 변화 방향과 흐름**에 중점을 둬서 설명해 주세요.
- **무엇이 어떻게 변했고**, **그게 왜 중요한지** 중심으로 분석해 주세요.
- 가능하면 **간단한 비유나 사례**를 통해 설명을 돕습니다.
- 감정적 평가나 추천 없이, **중립적이고 명확한** 설명만 해주세요.

## [사용자 질문]
{question}

## [재무제표 요약]
{formatted}
"""
    return generate_gpt4o_response_from_history_stream(system_prompt)

# ✅ 공통 포맷팅 함수들
def format_ratios(ratios: list) -> str:
    """재무비율 포맷팅"""
    if not ratios:
        return "재무비율 정보 없음"
    return "\n".join([f"- {r['year']}년 {r['metric']}: {r['value']}" for r in ratios])

def format_news_simple(news_items: list) -> str:
    """뉴스 간단 포맷팅"""
    if not news_items:
        return "관련 뉴스 없음"
    return "\n".join([
        f"- [{n.get('news_name', '기사 없음')}]({n.get('link', '')})"
        for n in news_items
    ])

def format_macro(macro: list) -> str:
    """거시경제 지표 포맷팅"""
    if not macro:
        return "거시경제 정보 없음"
    df = pd.DataFrame(macro)
    df = df.sort_values(by="date", ascending=False)
    latest = df.groupby("indicator").first().reset_index()
    return "\n".join([
        f"- {row['indicator']}: {row['value']} (기준일: {row['date']})"
        for _, row in latest.iterrows()
    ])

# ✅ 종합 분석 + RAG 결합 GPT 답변 (개선됨)
def answer_with_context_and_rag_stream(question: str, data: dict, qa_chain, news_items: list = []):
    """LLM이 다양한 기업 데이터를 바탕으로 종합 분석과 RAG 문서를 결합해 해석적 설명을 생성합니다."""

    # 🔹 데이터 추출
    ratios = data.get("financial_ratios", [])
    fin_data = data.get("financial_raw", [])
    macro = data.get("econ_idx", [])

    # 🔹 RAG 문서 요약
    rag_docs = qa_chain.invoke({"query": question}).get("source_documents", [])
    rag_summary = "\n".join([doc.page_content for doc in rag_docs[:3]]) or "관련 문서 없음"

    # 🔹 재무제표 처리 (캐시된 함수 사용)
    df = process_financial_data(fin_data)
    formatted_financials = format_financial_summary(df)
    
    # 🔹 프롬프트 구성
    system_prompt = f"""
당신은 **투자 리서치 전문가**입니다.

사용자의 질문에 대해 다음 기준으로 설명해 주세요:

- 기업의 **사업 모델, 성장 전략** 등 개요를 간단히 설명
- **재무제표와 재무비율**을 통해 수익성·안정성·성장성 분석
- **뉴스와 관련 문서(RAG)**로 최근 이슈와 방향성 해석
- **거시경제 흐름(금리, 환율 등)**이 기업에 미치는 영향 해석
- 마지막에는 **투자자에게 중요한 시사점**을 요약해 주세요 (2~3줄)

📌 분석은 간결하고 논리적으로, 수치 중심이 아닌 **해석과 설명 중심**으로 구성하세요.

질문: {question}

[재무비율]
{format_ratios(ratios)}

[재무제표]
{formatted_financials}

[거시경제 지표]
{format_macro(macro)}

[최근 뉴스]
{format_news_simple(news_items)}

[RAG 기반 관련 문서]
{rag_summary}

📝 [투자자 시사점 요약]
- 핵심 재무 흐름과 최근 이슈를 바탕으로 투자자에게 도움이 될 **간결한 인사이트**를 제공하세요.
"""
    return generate_gpt4o_response_from_history_stream(system_prompt)

# ------------------ Streamlit UI ------------------
import streamlit as st

if "message" not in st.session_state:
    st.session_state.message = []

# 🔹 Q&A 영역 타이틀
st.markdown("---")
st.markdown("## 💬 Q&A")


# 🔹 Q&A 영역을 하나의 컨테이너로 묶음
with st.container():
    company_name = st.session_state.get("company")
    if not company_name:
        st.warning("기업명을 먼저 선택해 주세요.")
        st.stop()

    try:
        data = fetch_company_data(company_name)
    except Exception as e:
        st.error(f"데이터 호출 실패: {e}")
        st.stop()

    # 🔁 최근 메시지만 출력 (일관성 유지)
    messages_to_show = st.session_state.get("message", [])[-3:]  # 최근 3개만
    
# 💬 사용자 입력 (최상단에 위치)
question = st.chat_input("기업 및 시장 관련 질문을 입력해보세요.")

# 채팅 영역 스타일 개선
st.markdown("""
<style>
    .stChatMessage {
        margin-bottom: 1rem;
    }
    .main .block-container {
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# 메시지 표시
if 'message' not in st.session_state:
    st.session_state.message = []

# 보여줄 메시지 수 제한 (메모리 절약)
messages_to_show = st.session_state.message[-20:] if len(st.session_state.message) > 20 else st.session_state.message

for msg in messages_to_show:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

# 사용자 입력 처리
if question:
    # 사용자 메시지 추가 및 표시
    st.session_state.message.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    # AI 응답 생성
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.info("AI가 답변 생성 중입니다...")
        
        category, qa_chain = get_retrieval_chain_by_question(question)

        if category == "news":
            stream_gen = ask_from_news_summary(
                question, data.get("news", [])
            )
        elif category == "meta":
            stream_gen = generate_financial_based_answer_stream(
                question, data.get("financial_raw", [])
            )
        else:
            stream_gen = answer_with_context_and_rag_stream(
                question, data, qa_chain, data.get("news", [])
            )

        # 스트리밍 출력
        last_answer = ""
        for partial in stream_gen:
            placeholder.markdown(partial + "▌")
            last_answer = partial
        placeholder.markdown(last_answer)
        answer = last_answer

    st.session_state.message.append({"role": "assistant", "text": answer})

# ✅ 세션 메시지 정리 (최근 3개만 유지)
MAX_MESSAGES = 3
if len(st.session_state.message) > MAX_MESSAGES:
    st.session_state.message = st.session_state.message[-MAX_MESSAGES:]

