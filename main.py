import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="기온 상승 트렌드 분석 (결측치 리커버리)",
    page_icon="🌡️",
    layout="wide"
)

# 타이틀 및 대시보드 소개 (st.html과 내부 API를 사용하여 컴파일 에러 원천 차단)
st.title("🌡️ 1980년대 전후 기온 상승 가설 검증 웹앱")
st.caption("데이터 내 결측치를 자동으로 복원(Recovery)하고, 기준 연도 전후의 기온 상승 속도를 비교합니다.")
st.divider()

# 2. [리커버리 핵심] 데이터 로드 및 누락값 자동 복원 전처리 함수
@st.cache_data
def load_and_recover_data(file_path):
    # CSV 읽기 및 컬럼 공백 제거
    df = pd.read_csv(file_path, encoding='utf-8')
    df.columns = df.columns.str.strip()
    
    # 날짜 컬럼 정제 (\t, 공백, 따옴표 제거)
    df['날짜'] = df['날짜'].astype(str).str.replace(r'[\t\s"]', '', regex=True)
    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    
    # 날짜 자체가 누락된 비정상 행만 제외
    df = df.dropna(subset=['날짜'])
    df['연도'] = df['날짜'].dt.year
    
    target_cols = ['평균기온(℃)', '최저기온(℃)', '최고기온(℃)']
    
    # ⭐ [리커버리] 기온 데이터가 누락된 경우, 앞뒤 데이터를 기반으로 한 선형 보간법으로 자동 복원
    for col in target_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # 선형 보간 후, 처음이나 끝에 남은 결측치는 직전/직후 값으로 채움
            df[col] = df[col].interpolate(method='linear').bfill().ffill()
            
    # 연도별 평균 데이터로 그룹화
    annual_df = df.groupby('연도')[target_cols].mean().reset_index()
    return annual_df

# 데이터 불러오기 실행
try:
    data = load_and_recover_data("ta_20260601093156.csv")
    
    # 3. 사이드바 제어 패널
    st.sidebar.header("⚙️ 분석 설정")
    st.sidebar.info("원하는 기온 지표와 기준 연도를 선택하면 대시보드가 실시간으로 업데이트됩니다.")
    
    target_col = st.sidebar.selectbox(
        "분석할 기온 지표 선택",
        ['평균기온(℃)', '최저기온(℃)', '최고기온(℃)']
    )
    
    split_year = st.sidebar.slider(
        "가설 기준 연도 설정", 
        int(data['연도'].min()) + 5, 
        int(data['연도'].max()) - 5, 
        1980, 
        step=1
    )

    # 데이터 분할
    df_before = data[data['연도'] < split_year].copy()
    df_after = data[data['연도'] >= split_year].copy()

    # 4. 선형 회귀(트렌드) 계산 함수
    def calculate_trend(df, col):
        if len(df) < 2:
            return 0, 0, np.array([])
        X = df[['연도']].values
        y = df[col].values
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        pred_y = model.predict(X)
        return slope, intercept, pred_y

    slope_b, intercept_b, pred_b = calculate_trend(df_before, target_col)
    slope_a, intercept_a, pred_a = calculate_trend(df_after, target_col)

    # 5. 핵심 스탯 지표 (Metrics) 시각화 - 박스 스타일 제거 후 표준 에러 없는 안전한 컴포넌트 구조로 변경
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label=f"📉 {split_year}년 이전 평균 기온", 
            value=f"{df_before[target_col].mean():.2f} ℃",
            delta=f"10년당 +{slope_b * 10:.3f} ℃",
            delta_color="inverse"
        )
        
    with col2:
        mean_diff = df_after[target_col].mean() - df_before[target_col].mean()
        st.metric(
            label=f"📈 {split_year}년 이후 평균 기온", 
            value=f"{df_after[target_col].mean():.2f} ℃",
            delta=f"10년당 +{slope_a * 10:.3f} ℃"
        )
        
    with col3:
        rate_diff = (slope_a / slope_b) if slope_b > 0 else 0
        status_text = "온난화 가속화" if slope_a > slope_b else "온난화 둔화"
        
        st.metric(
            label="🔄 기온 상승 속도 변화", 
            value=f"{rate_diff:.1f}배 속도",
            delta=f"{status_text} 상태",
            delta_color="normal" if slope_a > slope_b else "inverse"
        )

    st.subheader("📈 연도별 추세선 및 가설 검증 그래프")
    
    # 6. 인터랙티브 Plotly 차트 시각화
    fig = go.Figure()
    
    # 전체 실제 데이터 산점도
    fig.add_trace(go.Scatter(
        x=data['연도'], y=data[target_col],
        mode='markers', name='연평균 기온 (복원완료)',
        marker=dict(color='#94A3B8', size=6, opacity=0.6)
    ))
    
    # 이전 기간 트렌드 선
    if len(df_before) > 0:
        fig.add_trace(go.Scatter(
            x=df_before['연도'], y=pred_b,
            mode='lines', name=f'{split_year}년 이전 추세선 (10년당 {slope_b*10:.2f}℃)',
            line=dict(color='#2563EB', width=3.5)
        ))
        
    # 이후 기간 트렌드 선
    if len(df_after) > 0:
        fig.add_trace(go.Scatter(
            x=df_after['연도'], y=pred_a,
            mode='lines', name=f'{split_year}년 이후 추세선 (10년당 {slope_a*10:.2f}℃)',
            line=dict(color='#DC2626', width=3.5)
        ))

    # 기준 연도 점선 표기
    fig.add_vline(x=split_year, line_width=1.5, line_dash="dash", line_color="#475569")
    
    fig.update_layout(
        xaxis_title="연도 (Year)",
        yaxis_title="기온 (Temperature, ℃)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
