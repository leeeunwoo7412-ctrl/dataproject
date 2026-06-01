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

# 타이틀 및 대시보드 소개
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

    # 5. 핵심 스탯 지표 (Metrics) 시각화
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
    
    # ⚠️ [오류 수정 핵심] 괄호가 꼬이지 않도록 복잡한 줄바꿈을 정돈하고, 가독성 높은 일렬/명확한 인덴트로 완전히 다시 작성했습니다.
    fig.update_layout(
        xaxis_title="연도 (Year)",
        yaxis_title="기온 (Temperature, ℃)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor='white',
        height=550
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#F1F5F9')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#F1F5F9')
    
    st.plotly_chart(fig, use_container_width=True)

    # 7. 데이터 테이블 뷰어
    with st.expander("🔍 결측치가 완벽히 복원된(Recovery) 전체 연도별 데이터 보기"):
        st.dataframe(
            data.style.format({"연도": "{:.0f}", "평균기온(℃)": "{:.2f}", "최저기온(℃)": "{:.2f}", "최고기온(℃)": "{:.2f}"}),
            use_container_width=True
        )

except FileNotFoundError:
    st.error("❌ 파일(`ta_20260601093156.csv`)을 찾을 수 없습니다. 대시보드 스크립트와 동일한 디렉토리에 위치시켜 주세요.")
except Exception as e:
    st.error(f"❌ 데이터 로드 및 가공 중 에러 발생: {e}")
