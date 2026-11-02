import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from openpyxl.drawing.image import Image as OpenpyxlImage
from datetime import datetime, timedelta
from openpyxl.styles import Font, PatternFill, Alignment

# 1. 사용자 입력 부분 (자유롭게 종목 추가/삭제 가능)
tickers = ['005930.KS', '000660.KS', 'NVDA', 'AAPL', 'MSFT']
risk_free_ticker = '^TNX'  # 미국 10년물 국채 금리
num_portfolios = 10000     # 시뮬레이션할 무작위 포트폴리오 개수

# 3년치 날짜 계산
end_date = datetime.today()
start_date = end_date - timedelta(days=3*365)

print("1. 데이터를 다운로드하고 있습니다...")

# 2. 주가 및 무위험 수익률 다운로드
raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
data = raw_data['Close']

if isinstance(data, pd.Series):
    data = data.to_frame(name=tickers[0])

# 다운로드된 데이터프레임의 실제 컬럼 순서대로 티커 리스트 정렬 (데이터 꼬임 방지)
actual_tickers = list(data.columns)
num_assets = len(actual_tickers)

data = data.ffill().bfill()

# 미국 10년물 국채 금리 다운로드
rf_data = yf.Ticker(risk_free_ticker).history(period='5d')
rf_rate = float(rf_data['Close'].dropna().iloc[-1]) / 100
rf_df = pd.DataFrame({'Metric': ['10-Year US Treasury Yield'], 'Current_Rate': [rf_rate]})

# 3. 수익률 및 공분산 행렬 계산
daily_returns = data.pct_change().dropna()
annual_mean_returns = daily_returns.mean() * 252
annual_cov_matrix = daily_returns.cov() * 252

# ---------------------------------------------------------
# 4. 무작위 포트폴리오 시뮬레이션 (몬테카를로)
# ---------------------------------------------------------
print(f"\n2. {num_portfolios:,}개의 무작위 포트폴리오를 시뮬레이션 중입니다...")

# 재현성을 위한 시드 고정 (실행할 때마다 완전히 다른 난수를 원하면 주석 처리하세요)
np.random.seed(42)

# 난수 생성 및 가중치 합이 1이 되도록 정규화 (행: 포트폴리오 수, 열: 종목 수)
random_weights = np.random.random((num_portfolios, num_assets))
random_weights /= np.sum(random_weights, axis=1)[:, np.newaxis]

# 포트폴리오 기대 수익률 계산: W * R
sim_returns = np.dot(random_weights, annual_mean_returns.values)

# 포트폴리오 변동성(표준편차) 계산: sqrt(W * Cov * W^T)
sim_volatilities = np.sqrt(np.einsum('ij,jk,ik->i', random_weights, annual_cov_matrix.values, random_weights))

# 샤프 지수 계산: (기대수익률 - 무위험수익률) / 변동성
sim_sharpe_ratios = (sim_returns - rf_rate) / sim_volatilities

# 시뮬레이션 결과를 데이터프레임으로 정리
sim_data = {
    'Expected_Return': sim_returns,
    'Volatility': sim_volatilities,
    'Sharpe_Ratio': sim_sharpe_ratios
}
# 각 종목별 비중 컬럼 추가
for i, ticker in enumerate(actual_tickers):
    sim_data[f'Weight_{ticker}'] = random_weights[:, i]

sim_df = pd.DataFrame(sim_data)

# 시뮬레이션 중 최고 샤프 지수 & 최소 변동성 포트폴리오 추출
max_sharpe_idx = sim_df['Sharpe_Ratio'].idxmax()
min_vol_idx = sim_df['Volatility'].idxmin()

sim_summary_df = pd.DataFrame(
    [sim_df.loc[max_sharpe_idx], sim_df.loc[min_vol_idx]],
    index=['Max_Sharpe (Simulated)', 'Min_Volatility (Simulated)']
)

# ---------------------------------------------------------
# 4-1. 시뮬레이션 산점도 그래프 생성 (메모리에 저장)
# ---------------------------------------------------------
print("시뮬레이션 그래프를 생성 중입니다...")

plt.figure(figsize=(10, 6))
# 1만 개 포트폴리오 산점도 (샤프 지수가 높을수록 밝은 색)
scatter = plt.scatter(
    sim_df['Volatility'], 
    sim_df['Expected_Return'], 
    c=sim_df['Sharpe_Ratio'], 
    cmap='viridis', 
    alpha=0.5, 
    s=10
)
plt.colorbar(scatter, label='Sharpe Ratio')

# 최고 샤프 지수 포트폴리오 (빨간 별표)
plt.scatter(
    sim_df.loc[max_sharpe_idx, 'Volatility'], 
    sim_df.loc[max_sharpe_idx, 'Expected_Return'], 
    color='red', marker='*', s=250, edgecolor='black', label='Max Sharpe (Simulated)'
)

# 최소 변동성 포트폴리오 (파란 별표)
plt.scatter(
    sim_df.loc[min_vol_idx, 'Volatility'], 
    sim_df.loc[min_vol_idx, 'Expected_Return'], 
    color='blue', marker='*', s=250, edgecolor='black', label='Min Volatility (Simulated)'
)

plt.title('Monte Carlo Portfolio Simulation (10,000 Portfolios)', fontsize=14, fontweight='bold')
plt.xlabel('Annualized Volatility (Risk)', fontsize=12)
plt.ylabel('Annualized Expected Return', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()


# (기존 코드: 파란 별표 찍는 부분 바로 아래에 추가하세요)

# 1) 무위험자산 위치 표시 (변동성 = 0, 수익률 = rf_rate) - 초록색 다이아몬드 마커
plt.scatter(
    0, 
    rf_rate, 
    color='limegreen', marker='D', s=150, edgecolor='black', zorder=5,
    label=f'Risk-Free Asset ({rf_rate:.2%})'
)

# 2) 무위험자산과 최고 샤프지수 포트폴리오를 잇는 예비 자본배분선(CAL) 점선 추가
plt.plot(
    [0, sim_df.loc[max_sharpe_idx, 'Volatility']], 
    [rf_rate, sim_df.loc[max_sharpe_idx, 'Expected_Return']], 
    color='crimson', linestyle='--', linewidth=1.5, label='Simulated CAL'
)

# 3) X축이 0(무위험자산 위치)부터 잘 보이도록 여백 설정
plt.xlim(left=-0.02)

# 파일로 저장하지 않고 메모리 버퍼(BytesIO)에 고해상도 이미지로 저장
img_buffer = io.BytesIO()
plt.savefig(img_buffer, format='png', dpi=150)
plt.close()
img_buffer.seek(0)

# ---------------------------------------------------------
# 5. 엑셀 파일로 저장 및 스타일링
# ---------------------------------------------------------
excel_filename = 'portfolio_step2_simulation.xlsx'
print(f"\n3. 시뮬레이션 완료! 엑셀 파일({excel_filename})을 생성합니다 (약 5~10초 소요)...")

with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    # 앞쪽에 중요 요약 시트 배치
    sim_summary_df.to_excel(writer, sheet_name='Simulation_Summary')
    annual_mean_returns.to_frame('Annual_Expected_Return').to_excel(writer, sheet_name='Expected_Returns')
    annual_cov_matrix.to_excel(writer, sheet_name='Covariance_Matrix')
    rf_df.to_excel(writer, sheet_name='Risk_Free_Rate', index=False)
    # 1만 개 전체 데이터 및 원본 주가 시트
    sim_df.to_excel(writer, sheet_name='Simulation_All', index=False)
    data.to_excel(writer, sheet_name='Adj_Close_Prices')
    daily_returns.to_excel(writer, sheet_name='Daily_Returns')
    
    # --- 엑셀 스타일링 코드 ---
    workbook = writer.book
    header_fill = PatternFill(start_color="2F3542", end_color="2F3542", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True)
    
    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
        
        # 첫 번째 행(헤더) 스타일 적용
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
            
        # 열 너비 자동 조절 (속도를 위해 상위 50행만 기준으로 측정)
        for col in sheet.columns:
            max_length = 0
            column = col[0].column_letter 
            for cell in col[:50]:
                try:
                    if cell.value is not None and len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            sheet.column_dimensions[column].width = max(max_length + 3, 14)
            
        # 시트별 숫자 포맷 적용
        if sheet_name in ['Daily_Returns', 'Expected_Returns', 'Risk_Free_Rate']:
            for row in sheet.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = '0.00%'
        elif sheet_name == 'Covariance_Matrix':
            for row in sheet.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = '0.000000'
        elif sheet_name in ['Simulation_Summary', 'Simulation_All']:
            # 헤더 이름을 기준으로 수익률/변동성/비중은 %, 샤프 지수는 소수점 둘째 자리로 포맷팅
            headers = [cell.value for cell in sheet[1]]
            for row in sheet.iter_rows(min_row=2):
                for idx, cell in enumerate(row):
                    if isinstance(cell.value, (int, float)):
                        col_name = str(headers[idx])
                        if col_name == 'Sharpe_Ratio':
                            cell.number_format = '0.00'
                        else:
                            cell.number_format = '0.00%'

    # 👇 Simulation_Summary 시트의 B6 셀 위치에 그래프 이미지 삽입!
    summary_sheet = workbook['Simulation_Summary']
    excel_img = OpenpyxlImage(img_buffer)
    summary_sheet.add_image(excel_img, 'B6')

print(f"완료되었습니다! 프로젝트 폴더에서 {excel_filename} 파일을 확인해주세요.")