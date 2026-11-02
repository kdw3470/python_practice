import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
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
# 5. Scipy 수학적 최적화 (시장 포트폴리오 & 효율적 프론티어)
# ---------------------------------------------------------
print("\n3. Scipy를 활용해 수학적 최적 포트폴리오와 효율적 프론티어를 계산 중입니다...")

# 가중치(w)를 넣으면 수익률, 변동성, 샤프 지수를 반환하는 헬퍼 함수
def get_portfolio_stats(weights):
    weights = np.array(weights)
    port_return = np.sum(annual_mean_returns.values * weights)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix.values, weights)))
    sharpe = (port_return - rf_rate) / port_vol
    return port_return, port_vol, sharpe

# 목적함수 1: 음(-)의 샤프 지수 최소화 (= 샤프 지수 최대화)
def min_func_sharpe(weights):
    return -get_portfolio_stats(weights)[2]

# 목적함수 2: 포트폴리오 변동성(표준편차) 최소화
def min_func_volatility(weights):
    return get_portfolio_stats(weights)[1]

# 공통 제약조건(비중의 합 = 1) 및 범위(공매도 금지: 0 <= w <= 1)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0.0, 1.0) for _ in range(num_assets))
init_guess = num_assets * [1. / num_assets]  # 초기 탐색점: 동일 비중(1/N)

# ① 최대 샤프 지수 포트폴리오 (시장 포트폴리오) 최적화
opt_sharpe = sco.minimize(min_func_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
opt_sharpe_w = opt_sharpe.x
opt_sharpe_ret, opt_sharpe_vol, opt_sharpe_sr = get_portfolio_stats(opt_sharpe_w)

# ② 최소 분산 포트폴리오 최적화
opt_var = sco.minimize(min_func_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
opt_var_w = opt_var.x
opt_var_ret, opt_var_vol, opt_var_sr = get_portfolio_stats(opt_var_w)

# 최적화 요약 데이터프레임 생성 및 기존 시뮬레이션 요약과 병합 (4줄 비교표)
opt_summary_data = [
    [opt_sharpe_ret, opt_sharpe_vol, opt_sharpe_sr] + list(opt_sharpe_w),
    [opt_var_ret, opt_var_vol, opt_var_sr] + list(opt_var_w)
]
opt_summary_df = pd.DataFrame(
    opt_summary_data,
    columns=sim_summary_df.columns,
    index=['Max_Sharpe (Optimized)', 'Min_Volatility (Optimized)']
)
comparison_summary_df = pd.concat([opt_summary_df, sim_summary_df])

# ③ 효율적 프론티어 곡선(50개 목표 수익률 구간) 계산
# 최소 분산 수익률부터 개별 종목 최고 수익률까지 50개 구간으로 쪼갬
target_returns = np.linspace(opt_var_ret, annual_mean_returns.max(), 50)
ef_volatilities = []
ef_weights_list = []

for tret in target_returns:
    # 제약조건 2개: 비중의 합 = 1 AND 포트폴리오 기대수익률 = 목표수익률(tret)
    ef_cons = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x, tret=tret: get_portfolio_stats(x)[0] - tret}
    )
    res = sco.minimize(min_func_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=ef_cons)
    ef_volatilities.append(res['fun'])
    ef_weights_list.append(res['x'])

ef_volatilities = np.array(ef_volatilities)
ef_sharpes = (target_returns - rf_rate) / ef_volatilities

ef_data = {
    'Target_Return': target_returns,
    'Min_Volatility': ef_volatilities,
    'Sharpe_Ratio': ef_sharpes
}
for i, ticker in enumerate(actual_tickers):
    ef_data[f'Weight_{ticker}'] = [w[i] for w in ef_weights_list]
ef_df = pd.DataFrame(ef_data)

# ---------------------------------------------------------
# 5-1. 시뮬레이션 + 효율적 프론티어 + CAL 통합 그래프 생성
# ---------------------------------------------------------
print("4. 효율적 프론티어 및 자본배분선(CAL) 통합 그래프를 생성 중입니다...")

plt.figure(figsize=(11, 7))

# 1) 1만 개 무작위 포트폴리오 산점도 (배경)
scatter = plt.scatter(
    sim_df['Volatility'], sim_df['Expected_Return'], 
    c=sim_df['Sharpe_Ratio'], cmap='viridis', alpha=0.35, s=10, label='Simulated Portfolios'
)
plt.colorbar(scatter, label='Sharpe Ratio')

# 2) 효율적 프론티어 곡선 (검은색 굵은 실선)
plt.plot(ef_volatilities, target_returns, 'k-', linewidth=2.5, zorder=4, label='Efficient Frontier (Scipy)')

# 3) 자본배분선 (CAL): 무위험자산(0, rf_rate)에서 시장 포트폴리오를 지나 뻗어나가는 직선
max_vol_plot = max(sim_df['Volatility'].max(), ef_volatilities.max()) * 1.05
cal_x = np.linspace(0, max_vol_plot, 100)
cal_y = rf_rate + opt_sharpe_sr * cal_x
plt.plot(cal_x, cal_y, color='crimson', linestyle='--', linewidth=2, zorder=4, label='Capital Allocation Line (CAL)')

# 4) 수학적 최적점 & 무위험자산 마커 표시
# 시장 포트폴리오 (최대 샤프 지수 - 빨간 별표)
plt.scatter(opt_sharpe_vol, opt_sharpe_ret, color='red', marker='*', s=300, edgecolor='black', zorder=6, 
            label=f'Market Portfolio (Sharpe: {opt_sharpe_sr:.2f})')
# 최소 분산 포트폴리오 (파란 별표)
plt.scatter(opt_var_vol, opt_var_ret, color='blue', marker='*', s=300, edgecolor='black', zorder=6, 
            label='Min Volatility (Optimized)')
# 무위험자산 (미국채 10년물 - 초록 다이아몬드)
plt.scatter(0, rf_rate, color='limegreen', marker='D', s=150, edgecolor='black', zorder=6, 
            label=f'Risk-Free Asset ({rf_rate:.2%})')

plt.title('Efficient Frontier & Capital Allocation Line (CAL)', fontsize=14, fontweight='bold')
plt.xlabel('Annualized Volatility (Risk)', fontsize=12)
plt.ylabel('Annualized Expected Return', fontsize=12)
plt.xlim(left=-0.02, right=max_vol_plot)
plt.ylim(bottom=min(0, rf_rate - 0.05), top=max(sim_df['Expected_Return'].max(), target_returns.max()) * 1.1)
plt.legend(loc='upper left', fontsize=9.5)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# 메모리에 이미지 저장
img_buffer = io.BytesIO()
plt.savefig(img_buffer, format='png', dpi=150)
plt.close()
img_buffer.seek(0)



# ---------------------------------------------------------
# 6. 엑셀 파일로 저장 및 스타일링
# ---------------------------------------------------------
excel_filename = 'portfolio_final_optimization.xlsx'
print(f"\n5. 모든 계산 완료! 최종 엑셀 파일({excel_filename})을 생성합니다...")

with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    # 1) 시트별 데이터 저장 (최적화 요약 및 효율적 프론티어 시트를 맨 앞에 배치)
    comparison_summary_df.to_excel(writer, sheet_name='Optimization_Summary')
    ef_df.to_excel(writer, sheet_name='Efficient_Frontier', index=False)
    annual_mean_returns.to_frame('Annual_Expected_Return').to_excel(writer, sheet_name='Expected_Returns')
    annual_cov_matrix.to_excel(writer, sheet_name='Covariance_Matrix')
    rf_df.to_excel(writer, sheet_name='Risk_Free_Rate', index=False)
    sim_df.to_excel(writer, sheet_name='Simulation_All', index=False)
    data.to_excel(writer, sheet_name='Adj_Close_Prices')
    daily_returns.to_excel(writer, sheet_name='Daily_Returns')
    
    # 2) 엑셀 스타일링 코드
    workbook = writer.book
    header_fill = PatternFill(start_color="2F3542", end_color="2F3542", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True)
    
    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
        
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
            
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
            
        # 퍼센트 및 소수점 숫자 포맷 적용
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
        # 👇 Optimization_Summary와 Efficient_Frontier 시트도 % 및 소수점 포맷팅 적용!
        elif sheet_name in ['Optimization_Summary', 'Efficient_Frontier', 'Simulation_All']:
            headers = [cell.value for cell in sheet[1]]
            for row in sheet.iter_rows(min_row=2):
                for idx, cell in enumerate(row):
                    if isinstance(cell.value, (int, float)):
                        if str(headers[idx]) == 'Sharpe_Ratio':
                            cell.number_format = '0.00'
                        else:
                            cell.number_format = '0.00%'

    # 3) Optimization_Summary 시트의 B8 셀 위치에 완성된 그래프 삽입
    summary_sheet = workbook['Optimization_Summary']
    excel_img = OpenpyxlImage(img_buffer)
    summary_sheet.add_image(excel_img, 'B8')

print(f"완료되었습니다! 프로젝트 폴더에서 {excel_filename} 파일을 확인해주세요.")