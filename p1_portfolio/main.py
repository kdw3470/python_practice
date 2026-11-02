import warnings
# 지저분한 Pandas 경고 메시지 숨기기
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openpyxl.styles import Font, PatternFill, Alignment

# 1. 사용자 입력 부분 (자유롭게 종목 추가/삭제 가능)
tickers = ['005930.KS', '000660.KS', 'NVDA', 'AAPL', 'MSFT']
risk_free_ticker = '^TNX'  # 미국 10년물 국채 금리

# 3년치 날짜 계산
end_date = datetime.today()
start_date = end_date - timedelta(days=3*365)

print("데이터를 다운로드하고 있습니다...")

# 2. 주가 및 무위험 수익률 다운로드
# auto_adjust=True를 명시하면 'Close' 컬럼이 곧 '기말수정종가(Adj Close)'가 됩니다.
raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
data = raw_data['Close']

# 종목이 1개일 경우 Series로 반환되므로 DataFrame으로 형태 유지
if isinstance(data, pd.Series):
    data = data.to_frame(name=tickers[0])

# 결측치 처리 (한국/미국 휴장일 차이로 인한 빈칸을 앞뒤 데이터로 채움)
data = data.ffill().bfill()

# 미국 10년물 국채 금리 다운로드 (최근 데이터)
rf_data = yf.Ticker(risk_free_ticker).history(period='5d')
rf_rate = float(rf_data['Close'].dropna().iloc[-1]) / 100
rf_df = pd.DataFrame({'Metric': ['10-Year US Treasury Yield'], 'Current_Rate': [rf_rate]})

# 3. 수익률 및 공분산 행렬 계산
# 일일 수익률 계산
daily_returns = data.pct_change().dropna()

# 연율화된 기대수익률 (1년 거래일 252일 가정)
annual_mean_returns = daily_returns.mean() * 252

# 연율화된 분산-공분산 행렬 계산
annual_cov_matrix = daily_returns.cov() * 252

# 4. 엑셀 파일로 저장 및 스타일링
excel_filename = 'portfolio_step1_data.xlsx'
print(f"\n데이터 계산 완료! 엑셀 파일({excel_filename})을 생성합니다...")

with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    # 각각 다른 시트에 데이터 저장
    data.to_excel(writer, sheet_name='Adj_Close_Prices')
    daily_returns.to_excel(writer, sheet_name='Daily_Returns')
    annual_mean_returns.to_frame('Annual_Expected_Return').to_excel(writer, sheet_name='Expected_Returns')
    annual_cov_matrix.to_excel(writer, sheet_name='Covariance_Matrix')
    rf_df.to_excel(writer, sheet_name='Risk_Free_Rate', index=False)
    
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
            
        # 열 너비 자동 조절
        for col in sheet.columns:
            max_length = 0
            column = col[0].column_letter 
            for cell in col:
                try:
                    if cell.value is not None and len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            sheet.column_dimensions[column].width = max(max_length + 3, 12)
            
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

print(f"완료되었습니다! 프로젝트 폴더에서 {excel_filename} 파일을 확인해주세요.")