from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from io import BytesIO
import os
import plotly.graph_objects as go
import plotly.io as pio
import base64
from PIL import Image

def register_korean_font():
    """한글 폰트 등록 함수"""
    try:
        # 여러 폰트 경로 시도
        font_paths = [
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/System/Library/Fonts/AppleGothic.ttf',  # macOS
            'C:/Windows/Fonts/malgun.ttf',  # Windows
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'  # 대체 폰트
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                if 'NanumGothic' in path:
                    pdfmetrics.registerFont(TTFont('Korean', path))
                elif 'AppleGothic' in path:
                    pdfmetrics.registerFont(TTFont('Korean', path))
                elif 'malgun' in path:
                    pdfmetrics.registerFont(TTFont('Korean', path))
                else:
                    pdfmetrics.registerFont(TTFont('Korean', path))
                return True
        
        # 폰트를 찾을 수 없으면 기본 폰트 사용
        return False
    except Exception:
        return False

def create_financial_chart():
    """재무지표 차트 생성"""
    fig = go.Figure(go.Bar(
        x=['매출', '순이익', 'ROE', '부채비율', 'PER', 'PBR'],
        y=[20, 2.8, 10.2, 95.0, 21.4, 1.9],
        marker_color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#2E86AB', '#A23B72']
    ))
    fig.update_layout(
        title="재무지표 현황",
        xaxis_title="지표",
        yaxis_title="값",
        width=600,
        height=400,
        font=dict(size=12)
    )
    return fig

def create_stock_chart(stock_date, stock_price):
    """주가 차트 생성"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stock_date, 
        y=stock_price, 
        mode='lines+markers',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=8)
    ))
    fig.update_layout(
        title="주가 변동 추이",
        xaxis_title="날짜",
        yaxis_title="주가 (원)",
        width=600,
        height=400,
        font=dict(size=12)
    )
    return fig

def create_forecast_chart(forecast_date, forecast_price):
    """예측 주가 차트 생성"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_date, 
        y=forecast_price, 
        mode='lines+markers',
        line=dict(color='#F18F01', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    fig.update_layout(
        title="예측 주가 전망",
        xaxis_title="날짜",
        yaxis_title="예측 주가 (원)",
        width=600,
        height=400,
        font=dict(size=12)
    )
    return fig

def plotly_to_image(fig):
    """Plotly 차트를 이미지로 변환"""
    try:
        # Plotly 차트를 PNG 바이트로 변환
        img_bytes = pio.to_image(fig, format="png", engine="kaleido")
        return BytesIO(img_bytes)
    except Exception as e:
        print(f"차트 변환 오류: {e}")
        return None

def generate_pdf_with_charts(company_name, export_sections, data, chart_data=None):
    """차트가 포함된 PDF 생성 함수"""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # 한글 폰트 등록
    font_available = register_korean_font()
    font_name = 'Korean' if font_available else 'Helvetica'
    
    # 색상 정의
    title_color = HexColor('#2E86AB')
    section_color = HexColor('#A23B72')
    text_color = HexColor('#333333')
    
    # 시작 위치
    y = height - 80
    margin_left = 50
    margin_right = width - 50
    
    # 제목 작성
    c.setFillColor(title_color)
    c.setFont(font_name, 20)
    title = f"📊 {company_name} 기업분석 리포트"
    c.drawString(margin_left, y, title)
    
    # 제목 밑줄
    c.setStrokeColor(title_color)
    c.setLineWidth(2)
    c.line(margin_left, y-5, margin_right, y-5)
    
    y -= 60
    
    # 각 섹션 작성
    for section in export_sections:
        # 페이지 넘김 체크
        if y < 200:  # 차트 공간을 위해 여유 확보
            c.showPage()
            y = height - 80
        # 섹션 제목
        c.setFillColor(section_color)
        c.setFont(font_name, 16)
        section_title = f"🔹 {section}"
        c.drawString(margin_left, y, section_title)
        y -= 30
        
        # 섹션별 차트 삽입
        if section == "재무 시각화" and chart_data:
            try:
                fig = create_financial_chart()
                img_buffer = plotly_to_image(fig)
                if img_buffer:
                    img = ImageReader(img_buffer)
                    c.drawImage(img, margin_left, y-300, width=400, height=250)
                    y -= 320
            except Exception as e:
                c.setFillColor(text_color)
                c.setFont(font_name, 12)
                c.drawString(margin_left + 20, y, f"• 차트 생성 오류: {str(e)}")
                y -= 20
        
        elif section == "주가 그래프" and chart_data:
            try:
                fig = create_stock_chart(
                    chart_data.get('stock_date', []), 
                    chart_data.get('stock_price', [])
                )
                img_buffer = plotly_to_image(fig)
                if img_buffer:
                    img = ImageReader(img_buffer)
                    c.drawImage(img, margin_left, y-300, width=400, height=250)
                    y -= 320
            except Exception as e:
                c.setFillColor(text_color)
                c.setFont(font_name, 12)
                c.drawString(margin_left + 20, y, f"• 차트 생성 오류: {str(e)}")
                y -= 20
        
        elif section == "예측 주가" and chart_data:
            try:
                fig = create_forecast_chart(
                    chart_data.get('forecast_date', []), 
                    chart_data.get('forecast_price', [])
                )
                img_buffer = plotly_to_image(fig)
                if img_buffer:
                    img = ImageReader(img_buffer)
                    c.drawImage(img, margin_left, y-300, width=400, height=250)
                    y -= 320
            except Exception as e:
                c.setFillColor(text_color)
                c.setFont(font_name, 12)
                c.drawString(margin_left + 20, y, f"• 차트 생성 오류: {str(e)}")
                y -= 20
        
        # 섹션 텍스트 내용
        c.setFillColor(text_color)
        c.setFont(font_name, 12)
        
        section_data = data.get(section, [])
        for line in section_data:
            # 페이지 넘김 체크
            if y < 80:
                c.showPage()
                y = height - 80
            
            # 긴 텍스트 줄바꿈 처리
            if len(line) > 80:
                words = line.split(' ')
                current_line = ""
                for word in words:
                    if len(current_line + word) < 80:
                        current_line += word + " "
                    else:
                        c.drawString(margin_left + 20, y, f"• {current_line.strip()}")
                        y -= 18
                        current_line = word + " "
                        if y < 80:
                            c.showPage()
                            y = height - 80
                if current_line:
                    c.drawString(margin_left + 20, y, f"• {current_line.strip()}")
                    y -= 18
            else:
                c.drawString(margin_left + 20, y, f"• {line}")
                y -= 18
        
        y -= 30  # 섹션 간 간격
    
    # 페이지 하단에 생성 정보
    c.setFillColor(HexColor('#888888'))
    c.setFont(font_name, 10)
    footer_text = f"Generated by AI Analysis System | Company: {company_name}"
    c.drawString(margin_left, 30, footer_text)
    
    # PDF 저장
    c.save()
    
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# 기존 함수는 호환성을 위해 유지
def generate_pdf(company_name, export_sections, data):
    """기존 PDF 생성 함수 (차트 없음)"""
    return generate_pdf_with_charts(company_name, export_sections, data, None)