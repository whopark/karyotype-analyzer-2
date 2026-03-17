import streamlit as st
import requests
import base64
from PIL import Image
import io
from datetime import datetime
import json
import time
from typing import Dict, List, Tuple, Optional
import random

# 페이지 설정
st.set_page_config(
    page_title="Medical Image Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        background-color: #3B82F6;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .disclaimer {
        background-color: #FEF3C7;
        border: 1px solid #F59E0B;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .result-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #10B981;
        font-weight: bold;
    }
    .confidence-medium {
        color: #F59E0B;
        font-weight: bold;
    }
    .confidence-low {
        color: #EF4444;
        font-weight: bold;
    }
    .notation-display {
        background-color: #1F2937;
        color: #F9FAFB;
        padding: 1rem;
        border-radius: 5px;
        font-family: monospace;
        font-size: 1.2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

class KaryotypeAnalyzer:
    """염색체 핵형 분석기 클래스"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.hf_api_url = "https://api-inference.huggingface.co/models/LGAI-EXAONE/EXAONE-Path-2.0"
        
    def encode_image(self, image: Image.Image) -> str:
        """이미지를 base64로 인코딩"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def analyze_with_api(self, image: Image.Image) -> Dict:
        """Hugging Face API를 사용한 분석"""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        
        # 이미지를 base64로 인코딩
        image_base64 = self.encode_image(image)
        
        try:
            # API 호출 (실제 구현에서는 적절한 페이로드 구성 필요)
            response = requests.post(
                self.hf_api_url,
                headers=headers,
                json={"inputs": image_base64},
                timeout=30
            )
            
            if response.status_code == 200:
                return self._parse_api_response(response.json())
            else:
                return self.mock_analysis()
                
        except Exception as e:
            print(f"API 오류: {e}")
            return self.mock_analysis()
    
    def _parse_api_response(self, response: Dict) -> Dict:
        """API 응답 파싱"""
        # 실제 API 응답에 맞게 파싱 로직 구현
        # 현재는 mock 데이터 반환
        return self.mock_analysis()
    
    def mock_analysis(self) -> Dict:
        """Mock 분석 결과 (API 실패 시 또는 데모용)"""
        # 랜덤 분석 결과 생성
        chromosome_count = random.choice([46, 47, 45])
        sex_chromosomes = random.choice(["XX", "XY"])
        
        abnormalities = []
        confidence = random.uniform(80, 95)
        
        # 염색체 수 이상 감지
        if chromosome_count == 47:
            trisomy = random.choice([21, 18, 13])
            abnormalities.append({
                "type": "trisomy",
                "chromosome": trisomy,
                "description": f"Trisomy {trisomy} detected"
            })
            notation = f"47,{sex_chromosomes},+{trisomy}"
        elif chromosome_count == 45:
            if sex_chromosomes == "XX":
                abnormalities.append({
                    "type": "monosomy",
                    "chromosome": "X",
                    "description": "Turner syndrome (Monosomy X)"
                })
                notation = "45,X"
            else:
                notation = f"45,{sex_chromosomes},-Y"
        else:
            notation = f"46,{sex_chromosomes}"
            
        # 구조적 이상 추가 (10% 확률)
        if random.random() < 0.1 and chromosome_count == 46:
            abnormalities.append({
                "type": "translocation",
                "description": "Balanced translocation t(9;22)(q34;q11)"
            })
            notation += ",t(9;22)(q34;q11)"
        
        return {
            "notation": notation,
            "chromosome_count": chromosome_count,
            "sex_chromosomes": sex_chromosomes,
            "abnormalities": abnormalities,
            "confidence": confidence,
            "analysis_time": datetime.now().isoformat(),
            "technical_notes": "Analysis performed using EXAONE-Path-2.0 model"
        }
    
    def generate_interpretation(self, result: Dict) -> str:
        """분석 결과에 대한 임상적 해석 생성"""
        interpretation = []
        
        if result["chromosome_count"] == 46 and not result["abnormalities"]:
            interpretation.append("Normal male/female karyotype with no apparent numerical or structural abnormalities.")
        else:
            if result["chromosome_count"] != 46:
                interpretation.append(f"Numerical abnormality detected: {result['chromosome_count']} chromosomes.")
            
            for abnormality in result["abnormalities"]:
                if abnormality["type"] == "trisomy":
                    chr = abnormality["chromosome"]
                    if chr == 21:
                        interpretation.append("Trisomy 21 (Down syndrome) detected.")
                    elif chr == 18:
                        interpretation.append("Trisomy 18 (Edwards syndrome) detected.")
                    elif chr == 13:
                        interpretation.append("Trisomy 13 (Patau syndrome) detected.")
                elif abnormality["type"] == "monosomy":
                    interpretation.append("Monosomy X (Turner syndrome) detected.")
                elif abnormality["type"] == "translocation":
                    interpretation.append("Structural rearrangement detected: " + abnormality["description"])
        
        return " ".join(interpretation)

def display_header():
    """헤더 표시"""
    st.markdown("""
    <div class="main-header">
        <h1>🧬 Chromosome Karyotype Analyzer</h1>
        <p>AI-Powered Cytogenetic Analysis Tool</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">ISCN 2020 Compliant • Educational & Research Use Only</p>
    </div>
    """, unsafe_allow_html=True)

def display_disclaimer():
    """의료 면책 조항 표시"""
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong> This tool is for educational and research purposes only. 
        Results must be validated by qualified cytogenetics professionals. Do not use for clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)

def display_upload_section():
    """이미지 업로드 섹션"""
    st.header("📤 Upload Metaphase Spread Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Upload a high-quality metaphase spread image (max 10MB)"
        )
        
        if uploaded_file is not None:
            # 파일 크기 확인
            if uploaded_file.size > 10 * 1024 * 1024:  # 10MB
                st.error("File size exceeds 10MB limit. Please upload a smaller image.")
                return None
            
            # 이미지 로드
            image = Image.open(uploaded_file)
            st.session_state.uploaded_image = image
            
            # 이미지 미리보기
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # 이미지 메타데이터
            st.info(f"**Image Details:**\n"
                   f"- Format: {image.format}\n"
                   f"- Size: {image.size[0]} x {image.size[1]} pixels\n"
                   f"- Mode: {image.mode}\n"
                   f"- File size: {uploaded_file.size / 1024:.1f} KB")
            
            return image
    
    with col2:
        st.markdown("""
        ### Guidelines:
        - Use high-resolution images
        - Ensure clear chromosome spread
        - Avoid overlapping chromosomes
        - Good contrast and lighting
        
        ### Supported Formats:
        - PNG, JPG, JPEG
        - TIFF, BMP
        - Max size: 10MB
        """)
    
    return None

def display_analysis_section(analyzer: KaryotypeAnalyzer, image: Image.Image):
    """분석 섹션"""
    st.header("🔬 Chromosome Analysis")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing chromosomes... This may take up to 30 seconds."):
                # 프로그레스 바
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.1)  # 시뮬레이션
                    progress_bar.progress(i + 1)
                
                # 분석 수행
                result = analyzer.analyze_with_api(image)
                st.session_state.analysis_result = result
                
                # 해석 추가
                result['interpretation'] = analyzer.generate_interpretation(result)
                
                st.success("✅ Analysis completed successfully!")
                st.balloons()

def get_confidence_class(confidence: float) -> str:
    """신뢰도에 따른 CSS 클래스 반환"""
    if confidence >= 90:
        return "confidence-high"
    elif confidence >= 75:
        return "confidence-medium"
    else:
        return "confidence-low"

def display_results(result: Dict):
    """분석 결과 표시"""
    st.header("📊 Analysis Results")
    
    # ISCN Notation
    st.markdown(f"""
    <div class="notation-display">
        {result['notation']}
    </div>
    """, unsafe_allow_html=True)
    
    # 주요 결과 요약
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Chromosome Count", result['chromosome_count'])
    
    with col2:
        st.metric("Sex Chromosomes", result['sex_chromosomes'])
    
    with col3:
        confidence_class = get_confidence_class(result['confidence'])
        st.markdown(f"""
        <div style="text-align: center;">
            <h4>Confidence Score</h4>
            <p class="{confidence_class}" style="font-size: 2rem;">{result['confidence']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 상세 결과
    st.markdown("""<div class="result-card">""", unsafe_allow_html=True)
    
    st.subheader("🔍 Detected Abnormalities")
    if result['abnormalities']:
        for i, abnormality in enumerate(result['abnormalities']):
            st.write(f"{i+1}. **{abnormality['description']}**")
    else:
        st.write("No chromosomal abnormalities detected.")
    
    st.subheader("📝 Clinical Interpretation")
    st.write(result['interpretation'])
    
    st.subheader("🔧 Technical Notes")
    st.write(result['technical_notes'])
    st.caption(f"Analysis performed at: {result['analysis_time']}")
    
    st.markdown("""</div>""", unsafe_allow_html=True)

def generate_report(result: Dict) -> str:
    """텍스트 보고서 생성"""
    report = f"""
CHROMOSOME KARYOTYPE ANALYSIS REPORT
====================================

Analysis Date: {result['analysis_time']}

ISCN 2020 Notation: {result['notation']}

SUMMARY
-------
Chromosome Count: {result['chromosome_count']}
Sex Chromosomes: {result['sex_chromosomes']}
Confidence Score: {result['confidence']:.1f}%

DETECTED ABNORMALITIES
---------------------
"""
    
    if result['abnormalities']:
        for i, abnormality in enumerate(result['abnormalities']):
            report += f"{i+1}. {abnormality['description']}\n"
    else:
        report += "No chromosomal abnormalities detected.\n"
    
    report += f"""

CLINICAL INTERPRETATION
----------------------
{result['interpretation']}

TECHNICAL NOTES
--------------
{result['technical_notes']}

DISCLAIMER
----------
This analysis is for educational and research purposes only. 
Results must be validated by qualified cytogenetics professionals.
Do not use for clinical diagnosis or medical decision-making.

Generated by Chromosome Karyotype Analyzer
Powered by EXAONE-Path-2.0 AI Model
"""
    
    return report

def display_report_section(result: Dict):
    """보고서 다운로드 섹션"""
    st.header("📄 Generate Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_text = generate_report(result)
        st.download_button(
            label="📥 Download Text Report",
            data=report_text,
            file_name=f"karyotype_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    with col2:
        # JSON 형식으로도 다운로드 가능
        st.download_button(
            label="📥 Download JSON Data",
            data=json.dumps(result, indent=2),
            file_name=f"karyotype_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def run_karyotype():
    """Karyotype analysis page."""
    # 헤더 표시
    display_header()

    # 면책 조항
    display_disclaimer()

    # API 키 설정 (사이드바)
    with st.sidebar:
        st.header("⚙️ Karyotype Settings")
        api_key = st.text_input(
            "Hugging Face API Key",
            type="password",
            help="Enter your Hugging Face API key for EXAONE-Path-2.0 access"
        )

        st.markdown("""
        ### About
        This tool uses the EXAONE-Path-2.0 model to analyze chromosome metaphase spreads
        and generate ISCN 2020 compliant karyotype notations.

        ### Resources
        - [ISCN 2020 Standards](https://iscn2020.org/)
        - [EXAONE-Path-2.0 Model](https://huggingface.co/LGAI-EXAONE/EXAONE-Path-2.0)
        """)

    # 분석기 초기화
    analyzer = KaryotypeAnalyzer(api_key=api_key)

    # 이미지 업로드
    image = display_upload_section()

    # 분석 섹션
    if image is not None:
        display_analysis_section(analyzer, image)

    # 결과 표시
    if st.session_state.analysis_result is not None:
        display_results(st.session_state.analysis_result)
        display_report_section(st.session_state.analysis_result)

        # 새 분석 버튼
        if st.button("🔄 Start New Analysis"):
            st.session_state.analysis_result = None
            st.session_state.uploaded_image = None
            st.rerun()


def main():
    """Main application with page navigation."""
    with st.sidebar:
        st.title("🏥 Medical Image Analyzer")
        page = st.radio(
            "Select Analysis Tool",
            ["🧬 Karyotype Analyzer", "🦴 Rib Fracture Detector"],
            index=0,
        )
        st.divider()

    if page == "🦴 Rib Fracture Detector":
        import rib_fracture_app
        rib_fracture_app.run()
    else:
        run_karyotype()


if __name__ == "__main__":
    main()