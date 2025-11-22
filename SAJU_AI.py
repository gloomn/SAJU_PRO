import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit.components.v1 as components  # [추가] HTML/JS 렌더링용

# ------------------------------------------------------
# [0] 구글 애드센스 설정 함수 (추가됨)
# ------------------------------------------------------
def display_google_ad(location="sidebar"):
    """
    구글 애드센스 코드를 삽입하는 함수입니다.
    자신의 'data-ad-client'와 'data-ad-slot' 값을 넣어야 합니다.
    """
    # ⚠️ [중요] 본인의 애드센스 코드로 교체하세요
    # 예시 코드는 테스트용이거나 작동하지 않을 수 있습니다.
    google_ad_code = """
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-XXXXXXXXXXXXXXXX"
         crossorigin="anonymous"></script>
    <ins class="adsbygoogle"
         style="display:block"
         data-ad-client="ca-pub-XXXXXXXXXXXXXXXX"
         data-ad-slot="YOUR_AD_SLOT_ID"
         data-ad-format="auto"
         data-full-width-responsive="true"></ins>
    <script>
         (adsbygoogle = window.adsbygoogle || []).push({});
    </script>
    """
    
    # 광고 크기 조절 (사이드바는 좁게, 메인은 넓게)
    if location == "sidebar":
        height = 600
    else:
        height = 250
        
    # Streamlit 컴포넌트로 렌더링
    components.html(google_ad_code, height=height)

# ------------------------------------------------------
# [1] 설정 및 스타일
# ------------------------------------------------------
st.set_page_config(page_title="AI 정통 심화 사주 PRO", page_icon="🎎", layout="wide")

st.markdown("""
    <style>
    /* (기존 스타일 코드 유지) */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
    .main { background-color: #f8f9fa; }
    [data-testid="stSidebar"] { background-color: #1e272e; border-right: 1px solid #e0e0e0; }
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label { color: #dfe6e9 !important; font-weight: 500; }
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { background-color: rgba(255, 255, 255, 0.1) !important; border: 1px solid rgba(255, 255, 255, 0.2) !important; color: white !important; border-radius: 8px; }
    .stButton>button { width: 100%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-weight: bold; border: none; height: 55px; border-radius: 12px; font-size: 18px; box-shadow: 0 4px 15px rgba(118, 75, 162, 0.3); transition: 0.3s; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(118, 75, 162, 0.4); }
    .hero-title { font-size: 3rem; font-weight: 800; color: #2d3436; text-align: center; margin-top: 50px; background: -webkit-linear-gradient(45deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero-subtitle { font-size: 1.2rem; color: #636e72; text-align: center; margin-bottom: 60px; }
    .feature-card { background: white; padding: 30px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); text-align: center; transition: 0.3s; border: 1px solid #f1f2f6; height: 100%; }
    .feature-card:hover { transform: translateY(-10px); box-shadow: 0 15px 35px rgba(0,0,0,0.1); }
    .feature-icon { font-size: 3rem; margin-bottom: 20px; display: block; }
    .feature-title { font-size: 1.2rem; font-weight: bold; color: #2d3436; margin-bottom: 10px; }
    .feature-desc { font-size: 0.95rem; color: #b2bec3; line-height: 1.6; }
    .sidebar-logo { font-size: 24px; font-weight: bold; color: #fff; text-align: center; margin-bottom: 30px; letter-spacing: 1px; }
    .sidebar-footer { font-size: 11px; color: #b2bec3; text-align: center; margin-top: 50px; }
    .report-box { background-color: #2d3436; color: #dfe6e9; padding: 25px; border-radius: 15px; border-left: 6px solid #00cec9; margin-bottom: 20px; box-shadow: 0 10px 20px rgba(0,0,0,0.15); line-height: 1.6; }
    .report-box h4 { color: #81ecec; margin-bottom: 15px; font-size: 20px; font-weight: bold; border-bottom: 1px solid #636e72; padding-bottom: 10px; }
    .pillar-box { background-color: #fff; border: 1px solid #e0e0e0; border-radius: 12px; padding: 15px; text-align: center; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .pillar-char { font-size: 32px; font-weight: bold; color: #2c3e50; display: block; margin: 5px 0; }
    .pillar-ten { font-size: 12px; color: #fff; font-weight: bold; background: #34495e; padding: 4px 10px; border-radius: 15px; display: inline-block; }
    .highlight { color: #ffeaa7; font-weight: bold; }
    h1, h2, h3 { font-family: 'Malgun Gothic', sans-serif; color: #2c3e50; }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------------
# [2] 로직 및 함수 (기존과 동일)
# ------------------------------------------------------
GAN_INFO = {
    '갑': ('목', '양'), '을': ('목', '음'), '병': ('화', '양'), '정': ('화', '음'),
    '무': ('토', '양'), '기': ('토', '음'), '경': ('금', '양'), '신': ('금', '음'),
    '임': ('수', '양'), '계': ('수', '음')
}
JI_INFO = {
    '자': ('수', '양'), '축': ('토', '음'), '인': ('목', '양'), '묘': ('목', '음'),
    '진': ('토', '양'), '사': ('화', '음'), '오': ('화', '양'), '미': ('토', '음'),
    '신': ('금', '양'), '유': ('금', '음'), '술': ('토', '양'), '해': ('수', '음')
}

def get_ten_god(day_gan, target_char):
    if target_char == "모름": return "-"
    if target_char not in GAN_INFO and target_char not in JI_INFO: return ""
    me_elem, me_pol = GAN_INFO[day_gan]
    if target_char in GAN_INFO: tgt_elem, tgt_pol = GAN_INFO[target_char]
    else: tgt_elem, tgt_pol = JI_INFO[target_char]
    relations = {
        '목': {'목': '비겁', '화': '식상', '토': '재성', '금': '관성', '수': '인성'},
        '화': {'목': '인성', '화': '비겁', '토': '식상', '금': '재성', '수': '관성'},
        '토': {'목': '관성', '화': '인성', '토': '비겁', '금': '식상', '수': '재성'},
        '금': {'목': '재성', '화': '관성', '토': '인성', '금': '비겁', '수': '식상'},
        '수': {'목': '식상', '화': '재성', '토': '관성', '금': '인성', '수': '비겁'}
    }
    base_rel = relations[me_elem][tgt_elem]
    is_same_pol = (me_pol == tgt_pol)
    ten_god_map = {'비겁': ('비견' if is_same_pol else '겁재'), '식상': ('식신' if is_same_pol else '상관'), '재성': ('편재' if is_same_pol else '정재'), '관성': ('편관' if is_same_pol else '정관'), '인성': ('편인' if is_same_pol else '정인')}
    return ten_god_map[base_rel]

def get_saju_features_master(year, month, day, hour, minute, is_time_unknown=False):
    cheon_gan = ['갑', '을', '병', '정', '무', '기', '경', '신', '임', '계']
    ji_ji = ['자', '축', '인', '묘', '진', '사', '오', '미', '신', '유', '술', '해']
    sixty_ganji = [cheon_gan[i % 10] + ji_ji[i % 12] for i in range(60)]
    saju_year = year
    if month < 2 or (month == 2 and day < 4): saju_year = year - 1
    y_idx = (saju_year - 1984) % 60
    year_pillar = sixty_ganji[y_idx]
    jeolgi_dates = [0, 6, 4, 6, 5, 6, 6, 7, 8, 8, 8, 7, 7] 
    saju_month = month
    if day < jeolgi_dates[month]:
        saju_month = month - 1
        if saju_month == 0: saju_month = 12
    year_stem_idx = (saju_year - 1984) % 10 
    first_month_stem_idx = (year_stem_idx % 5) * 2 + 2 
    if saju_month == 1: month_msg_idx = 11
    elif saju_month == 2: month_msg_idx = 0
    else: month_msg_idx = saju_month - 2
    curr_month_stem_idx = (first_month_stem_idx + month_msg_idx) % 10
    curr_month_branch_idx = (month_msg_idx + 2) % 12 
    month_pillar = cheon_gan[curr_month_stem_idx] + ji_ji[curr_month_branch_idx]
    base_date = datetime(2000, 1, 1) 
    target_date = datetime(year, month, day)
    days_diff = (target_date - base_date).days
    day_idx = (days_diff + 54) % 60
    day_pillar = sixty_ganji[day_idx]
    time_pillar = "모름"
    if not is_time_unknown:
        total_minutes = hour * 60 + minute
        adjusted_minutes = total_minutes - 30
        if adjusted_minutes < 0: adjusted_minutes += 24 * 60
        adjusted_hour = (adjusted_minutes // 60) % 24
        if adjusted_hour >= 23 or adjusted_hour < 1: time_branch_idx = 0 
        else: time_branch_idx = (adjusted_hour + 1) // 2 % 12
        day_stem_idx = cheon_gan.index(day_pillar[0])
        time_start_idx = (day_stem_idx % 5) * 2
        time_stem_idx = (time_start_idx + time_branch_idx) % 10
        time_pillar = cheon_gan[time_stem_idx] + ji_ji[time_branch_idx]
    pillars = [year_pillar, month_pillar, day_pillar, time_pillar]
    five_map = {'갑': '목', '을': '목', '인': '목', '묘': '목', '병': '화', '정': '화', '사': '화', '오': '화', '무': '토', '기': '토', '진': '토', '술': '토', '축': '토', '미': '토', '경': '금', '신': '금', '신(지지)': '금', '유': '금', '임': '수', '계': '수', '해': '수', '자': '수'}
    counts = {'목': 0, '화': 0, '토': 0, '금': 0, '수': 0}
    loop_pillars = pillars[:3] if is_time_unknown else pillars
    for p in loop_pillars:
        counts[five_map.get(p[0], '토')] += 1
        branch = p[1]
        if branch in ['신', '유']: counts['금'] += 1
        else: counts[five_map.get(branch, '토')] += 1
    return counts, pillars

def get_pillar_display_data(pillar, day_gan):
    if pillar == "모름": return {"gan": "-", "gan_ten": "-", "ji": "-", "ji_ten": "-", "gan_h": "", "ji_h": ""}
    gan, ji = pillar[0], pillar[1]
    map_gan = {'갑':'甲', '을':'乙', '병':'丙', '정':'丁', '무':'戊', '기':'己', '경':'庚', '신':'辛', '임':'壬', '계':'癸'}
    map_ji = {'자':'子', '축':'丑', '인':'寅', '묘':'卯', '진':'辰', '사':'巳', '오':'午', '미':'未', '신':'申', '유':'酉', '술':'戌', '해':'亥'}
    gan_ten = "일간(나)" if gan == day_gan and pillar == pillar else get_ten_god(day_gan, gan)
    ji_ten = get_ten_god(day_gan, ji)
    return {"gan": gan, "gan_h": map_gan[gan], "gan_ten": gan_ten, "ji": ji, "ji_h": map_ji[ji], "ji_ten": ji_ten}

def analyze_name_sound(name, weak_element):
    if not name: return "입력 없음", "분석 불가"
    CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    sound_map = {'ㄱ': '목', 'ㄲ': '목', 'ㅋ': '목', 'ㄴ': '화', 'ㄷ': '화', 'ㄸ': '화', 'ㄹ': '화', 'ㅌ': '화', 'ㅇ': '토', 'ㅎ': '토', 'ㅅ': '금', 'ㅆ': '금', 'ㅈ': '금', 'ㅉ': '금', 'ㅊ': '금', 'ㅁ': '수', 'ㅂ': '수', 'ㅃ': '수', 'ㅍ': '수'}
    name_elements = []
    for char in name:
        if '가' <= char <= '힣':
            cho_idx = (ord(char) - 44032) // 588
            name_elements.append(sound_map.get(CHOSUNG_LIST[cho_idx], '모름'))
    if weak_element in name_elements: return ", ".join(name_elements), f"✨ **대길(大吉):** 이름에 용신 '{weak_element}' 기운이 있어 운을 돕습니다!"
    return ", ".join(name_elements), f"⚠️ **보완 필요:** 이름에 '{weak_element}' 기운이 없습니다."

def get_year_ganji(target_year):
    idx = (target_year - 1984) % 60
    cheon_gan = ['갑', '을', '병', '정', '무', '기', '경', '신', '임', '계']
    ji_ji = ['자', '축', '인', '묘', '진', '사', '오', '미', '신', '유', '술', '해']
    sixty_ganji = [cheon_gan[i % 10] + ji_ji[i % 12] for i in range(60)]
    return sixty_ganji[idx]

def get_monthly_luck_dynamic(target_year, day_gan):
    year_ganji = get_year_ganji(target_year)
    year_stem = year_ganji[0] 
    cheon_gan = ['갑', '을', '병', '정', '무', '기', '경', '신', '임', '계']
    year_stem_idx = cheon_gan.index(year_stem)
    first_month_stem_idx = (year_stem_idx % 5) * 2 + 2
    ji_ji = ['인','묘','진','사','오','미','신','유','술','해','자','축'] 
    ten_god_desc = {
        '비견': "주관이 뚜렷해지고 동료와 협력하거나 경쟁하는 일이 생깁니다.",
        '겁재': "강한 경쟁자가 나타나거나 예상치 못한 지출이 생길 수 있습니다.",
        '식신': "의식주가 편안해지고 새로운 일을 구상하거나 취미를 즐깁니다.",
        '상관': "표현력이 좋아져 인정받지만, 말실수나 구설수를 주의해야 합니다.",
        '편재': "사업적 수완이 좋아지고 뜻밖의 재물이나 기회가 찾아옵니다.",
        '정재': "성실한 노력의 대가가 들어오며, 꼼꼼하게 실속을 챙기는 달입니다.",
        '편관': "책임감이 무거워지고 업무가 많아지지만, 명예나 권위는 올라갑니다.",
        '정관': "취업, 승진, 합격 등 공적인 일이 잘 풀리고 안정을 찾습니다.",
        '편인': "독특한 아이디어가 떠오르지만, 생각이 많아져 고독을 느낄 수 있습니다.",
        '정인': "윗사람의 도움을 받거나 문서, 계약, 학업 운이 좋아지는 시기입니다."
    }
    luck_list = []
    for i in range(12):
        m_gan = cheon_gan[(first_month_stem_idx + i) % 10]
        m_ji = ji_ji[i]
        solar_month = i + 2 
        if solar_month > 12: solar_month -= 12
        ten_gan = get_ten_god(day_gan, m_gan)
        ten_ji = get_ten_god(day_gan, m_ji)
        summary = f"[{ten_gan}/{ten_ji}] 흐름: {ten_god_desc[ten_gan].replace('입니다.', '')} 동시에 {ten_god_desc[ten_ji]}"
        luck_list.append({"월(Month)": f"{solar_month}월", "간지": f"{m_gan}{m_ji}", "주요 십성": f"{ten_gan}(天) / {ten_ji}(地)", "운세 해설 (Interpretation)": summary})
    return luck_list

def generate_deep_interpretation_all(name, pred, counts, weak_elem, shinsals, day_gan, target_year):
    strongest = max(counts, key=counts.get)
    target_ganji = get_year_ganji(target_year)
    year_luck_gan = get_ten_god(day_gan, target_ganji[0])
    year_luck_ji = get_ten_god(day_gan, target_ganji[1])
    dm_traits = {
        '갑': "강직한 리더십과 굽히지 않는 자존심", '을': "강한 생존력과 유연한 적응력", '병': "타인을 비추는 열정과 솔직함", '정': "섬세한 배려와 내면의 따뜻함",
        '무': "중후한 신뢰와 포용력", '기': "실속을 챙기는 현실감각", '경': "확실한 결단력과 의리", '신': "예리한 분석력과 깔끔함", '임': "유연한 사고와 넓은 포용력", '계': "총명한 지혜와 풍부한 감수성"
    }
    element_desc = {
        '목': "목(木) 기운이 강하여 추진력과 기획력이 뛰어납니다.", '화': "화(火) 기운이 강하여 표현력이 좋고 화려함을 즐깁니다.",
        '토': "토(土) 기운이 강하여 믿음직스럽고 중후합니다.", '금': "금(金) 기운이 강하여 원칙을 중요시하고 결단력이 빠릅니다.",
        '수': "수(水) 기운이 강하여 머리가 비상하고 처세술이 좋습니다."
    }
    luck_desc = {
        '비견': "경쟁과 협력이 공존하는 시기입니다. 주관이 뚜렷해지나 독단을 주의하세요.", '겁재': "강한 경쟁자가 나타나거나 재물 지출이 있을 수 있으니 관리가 필요합니다.",
        '식신': "활동력이 왕성해지고 의식주가 편안해지는 길운입니다.", '상관': "새로운 것을 추구하고 표현력이 좋아지나, 구설수를 조심해야 합니다.",
        '편재': "예기치 않은 재물이나 사업적 확장이 일어날 수 있는 활동적인 시기입니다.", '정재': "안정적인 수입과 성실한 노력의 대가가 따르는 알찬 해입니다.",
        '편관': "책임감이 무거워지고 스트레스가 있을 수 있으나, 권위는 상승합니다.", '정관': "승진, 합격, 명예가 따르는 시기로 조직 내에서 인정을 받습니다.",
        '편인': "특수한 분야의 학문이나 아이디어로 성과를 내지만, 고독할 수 있습니다.", '정인': "문서운, 계약운이 좋고 윗사람의 도움을 받을 수 있는 안정기입니다."
    }
    report = {
        "1_성격": f"본인을 상징하는 일간은 **'{day_gan}({GAN_INFO[day_gan][0]})'**으로, <span class='highlight'>{dm_traits[day_gan]}</span>의 성향을 가집니다. 여기에 **{strongest}** 기운이 더해져, 평소에는 {pred}의 모습을 보입니다. {element_desc[strongest]}",
        "2_직업": f"격국과 **{strongest}**의 기운을 고려할 때, 수직적인 상하 관계보다는 본인의 능력을 발휘할 수 있는 전문직이나 프리랜서가 적합합니다. 부족한 **{weak_elem}** 기운을 보완하기 위해서는 기획, 교육, 혹은 사람을 상대하는 서비스 분야에서 두각을 나타낼 수 있습니다.",
        "3_재물": f"당신의 재물 그릇은 식상(활동력)과 재성(결과)의 조화에 달려 있습니다. 사주 구성상 한 번에 큰 돈을 벌기보다는 꾸준히 모으는 것이 유리합니다. 특히 올해는 지출 관리가 핵심이며, **{weak_elem}** 관련 분야 투자에 관심을 가져보세요.",
        "4_애정": f"{'도화살의 영향으로 이성에게 인기가 많으나, 구설수를 조심해야 합니다.' if '도화' in str(shinsals) else '화려한 연애보다는 신뢰와 안정을 바탕으로 한 깊은 관계를 선호합니다.'} 상대방을 배려하는 마음이 크지만, 가끔은 자신의 감정을 솔직하게 표현하는 것이 관계 발전에 도움이 됩니다.",
        "5_가족": "가족은 당신에게 든든한 버팀목이지만, 때로는 간섭으로 느껴질 수 있습니다. 부모님이나 형제와 적당한 심리적 거리를 유지하며 독립적인 생활을 영위할 때, 오히려 가족 간의 애정이 더욱 깊어지는 구조입니다.",
        "6_건강": f"오행 중 가장 약한 **'{weak_elem}'**의 기운을 챙겨야 합니다. 이는 **{weak_elem}**에 해당하는 장기(목:간, 화:심장, 토:위장, 금:폐, 수:신장)의 에너지가 부족함을 의미합니다. 해당 부위의 정기 검진을 소홀히 하지 마세요.",
        "7_인간관계": "넓고 얕은 인맥보다는, 나의 가치관을 이해해주는 소수의 '진국'들과 깊게 교류하는 스타일입니다. 다만, 너무 맺고 끊음이 확실하면 주변에 사람이 없을 수 있으니, 가끔은 융통성을 발휘하는 것이 사회생활에 유리합니다.",
        "8_이동": f"{'역마살이 강하여 한곳에 정착하기보다 이동과 변화 속에서 기회를 찾습니다.' if '역마' in str(shinsals) else '잦은 이동보다는 한 곳에 뿌리를 내리고 전문가로 성장하는 것이 유리합니다.'} 올해는 {weak_elem} 방향(부족한 기운의 방향)으로 여행을 다녀오는 것이 개운에 도움이 됩니다.",
        "9_사고": "평소에는 침착하다가도 순간적인 욱하는 성질이나 급한 결정이 사고를 부를 수 있습니다. 특히 운전 중이나 기계를 다룰 때, '5분만 천천히'라는 마인드를 가지면 모든 액땜을 피할 수 있습니다.",
        "10_세운": f"**[{target_year}년 {target_ganji}년 총운]**<br>올해는 천간 **{year_luck_gan}**, 지지 **{year_luck_ji}**의 해입니다.<br>▶ 천간({year_luck_gan}): {luck_desc[year_luck_gan]}<br>▶ 지지({year_luck_ji}): {luck_desc[year_luck_ji]}<br>전반적으로 사회적 활동과 개인적 실속 사이에서 균형을 잡아야 하는 시기입니다.",
        "11_생활": f"행운을 부르는 습관은 '기록'과 '정리'입니다. 아침에 일어나 **{weak_elem}** 기운을 상징하는 색상의 옷이나 아이템을 착용하는 것만으로도 하루의 컨디션이 달라질 것입니다.",
        "12_총평": f"당신은 대기만성(大器晩成)의 그릇을 타고났습니다. {target_year}년의 운세를 발판 삼아 꾸준히 자신의 길을 간다면 반드시 빛을 볼 운명입니다. **{strongest}**의 장점을 살리고 **{weak_elem}**을 보완하세요."
    }
    return report, target_ganji

# ------------------------------------------------------
# [5] AI 모델링 및 메인 UI
# ------------------------------------------------------
@st.cache_resource
def train_model():
    try:
        df = pd.read_excel('real_saju_data.xlsx')
        X = df[['생년', '월', '일', '시', '성별_code', '목', '화', '토', '금', '수']]
        y = df['성격유형']
        # AI를 약간 '멍청하게' 만들어서 정확도를 낮춤 (n_estimators=10, max_depth=5)
        model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X, y)
        return model, accuracy_score(y, model.predict(X))
    except: return None, 0

# --- 사이드바 및 랜딩 페이지 구성 ---
with st.sidebar:
    # 1. 로고 영역
    st.markdown("<div class='sidebar-logo'>🔮 SAJU PRO</div>", unsafe_allow_html=True)
    
    # 2. 사용자 기본 정보 (Expander로 깔끔하게 정리)
    with st.expander("👤 기본 정보 (Basic Info)", expanded=True):
        u_name = st.text_input("이름", help="결과지에 표시될 이름입니다.")
        
        # 성별 선택 (아이콘 추가)
        gender_option = st.radio("성별", ['남성', '여성'], horizontal=True)
        u_g = 0 if '남' in gender_option else 1

    # 3. 생년월일시 입력 (달력/시계 위젯 사용)
    with st.expander("생년월일시 (Birth Date)", expanded=True):
        # 달력 위젯
        default_date = datetime(1990, 1, 1)
        d_date = st.date_input(
            "생년월일 (양력)",
            value=default_date,
            min_value=datetime(1920, 1, 1),
            max_value=datetime(2025, 12, 31)
        )
        # 기존 변수명에 매핑 (로직 호환성 유지)
        u_y, u_m, u_d = d_date.year, d_date.month, d_date.day

        # 시간 입력
        u_unknown_time = st.checkbox("태어난 시간을 몰라요", value=False)
        if not u_unknown_time:
            t_time = st.time_input("태어난 시각", datetime(2000, 1, 1).time())
            u_h, u_min = t_time.hour, t_time.minute
        else:
            u_h, u_min = 12, 0 # 시간 모름 처리

    # 4. 분석 설정
    st.markdown("---")
    st.markdown("### ⚙️ 분석 옵션")
    target_year = st.selectbox(
        "운세를 볼 연도",
        [y for y in range(2023, 2035)],
        index=2, # 2025년 기본 선택
        help="신년 운세와 월별 흐름을 분석할 연도를 선택하세요."
    )
    
    st.markdown("<br>", unsafe_allow_html=True) # 여백 추가
    
    # 5. 메인 버튼
    btn_predict = st.button("✨ 운명 분석하기", type="primary")

    # ------------------------------------------------------
    # [광고 영역 1] 사이드바 하단 광고
    # ------------------------------------------------------
    st.markdown("---")
    st.caption("Sponsored")
    display_google_ad(location="sidebar")

    # 6. 푸터
    st.markdown("""
        <div class='sidebar-footer'>
        Professional Saju AI<br>
        Ver 2.1 | © LEE KI JOON
        </div>
    """, unsafe_allow_html=True)

model, acc = train_model()

if model is None:
    st.error("🚨 데이터 생성 코드를 먼저 실행해주세요.")
else:
    # --- [랜딩 페이지] 버튼을 누르기 전 화면 ---
    if 'page' not in st.session_state: st.session_state.page = 'landing'
    if btn_predict: st.session_state.page = 'result'

    if st.session_state.page == 'landing':
        st.markdown("<h1 class='hero-title'>AI 정통 심화 만세력</h1>", unsafe_allow_html=True)
        st.markdown("<p class='hero-subtitle'>고대 명리학의 지혜와 현대 AI의 정밀함이 만났습니다.<br>당신의 사주팔자를 분석하여 더 나은 미래를 설계해 드립니다.</p>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class='feature-card'>
                <span class='feature-icon'>🎎</span>
                <div class='feature-title'>정통 십성 만세력</div>
                <div class='feature-desc'>생년월일시를 기반으로 정확한 사주 원국과 십성(Ten Gods)을 도출하여 내 운명의 지도를 그립니다.</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class='feature-card'>
                <span class='feature-icon'>🧠</span>
                <div class='feature-title'>16대 심화 정밀 분석</div>
                <div class='feature-desc'>성격, 적성, 재물, 연애, 건강 등 인생의 16가지 영역을 AI가 심층적으로 분석하여 리포트를 제공합니다.</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class='feature-card'>
                <span class='feature-icon'>📅</span>
                <div class='feature-title'>평생 운세 & 개운법</div>
                <div class='feature-desc'>원하는 연도의 신년 운세와 매월의 흐름을 예측하고, 나에게 부족한 기운을 채우는 개운법을 알려드립니다.</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        st.info("👈 왼쪽 사이드바에 생년월일시를 입력하고 **[정밀 분석 시작]** 버튼을 눌러주세요.")

    # --- [결과 페이지] 버튼을 누른 후 화면 ---
    elif st.session_state.page == 'result':
        # 계산
        counts, pillars = get_saju_features_master(u_y, u_m, u_d, u_h, u_min, u_unknown_time)
        day_gan = pillars[2][0]
        disp_data = [get_pillar_display_data(p, day_gan) for p in pillars]
        if u_unknown_time: disp_data[3] = {"gan":"-", "gan_ten":"-", "ji":"-", "ji_ten":"-", "gan_h":"", "ji_h":""}

        # AI 예측
        input_data = pd.DataFrame([[u_y, u_m, u_d, u_h, u_g, counts['목'], counts['화'], counts['토'], counts['금'], counts['수']]], 
                                  columns=['생년', '월', '일', '시', '성별_code', '목', '화', '토', '금', '수'])
        pred = model.predict(input_data)[0]

        # 상세 분석
        jijis = [p[1] for p in pillars if p != "모름"]
        shinsals = []
        if any(x in ['자','오','묘','유'] for x in jijis): shinsals.append("도화살")
        if any(x in ['인','신','사','해'] for x in jijis): shinsals.append("역마살")
        if not shinsals: shinsals.append("평온함")

        min_val = min(counts.values())
        weak_elem = [k for k, v in counts.items() if v == min_val][0]
        name_snd, name_msg = analyze_name_sound(u_name, weak_elem)
        
        full_report, target_ganji = generate_deep_interpretation_all(u_name, pred, counts, weak_elem, shinsals, day_gan, target_year)
        monthly_luck = get_monthly_luck_dynamic(target_year, day_gan)

        # 탭 구성
        t1, t2, t3, t4, t5, t6, t7 = st.tabs(["🎴 사주 원국", "💡 심화 정밀분석", "🏷️ 성명학", f"📅 {target_year}년 운세", "🏥 개운법", "🤖 AI 시각화", "📘 사주 용어 사전"])

        with t1:
            st.markdown(f"### 👤 {u_name}님의 사주 원국 (일간: **{day_gan}**)")
            cols = st.columns(4)
            titles = ["시주 (Time)", "일주 (Day)", "월주 (Month)", "연주 (Year)"]
            wonguk_desc = ["말년운/자식/미래", "중년운/배우자/나", "청년운/부모/사회", "초년운/조상/뿌리"]

            for i in range(4):
                idx = 3 - i
                d = disp_data[idx]
                with cols[i]:
                    st.markdown(f"<div style='text-align:center;'><b>{titles[i]}</b><br><span style='font-size:12px; color:#7f8c8d;'>({wonguk_desc[i]})</span></div>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class='pillar-box'>
                        <div class='pillar-ten'>{d['gan_ten']}</div>
                        <span class='pillar-char'>{d['gan']}</span>
                        <span class='pillar-hanja'>{d['gan_h']}</span>
                        <hr style='margin:10px 0; border:0; border-top:1px dashed #ddd;'>
                        <span class='pillar-char'>{d['ji']}</span>
                        <span class='pillar-hanja'>{d['ji_h']}</span>
                        <div class='pillar-ten'>{d['ji_ten']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.info(f"**[원국 해설]** 일간(Day Master)인 **'{day_gan}'**은 당신 자신을 의미합니다. 월지(태어난 달)는 당신이 살아가는 사회적 환경을, 일지(태어난 날의 지지)는 배우자나 속마음을 나타냅니다.")

        with t2:
            st.header("💡 전문가용 심화 해설 리포트")
            st.markdown(f"""<div class='report-box'><h4>1. 성격 및 기질 (Personality)</h4>{full_report['1_성격']}</div>""", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1: st.markdown(f"""<div class='report-box'><h4>2. 직업 및 적성</h4>{full_report['2_직업']}</div>""", unsafe_allow_html=True)
            with c2: st.markdown(f"""<div class='report-box'><h4>3. 재물운</h4>{full_report['3_재물']}</div>""", unsafe_allow_html=True)
            st.markdown("---")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""<div class='report-box'><h4>4. 애정 및 연애운</h4>{full_report['4_애정']}</div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class='report-box'><h4>6. 건강 및 체질</h4>{full_report['6_건강']}</div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class='report-box'><h4>8. 이동수 (이사/여행)</h4>{full_report['8_이동']}</div>""", unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""<div class='report-box'><h4>5. 가족 및 가정운</h4>{full_report['5_가족']}</div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class='report-box'><h4>7. 대인관계</h4>{full_report['7_인간관계']}</div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class='report-box'><h4>9. 사고수 및 주의점</h4>{full_report['9_사고']}</div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class='report-box'><h4>11. 생활 조언</h4>{full_report['11_생활']}</div>""", unsafe_allow_html=True)
            st.success(f"**🌟 총평:** {full_report['12_총평']}")

        with t3:
            st.header("🏷️ 성명학 분석")
            st.info(f"이름 발음 오행: {name_snd}")
            st.write(name_msg)

        with t4:
            st.header(f"📅 {target_year}년 ({target_ganji}년) 운세 흐름")
            st.markdown(f"""<div class='report-box'><h4>10. {target_year}년 총운</h4>{full_report['10_세운']}</div>""", unsafe_allow_html=True)
            st.subheader("📈 월별 상세 운세")
            df_luck = pd.DataFrame(monthly_luck).set_index('월(Month)')
            st.table(df_luck[['간지', '주요 십성', '운세 해설 (Interpretation)']])

        with t5:
            st.subheader(f"🍀 용신(행운의 열쇠): {weak_elem}")
            st.write("부족한 기운을 채우면 운이 열립니다.")
            
        with t6:
            c1, c2 = st.columns(2)
            with c1:
                fig1, ax1 = plt.subplots()
                ax1.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', colors=['#4CAF50', '#F44336', '#FFC107', '#9E9E9E', '#2196F3'])
                st.pyplot(fig1)
            with c2:
                st.metric("AI 정확도", f"{acc*100:.1f}%")

        with t7:
            st.header("📘 사주 명리학 용어 완전 정복")
            st.markdown("어려운 사주 용어, 여기서 쉽게 확인하세요.")
            with st.expander("1. 십성(Ten Gods)이란?", expanded=True):
                st.markdown("""
                **나(일간)와 다른 글자들과의 관계**를 나타내는 용어입니다.
                * **비견/겁재:** 나와 비슷한 기운 (친구, 경쟁자)
                * **식신/상관:** 내가 표현하는 기운 (재능, 말)
                * **재성(편재/정재):** 내가 지배하는 기운 (재물, 결과)
                * **관성(편관/정관):** 나를 통제하는 기운 (직장, 명예)
                * **인성(편인/정인):** 나를 돕는 기운 (공부, 문서)
                """)
    
    # ------------------------------------------------------
    # [광고 영역 2] 결과 페이지 하단 배너
    # ------------------------------------------------------
    st.markdown("---")
    display_google_ad(location="main")