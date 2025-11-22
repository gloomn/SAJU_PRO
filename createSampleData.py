import pandas as pd
import random
from datetime import datetime

# ------------------------------------------------------
# [1] 사주 기초 데이터
# ------------------------------------------------------
cheon_gan = ['갑', '을', '병', '정', '무', '기', '경', '신', '임', '계']
ji_ji = ['자', '축', '인', '묘', '진', '사', '오', '미', '신', '유', '술', '해']

# 오행 매핑 (천간/지지 -> 오행)
five_elem_map = {
    '갑': '목', '을': '목', '인': '목', '묘': '목',
    '병': '화', '정': '화', '사': '화', '오': '화',
    '무': '토', '기': '토', '진': '토', '술': '토', '축': '토', '미': '토',
    '경': '금', '신': '금', '신(지지)': '금', '유': '금', # 신(申)과 신(辛) 구분
    '임': '수', '계': '수', '해': '수', '자': '수'
}

# ------------------------------------------------------
# [2] 만세력 계산 엔진 (앱과 동일한 로직)
# ------------------------------------------------------
def get_real_saju(year, month, day, hour):
    sixty_ganji = [cheon_gan[i % 10] + ji_ji[i % 12] for i in range(60)]
    
    # 1. 연주 (Year Pillar) - 입춘(2월 4일) 기준
    saju_year = year
    if month < 2 or (month == 2 and day < 4):
        saju_year = year - 1
    y_idx = (saju_year - 1984) % 60
    year_pillar = sixty_ganji[y_idx]

    # 2. 월주 (Month Pillar) - 절기 기준
    jeolgi_dates = [0, 6, 4, 6, 5, 6, 6, 7, 8, 8, 8, 7, 7] 
    saju_month = month
    if day < jeolgi_dates[month]:
        saju_month = month - 1
        if saju_month == 0: saju_month = 12

    # 월두법
    year_stem_idx = (saju_year - 1984) % 10 
    first_month_stem_idx = (year_stem_idx % 5) * 2 + 2 
    if saju_month == 1: month_msg_idx = 11
    elif saju_month == 2: month_msg_idx = 0
    else: month_msg_idx = saju_month - 2
    
    curr_month_stem_idx = (first_month_stem_idx + month_msg_idx) % 10
    curr_month_branch_idx = (month_msg_idx + 2) % 12 
    month_pillar = cheon_gan[curr_month_stem_idx] + ji_ji[curr_month_branch_idx]

    # 3. 일주 (Day Pillar) - 2000년 1월 1일 기준
    base_date = datetime(2000, 1, 1) 
    # 날짜 오류 방지 (예: 2월 30일)
    try:
        target_date = datetime(year, month, day)
    except ValueError:
        day = 28
        target_date = datetime(year, month, day)
        
    days_diff = (target_date - base_date).days
    day_idx = (days_diff + 54) % 60
    day_pillar = sixty_ganji[day_idx]
    
    # 4. 시주 (Time Pillar)
    if hour >= 23 or hour < 1: time_branch_idx = 0 
    else: time_branch_idx = (hour + 1) // 2 % 12
    
    day_stem_idx = cheon_gan.index(day_pillar[0])
    time_start_idx = (day_stem_idx % 5) * 2
    time_stem_idx = (time_start_idx + time_branch_idx) % 10
    time_pillar = cheon_gan[time_stem_idx] + ji_ji[time_branch_idx]
    
    return [year_pillar, month_pillar, day_pillar, time_pillar]

# ------------------------------------------------------
# [3] 오행 카운트 및 라벨링
# ------------------------------------------------------
def count_five_elements(pillars):
    counts = {'목': 0, '화': 0, '토': 0, '금': 0, '수': 0}
    for pillar in pillars:
        stem = pillar[0]
        branch = pillar[1]
        counts[five_elem_map[stem]] += 1
        # 지지 신(申)과 유(酉) 처리
        if branch in ['신', '유']: counts['금'] += 1
        else: counts[five_elem_map.get(branch, '토')] += 1
    return counts

# 기존 함수를 지우고 이걸로 교체하세요
def get_personality_label(elements):
    # [노이즈 추가] 10%의 확률로 엉뚱한 성격이 나옴 (현실성 부여)
    if random.random() < 0.1: # 10% 확률
        types = ['추진력있는 성장가 (목형)', '열정적인 리더 (화형)', 
                 '믿음직한 중재자 (토형)', '철저한 원칙주의자 (금형)', 
                 '지혜로운 전략가 (수형)', '유연한 밸런서 (조화형)']
        return random.choice(types)

    # 나머지 90%는 정상 로직
    max_element = max(elements, key=elements.get)
    max_count = elements[max_element]
    if max_count >= 3:
        if max_element == '목': return '추진력있는 성장가 (목형)'
        elif max_element == '화': return '열정적인 리더 (화형)'
        elif max_element == '토': return '믿음직한 중재자 (토형)'
        elif max_element == '금': return '철저한 원칙주의자 (금형)'
        elif max_element == '수': return '지혜로운 전략가 (수형)'
    return '유연한 밸런서 (조화형)'

# ------------------------------------------------------
# [4] 데이터 생성 실행 (10,000건)
# ------------------------------------------------------
print("데이터 생성을 시작합니다... (약 10초 소요)")
data = []
for _ in range(10000):
    year = random.randint(1960, 2005)
    month = random.randint(1, 12)
    
    # 월별 일수 처리
    if month == 2: max_day = 28
    elif month in [4, 6, 9, 11]: max_day = 30
    else: max_day = 31
    
    day = random.randint(1, max_day)
    hour = random.randint(0, 23)
    gender = random.choice(['남', '여'])
    
    # 사주 계산
    pillars = get_real_saju(year, month, day, hour)
    elem_counts = count_five_elements(pillars)
    label = get_personality_label(elem_counts)
    
    row = {
        '생년': year, '월': month, '일': day, '시': hour, '성별': gender,
        '연주': pillars[0], '월주': pillars[1], '일주': pillars[2], '시주': pillars[3],
        '목': elem_counts['목'], '화': elem_counts['화'], 
        '토': elem_counts['토'], '금': elem_counts['금'], 
        '수': elem_counts['수'],
        '성격유형': label, # 정답 레이블
        '성별_code': 0 if gender == '남' else 1
    }
    data.append(row)

# 저장
df = pd.DataFrame(data)
df.to_excel('real_saju_data.xlsx', index=False)
print(f"완료! 총 {len(df)}개의 데이터가 'real_saju_data.xlsx'에 저장되었습니다.")
print(df.head())