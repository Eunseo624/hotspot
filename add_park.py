import pandas as pd
import numpy as np
from itertools import product

# ---------- 공원 격자 피처 생성 함수 ----------
def build_park_3x3_features(crime_df, park_grid_df):
    """
    공원 격자 데이터를 기반으로 각 범죄 노드에 3x3 공원 피처 생성
    
    Args:
        crime_df: 범죄 데이터프레임 (grid_row, grid_col 컬럼)
        park_grid_df: 공원 격자 데이터프레임 (grid_row, grid_col 컬럼)
                      공원이 있는 격자만 포함된 데이터
    
    Returns:
        공원 피처 데이터프레임 (park_center, park_count_3x3, park_ratio_3x3)
    """
    # 공원 격자를 set으로 변환
    park_set = set(zip(park_grid_df['grid_row'], park_grid_df['grid_col']))
    
    print(f"공원이 있는 격자 수: {len(park_set)}")
    
    # 3x3 격자 오프셋
    offsets = list(product([-1, 0, 1], [-1, 0, 1]))
    
    park_features = [] # 각 노드의 공원 피처를 저장할 리스트
    
    # 각 범죄 노드별로 3x3 영역 내 공원 정보 계산
    for _, row in crime_df.iterrows():
        # 현재 노드의 격자 좌표
        r0, c0 = int(row['grid_row']), int(row['grid_col'])
        
        # 3x3 격자 내 공원 정보 수집
        park_count = 0      # 3x3 내 공원 격자 개수
        center_park = 0     # 중심 격자의 공원 존재 여부
        
        for dr, dc in offsets:
            grid_key = (r0 + dr, c0 + dc)
            
            if grid_key in park_set:
                park_count += 1
                
            # 중심 격자의 공원 존재 여부
            if dr == 0 and dc == 0 and grid_key in park_set:
                center_park = 1
        
        park_features.append({
            'park_center': center_park,  # 중심 격자에 공원 있음(1)/없음(0)
            'park_count_3x3': park_count,  # 3x3 내 공원 격자 수 (0~9)
            'park_ratio_3x3': park_count / 9.0  # 3x3 내 공원 비율 (0.0~1.0)
        })
    
    return pd.DataFrame(park_features)

def main():
    # 파일 경로
    crime_path = "data/crime_arrest.csv"  # 기존 최종 파일
    park_path = "data/park_grid.csv"    # 공원 격자 파일
    output_path = "data/crime_a_p.csv"
    
    print("데이터 로딩 중")
    
    # 범죄 데이터 로드
    crime_df = pd.read_csv(crime_path)
    print(f"범죄 데이터: {len(crime_df)}행, {len(crime_df.columns)}열")
    
    # 공원 격자 데이터 로드
    try:
        park_df = pd.read_csv(park_path)
        print(f"공원 데이터: {len(park_df)}행, {len(park_df.columns)}열")
        print("공원 데이터 컬럼:", list(park_df.columns))
        
        # 필요한 컬럼 확인
        required_cols = ['grid_row', 'grid_col']
        if not all(col in park_df.columns for col in required_cols):
            print(f"공원 데이터에 필요한 컬럼이 없다: {required_cols}")
            print("첫 5행 확인:")
            print(park_df.head())
            return
            
    except Exception as e:
        print(f"공원 데이터 로드 실패: {e}")
        return
    
    # 필수 컬럼 확인
    if not {'grid_row', 'grid_col'}.issubset(crime_df.columns):
        raise ValueError("범죄 데이터에 grid_row, grid_col 컬럼이 필요함")
    
    # 공원 3x3 피처 생성
    print("\n공원 격자 피처 생성 중이요")
    park_features = build_park_3x3_features(crime_df, park_df)
    
    # 기존 범죄 데이터와 병합
    print("데이터 병합 중임")
    final_df = pd.concat([crime_df, park_features], axis=1)
    
    # 결과 저장
    final_df.to_csv(output_path, index=False)
    
    print(f"\n[완료] {output_path}")
    print(f"최종 데이터: {len(final_df)}행, {len(final_df.columns)}열")
    print(f"추가된 공원 피처: {len(park_features.columns)}개")
    
    # 공원 피처 통계
    print(f"\n공원 피처 통계:")
    print(f"중심 격자에 공원이 있는 노드: {final_df['park_center'].sum()}개 ({final_df['park_center'].mean()*100:.1f}%)")
    print(f"3x3 내 평균 공원 격자 수: {final_df['park_count_3x3'].mean():.2f}개")
    print(f"3x3 내 평균 공원 비율: {final_df['park_ratio_3x3'].mean()*100:.1f}%")
    
    # 분포 확인
    print(f"\n3x3 내 공원 격자 수 분포:")
    print(final_df['park_count_3x3'].value_counts().sort_index())

if __name__ == "__main__":
    main()
    