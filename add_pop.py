import pandas as pd
import numpy as np
from itertools import product

def build_population_grid_features(crime_df, pop_df):
    """
    인구/빈곤 데이터를 3x3 격자 기반으로 범죄 데이터에 추가
    중심값과 3x3 평균만 생성
    """
    # 인구 데이터를 grid_row, grid_col 기준으로 인덱싱
    pop_indexed = pop_df.set_index(['grid_row', 'grid_col'])
    
    # 인구/빈곤 관련 컬럼들 추출 (idx, grid_row, grid_col 제외)
    pop_columns = [col for col in pop_df.columns 
                   if col not in ['idx', 'grid_row', 'grid_col']]
    
    print(f"처리할 인구/빈곤 컬럼 수: {len(pop_columns)}")
    print("컬럼 목록:", pop_columns[:5], "..." if len(pop_columns) > 5 else "")
    
    # 3x3 격자 오프셋
    offsets = list(product([-1, 0, 1], [-1, 0, 1]))
    
    # 결과 저장용 딕셔너리
    result_features = {}
    
    # 각 인구/빈곤 컬럼에 대해 중심값과 3x3 평균만 생성
    for col in pop_columns:
        print(f"처리 중: {col}")
        
        # 각 범죄 노드에 대한 값들을 저장할 리스트
        mean_values = []    # 3x3 평균값들
        center_values = []  # 중심 격자값들
        
        # 각 범죄 데이터 포인트에 대해 격자 값 계산
        for _, row in crime_df.iterrows():
            r0, c0 = int(row['grid_row']), int(row['grid_col'])
            
            # 3x3 격자의 값들 수집
            grid_values = []
            center_val = 0.0
            
            for dr, dc in offsets:
                grid_key = (r0 + dr, c0 + dc)
                
                # 해당 격자에 인구 데이터가 있는지 확인
                if grid_key in pop_indexed.index:
                    val = float(pop_indexed.loc[grid_key, col])
                    grid_values.append(val)
                    
                    # 중심값 저장 (현재 격자)
                    if dr == 0 and dc == 0:
                        center_val = val
                else:
                    # 데이터가 없는 격자는 0으로 처리
                    grid_values.append(0.0)
                    
            # 통계값 계산
            grid_values = np.array(grid_values)
            
            # 중심값과 3x3 평균만 계산
            mean_values.append(float(grid_values.mean()))
            center_values.append(center_val)
        
        # 결과 저장
        result_features[f"{col}_center"] = center_values
        result_features[f"{col}_mean_3x3"] = mean_values
    
    # DataFrame으로 변환
    features_df = pd.DataFrame(result_features)
    
    return features_df

def main():
    # 파일 경로
    crime_path = "data/crime_a_p.csv"
    pop_path = "data/pop.csv"
    output_path = "data/crime_final.csv"
    
    print("데이터 로딩 중이여")
    
    # 데이터 로드
    crime_df = pd.read_csv(crime_path)
    pop_df = pd.read_csv(pop_path)
    
    print(f"범죄 데이터: {len(crime_df)}행, {len(crime_df.columns)}열")
    print(f"인구 데이터: {len(pop_df)}행, {len(pop_df.columns)}열")
    
    # 필수 컬럼 확인
    if not {'grid_row', 'grid_col'}.issubset(crime_df.columns):
        raise ValueError("crime 데이터에 grid_row, grid_col 컬럼 필요")
    
    if not {'grid_row', 'grid_col'}.issubset(pop_df.columns):
        raise ValueError("pop 데이터에 grid_row, grid_col 컬럼 필요")
    
    # 인구 격자 피처 생성
    print("\n인구/빈곤 격자 피처 생성 중요")
    pop_features = build_population_grid_features(crime_df, pop_df)
    
    # 기존 범죄 데이터와 병합
    print("데이터 병합 중")
    final_df = pd.concat([crime_df, pop_features], axis=1)
    
    # 결과 저장
    final_df.to_csv(output_path, index=False)
    
    print(f"\n[완료] {output_path}")
    print(f"최종 데이터: {len(final_df)}행, {len(final_df.columns)}열")
    print(f"추가된 인구 피처 수: {len(pop_features.columns)} (각 컬럼당 중심값 + 3x3평균)")
    
    # 샘플 통계 출력
    print("\n일부 인구 피처 통계:")
    sample_cols = [col for col in pop_features.columns if 'TOTAL' in col][:4]
    for col in sample_cols:
        vals = final_df[col]
        print(f"{col:35s} min={vals.min():8.1f} mean={vals.mean():8.1f} max={vals.max():8.1f}")

if __name__ == "__main__":
    main()
    