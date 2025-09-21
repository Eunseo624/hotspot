import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 피어슨 상관계수 계산 함수 ----------
def calculate_pearson_correlations(df, target_col='crime_count'):
    """
    모든 피처와 타겟 변수 간 피어슨 상관계수 계산
    """
    if target_col not in df.columns:
        raise ValueError(f"{target_col} 컬럼이 데이터에 없음")
    
    # 분석에서 제외할 컬럼들
    exclude_cols = ['grid_row', 'grid_col', 'latitude', 'longitude', target_col]
    
    # 피처 선택
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"분석 대상 피처 수: {len(feature_cols)}")
    print(f"타겟 변수: {target_col}")
    
    correlations = []
    p_values = []
    target_values = df[target_col].values
    
    for col in feature_cols:
        # 결측치 제거
        mask = ~(pd.isna(df[col]) | pd.isna(df[target_col]))
        if mask.sum() < 10:  # 데이터 부족 시 제외
            correlations.append(np.nan)
            p_values.append(np.nan)
            continue
        
        feature_values = df.loc[mask, col].values
        target_masked = target_values[mask]
        
        try:
            corr, p_val = pearsonr(feature_values, target_masked)
            correlations.append(corr)
            p_values.append(p_val)
        except:
            correlations.append(np.nan)
            p_values.append(np.nan)
    
    # 결과 DataFrame 생성
    result_df = pd.DataFrame({
        'feature': feature_cols,
        'pearson_corr': correlations,
        'p_value': p_values,
        'abs_corr': np.abs(correlations)
    })
    
    return result_df.dropna() # 유효한 결과만 필터링

# ---------- 카테고리별 상관계수 분석 ----------
def analyze_correlations_by_category(corr_df):
    categories = {
        'Facilities': ['alcohol', 'bus', 'metro_portal', 'metro', 'school'],
        'Traffic': ['traffic_time'],
        'Arrest': ['arrest_time'],
        'Grid': ['grid_'],
        'Population': ['POP23_'],
        'Poverty': ['POV23_']
    }
    
    category_results = {}

    for category, keywords in categories.items():
        cat_features = corr_df[corr_df['feature'].str.contains('|'.join(keywords), na=False)]
        if len(cat_features) > 0:
            category_results[category] = {
                'count': len(cat_features),
                'mean_abs_corr': cat_features['abs_corr'].mean(),
                'max_corr': cat_features['abs_corr'].max(),
                'top_feature': cat_features.loc[cat_features['abs_corr'].idxmax(), 'feature'],
                'top_corr': cat_features.loc[cat_features['abs_corr'].idxmax(), 'pearson_corr']
            }
    return category_results

# ---------- 시각화 ----------
def plot_top_correlations(corr_df, top_n=20, figsize=(12, 8)):
    """
    절댓값 기준 상위 N개 피처 시각화
    """
    top_features = corr_df.nlargest(top_n, 'abs_corr')
    
    plt.figure(figsize=figsize)
    colors = ['red' if x < 0 else 'blue' for x in top_features['pearson_corr']]
    bars = plt.barh(range(len(top_features)), top_features['pearson_corr'], color=colors, alpha=0.7)
    
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Pearson Correlation Coefficient')
    plt.title(f'Top {top_n} Features by Absolute Correlation with Crime Count')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)
    
    # 상관계수 값 표시
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + (0.01 if width >= 0 else -0.01),
                 bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}',
                 ha='left' if width >= 0 else 'right',
                 va='center', fontsize=8)
    
    plt.tight_layout()
    return plt

# ---------- 메인 ----------
def main():
    data_path = "data/crime_final.csv"
    
    print("데이터 로딩 중")
    df = pd.read_csv(data_path)
    
    print(f"데이터 크기: {df.shape}")
    print(f"범죄 건수 통계:")
    print(f"  평균: {df['crime_count'].mean():.1f}")
    print(f"  중위수: {df['crime_count'].median():.1f}")
    print(f"  최소: {df['crime_count'].min()}")
    print(f"  최대: {df['crime_count'].max()}")
    
    # 피어슨 상관계수 계산
    print("\n피어슨 상관계수 계산 중")
    corr_results = calculate_pearson_correlations(df, target_col='crime_count')
    corr_results = corr_results.sort_values('abs_corr', ascending=False)
    
    print(f"\n유효한 상관계수 결과: {len(corr_results)}개")
    
    # 상위 20개 출력
    print("\n상위 20개 피처 (절댓값 기준):")
    print("=" * 80)
    top_20 = corr_results.head(20)
    for _, row in top_20.iterrows():
        significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"{row['feature'][:45]:45s} {row['pearson_corr']:6.3f} {significance:3s} (p={row['p_value']:.4f})")
    
    # 카테고리별 분석
    print("\n카테고리별 상관계수 분석:")
    print("=" * 80)
    category_results = analyze_correlations_by_category(corr_results)
    for category, stats in category_results.items():
        print(f"{category:12s}: {stats['count']:2d}개 피처, "
              f"평균 |상관계수|={stats['mean_abs_corr']:.3f}, "
              f"최고 |상관계수|={stats['max_corr']:.3f}")
        print(f"             최고 피처: {stats['top_feature'][:50]} ({stats['top_corr']:.3f})")
    
    # 유의한 피처 개수
    # (p < 0.05) 개수
    significant_corrs = corr_results[corr_results['p_value'] < 0.05]
    print(f"\n(p < 0.05): {len(significant_corrs)}개 / {len(corr_results)}개 ({len(significant_corrs)/len(corr_results)*100:.1f}%)")
    
    # (|r| > 0.3) 개수
    strong_corrs = corr_results[corr_results['abs_corr'] > 0.3]
    print(f"(|r| > 0.3): {len(strong_corrs)}개")
    
    # 결과 저장
    output_path = "result_corr/pearson_global.csv"
    corr_results.to_csv(output_path, index=False)
    print(f"\n상관계수 결과 저장: {output_path}")
    
    # 시각화
    print("\n시각화 생성 중")
    plt_obj = plot_top_correlations(corr_results, top_n=25, figsize=(14, 10))

    # 그래프 저장
    plot_path = "result_corr/top_correlations_pearson_global.png"
    plt_obj.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"시각화 저장: {plot_path}")
    
     # 그래프 출력
    plt_obj.show()
    return corr_results

if __name__ == "__main__":
    results = main()
