import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Cosine similarity 계산 함수 ----------
def calculate_cosine_similarities(df, target_col='crime_count'):
    """
    모든 피처와 타겟 변수 간 cosine similarity 계산
    """
    if target_col not in df.columns:
        raise ValueError(f"{target_col} 컬럼이 데이터에 없음")

    # 분석에서 제외할 컬럼들
    exclude_cols = ['grid_row', 'grid_col', 'latitude', 'longitude', target_col]

    # 피처 컬럼 선택
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"분석 대상 피처 수: {len(feature_cols)}")
    print(f"타겟 변수: {target_col}")

    # target 벡터 준비
    target_values = df[target_col].to_numpy().reshape(1, -1)

    similarities = []

    for col in feature_cols:
        # 결측치 처리 → 결측 있으면 해당 행 제거
        mask = ~(pd.isna(df[col]) | pd.isna(df[target_col]))
        if mask.sum() < 10:  # 데이터가 너무 적은 경우 제외
            similarities.append(np.nan)
            continue

        feature_values = df.loc[mask, col].to_numpy().reshape(1, -1)
        target_masked = target_values[:, mask]

        try:
            sim = cosine_similarity(feature_values, target_masked)[0, 0]
            similarities.append(sim)
        except Exception:
            similarities.append(np.nan)

    # 결과 DataFrame
    result_df = pd.DataFrame({
        'feature': feature_cols,
        'cosine_sim': similarities,
        'abs_sim': np.abs(similarities)
    })

    # 유효 결과만
    result_df = result_df.dropna()

    return result_df


# ---------- 카테고리별 분석 ----------
def analyze_similarities_by_category(sim_df):
    """
    피처 카테고리별 cosine similarity 분석
    """
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
        cat_features = sim_df[sim_df['feature'].str.contains('|'.join(keywords), na=False)]

        if len(cat_features) > 0:
            category_results[category] = {
                'count': len(cat_features),
                'mean_abs_sim': cat_features['abs_sim'].mean(),
                'max_sim': cat_features['abs_sim'].max(),
                'top_feature': cat_features.loc[cat_features['abs_sim'].idxmax(), 'feature'],
                'top_sim': cat_features.loc[cat_features['abs_sim'].idxmax(), 'cosine_sim']
            }

    return category_results


# ---------- 상위 피처 시각화 ----------
def plot_top_similarities(sim_df, top_n=20, figsize=(12, 8)):
    """
    상위 N개 피처의 cosine similarity 시각화
    """
    top_features = sim_df.nlargest(top_n, 'abs_sim')

    plt.figure(figsize=figsize)
    colors = ['red' if x < 0 else 'blue' for x in top_features['cosine_sim']]

    bars = plt.barh(range(len(top_features)), top_features['cosine_sim'], color=colors, alpha=0.7)

    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Cosine Similarity')
    plt.title(f'Top {top_n} Features by Cosine Similarity with Crime Count')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=8)

    plt.tight_layout()
    return plt


# ---------- 메인 ----------
def main():
    data_path = "data/crime_final.csv"
    output_path = "result_corr/cos_similarity_global.csv"

    print("데이터 로딩 중..")
    df = pd.read_csv(data_path)

    print(f"데이터 크기: {df.shape}")
    print(f"범죄 건수 통계:")
    print(f"  평균: {df['crime_count'].mean():.1f}")
    print(f"  중위수: {df['crime_count'].median():.1f}")
    print(f"  최소: {df['crime_count'].min()}")
    print(f"  최대: {df['crime_count'].max()}")

    # Cosine similarity 계산
    print("\nCosine similarity 계산 중..")
    sim_results = calculate_cosine_similarities(df, target_col='crime_count')

    # 절댓값 기준 정렬
    sim_results = sim_results.sort_values('abs_sim', ascending=False)

    print(f"\n유효한 similarity 결과: {len(sim_results)}개")

    # 상위 20개 출력
    print(f"\n상위 20개 피처 (절댓값 기준):")
    print("=" * 80)
    top_20 = sim_results.head(20)
    for idx, row in top_20.iterrows():
        print(f"{row['feature'][:45]:45s} {row['cosine_sim']:6.3f}")

    # 카테고리별 분석
    print(f"\n카테고리별 similarity 분석:")
    print("=" * 80)
    category_results = analyze_similarities_by_category(sim_results)

    for category, stats in category_results.items():
        print(f"{category:12s}: {stats['count']:2d}개 피처, "
              f"평균 |similarity|={stats['mean_abs_sim']:.3f}, "
              f"최고 |similarity|={stats['max_sim']:.3f}")
        print(f"             최고 피처: {stats['top_feature'][:50]} ({stats['top_sim']:.3f})")

    # 결과 저장
    sim_results.to_csv(output_path, index=False)
    print(f"\nCosine similarity 결과 저장: {output_path}")

    # 시각화
    print("\n시각화 생성 중...")
    plt_obj = plot_top_similarities(sim_results, top_n=25, figsize=(14, 10))

    plot_path = "result_corr/top_cos_similarity_global.png"
    plt_obj.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"시각화 저장: {plot_path}")

    plt_obj.show()

    return sim_results


if __name__ == "__main__":
    results = main()
