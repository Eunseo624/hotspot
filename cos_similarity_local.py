import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 지리 좌표 변환 ----------
EARTH_RADIUS_M = 6371000.0
def to_rad(lat, lon): return np.radians(np.c_[lat, lon])
def radius_rad(m):    return m / EARTH_RADIUS_M

def main():
    data_path = "data/crime_final.csv"
    output_path = "result_corr/cos_similarity_local.csv"
    radius_m = 510   # 반경 (미터 단위)

    print("데이터 로딩 중..")
    df = pd.read_csv(data_path)
    print(f"데이터 크기: {df.shape}")

    # 좌표와 타겟 변수
    node_lat = df["latitude"].to_numpy()
    node_lon = df["longitude"].to_numpy()
    nodes_rad = to_rad(node_lat, node_lon)
    target_col = "crime_count"

    # 분석 제외 컬럼
    exclude_cols = ['grid_row', 'grid_col', 'latitude', 'longitude', target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"분석 대상 피처 수: {len(feature_cols)}")
    print(f"타겟 변수: {target_col}")

    # BallTree 생성
    tree = BallTree(nodes_rad, metric='haversine')
    ind_list = tree.query_radius(nodes_rad, r=radius_rad(radius_m))

    neighbor_counts = [len(ind) for ind in ind_list]
    few_neighbors = sum(c < 10 for c in neighbor_counts)

    print("\n===== 반경 510m 내 이웃 노드 개수 요약 =====")
    print(f"총 노드 수: {len(df)}")
    print(f"이웃노드가 10개 미만인 노드 수: {few_neighbors}")
    print(f"평균 이웃 노드 수: {np.mean(neighbor_counts):.1f}")
    print(f"최소: {np.min(neighbor_counts)}, 최대: {np.max(neighbor_counts)}\n")

    # 결과 저장용 리스트
    results = []

    # 각 노드별 cosine similarity 계산
    for node_id, neighbors in enumerate(ind_list):
        neighbor_df = df.iloc[neighbors]

        row_result = {"node_id": node_id}

        for col in feature_cols:
            try:
                # 결측치 제거
                mask = ~(pd.isna(neighbor_df[col]) | pd.isna(neighbor_df[target_col]))
                if mask.sum() < 2:   # 데이터가 너무 적으면 계산 불가
                    row_result[col] = np.nan
                    continue

                x = neighbor_df.loc[mask, col].to_numpy().reshape(1, -1)
                y = neighbor_df.loc[mask, target_col].to_numpy().reshape(1, -1)

                sim = cosine_similarity(x, y)[0, 0]
                row_result[col] = sim
            except Exception:
                row_result[col] = np.nan

        results.append(row_result)

        if node_id < 2:  # 앞 2개 노드만 미리 보기 출력
            print(f"node_id={node_id}, neighbors={len(neighbor_df)}")
            sample_corrs = {f: row_result[f] for f in feature_cols[:2]}
            print("  샘플 cosine similarity:", sample_corrs)

    # 결과 DataFrame
    result_df = pd.DataFrame(results)

    # CSV 저장
    result_df.to_csv(output_path, index=False)
    print(f"\n[완료] 로컬 Cosine similarity 결과 저장: {output_path}")
    print(f"결과 크기: {result_df.shape} (노드 × 피처 매트릭스)")

    # ===================================================================
    # 노드별 요약 통계
    # ===================================================================
    print("\n===== 노드별 요약 통계 생성 중 =====")
    result_df_indexed = result_df.set_index("node_id")

    summary = []
    for node_id, row in result_df_indexed.iterrows():
        abs_vals = row.abs()
        mean_abs_sim = abs_vals.mean()               # 절댓값 평균
        top_feature = abs_vals.idxmax()              # 가장 큰 피처 이름
        top_sim = row[top_feature]                   # 그 피처의 실제 값
        summary.append([node_id, mean_abs_sim, top_feature, top_sim])

    summary_df = pd.DataFrame(summary,
                              columns=["node_id", "mean_abs_sim", "top_feature", "top_sim"])

    summary_path = "result_corr/cos_similarity_local_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"[완료] 노드별 요약 통계 저장: {summary_path}")
    print(f"요약 데이터 크기: {summary_df.shape} (노드 × 4)")
    print("\n샘플 5개:")
    print(summary_df.head())

    """
    # --- 히트맵 시각화 ---
    # 저장된 상관계수 결과만 다시 불러오기
    corr_df = pd.read_csv(output_path).set_index("node_id")

    # 샘플링 (앞 20개 노드, 상위 20개 feature만)
    sample_nodes = corr_df.index[:20]
    sample_features = corr_df.columns[:20]
    sample_df = corr_df.loc[sample_nodes, sample_features]

    # 히트맵 그리기
    plt.figure(figsize=(14, 10))
    sns.heatmap(sample_df, cmap="coolwarm", vmin=-1, vmax=1, annot=False)
    plt.title("Cosine Similarity Heatmap (sample)")
    plt.xlabel("Features")
    plt.ylabel("Node ID")
    plt.tight_layout()

    # 저장
    plt.savefig("result/cosine_heatmap_sample.png", dpi=300)
    print("히트맵 저장: result/cosine_heatmap_sample.png")

    plt.show()
    """

if __name__ == "__main__":
    main()
