import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_M = 6371000.0
EPS = 1e-12

# ---------- 지리적 좌표 변환 유틸리티 ----------
def to_rad(lat, lon): return np.radians(np.c_[lat, lon])
def radius_rad(m):    return m / EARTH_RADIUS_M

# ---------- 시간 분류 유틸리티 ----------
def get_time_category(time_str):
    if pd.isna(time_str): return None
    s = ''.join(ch for ch in str(time_str) if ch.isdigit())
    if not s: return None
    try: hour = int(s.zfill(4)[:2])
    except:  return None
    if   0 <= hour < 3:   return 'time_0_3'
    elif 3 <= hour < 6:   return 'time_3_6'
    elif 6 <= hour < 9:   return 'time_6_9'
    elif 9 <= hour < 12:  return 'time_9_12'
    elif 12 <= hour < 15: return 'time_12_15'
    elif 15 <= hour < 18: return 'time_15_18'
    elif 18 <= hour < 21: return 'time_18_21'
    elif 21 <= hour <= 23:return 'time_21_24'
    return None

# ---------- 거리 기반 가중치 커널 함수들 ----------
def weights_from_dist(d_m, kernel="exp", tau=None, sigma=None, p=1.0):
    """거리 기반 가중치 계산"""
    d_m = np.maximum(d_m, EPS)  # 0으로 나누기 방지

    if kernel == "exp":
        tau = tau or 75.0
        return np.exp(-d_m / tau)
    elif kernel == "gaussian":
        sigma = sigma or 75.0
        return np.exp(-(d_m**2) / (2.0 * sigma**2))
    elif kernel == "idw":
        tau = tau or 75.0
        return 1.0 / (1.0 + d_m / max(tau, EPS))
    elif kernel == "custom":
        # K(d) = 1 / d^p
        k = 1.0 / (d_m ** p)
        # W(x) = x / (1 + x)
        return k / (1.0 + k)
    
    return np.ones_like(d_m)

# ---------- 체포 데이터 시간대별 소프트 카운팅 ----------
def arrest_soft_features_time_only(node_lat, node_lon, arrest_df, radius_m=200,
                                   kernel="exp", tau=None, sigma=None, p=1.0):
    """체포 데이터에서 시간대별 soft count 피처 생성"""
    if arrest_df.empty: 
        print("체포 데이터가 비어있음.")
        return {}
    
    if not {'latitude', 'longitude'}.issubset(arrest_df.columns):
        print("체포 데이터에 latitude, longitude 컬럼이 필요.")
        return {}
    
    # BallTree 생성
    tree = BallTree(to_rad(arrest_df['latitude'].to_numpy(),
                           arrest_df['longitude'].to_numpy()),
                    metric='haversine')
    nodes_rad = to_rad(node_lat, node_lon)
    ind_list, dist_list = tree.query_radius(
        nodes_rad, r=radius_rad(radius_m),
        return_distance=True, sort_results=True
    )

    # 8개 시간대
    times = ['time_0_3','time_3_6','time_6_9','time_9_12',
             'time_12_15','time_15_18','time_18_21','time_21_24']
    names = [f"arrest_{t}" for t in times]
    out = {n: np.zeros(len(node_lat), dtype=float) for n in names}

    # 시간 카테고리 변환
    time_col = arrest_df['Time'].map(get_time_category)
    
    print(f"시간 카테고리 분포:")
    print(time_col.value_counts().sort_index())

    # 각 노드에 대해 주변 체포 건수 계산
    for i, (nds, d_rad) in enumerate(zip(ind_list, dist_list)):
        if len(nds) == 0: continue
        d_m = d_rad * EARTH_RADIUS_M
        w = weights_from_dist(d_m, kernel=kernel, tau=tau, sigma=sigma, p=p)
        t = time_col.iloc[nds].to_numpy()
        for j in range(len(nds)):
            tt = t[j]
            if tt: out[f"arrest_{tt}"][i] += w[j]
    
    return out

# ---------- 메인 실행 함수 ----------
def main():
    # ========== 1. 파일 경로 및 파라미터 설정 ==========
    ap = argparse.ArgumentParser()
    ap.add_argument("--crime", default="data/crime_without_arrest.csv")
    ap.add_argument("--arrest", default="data/arrest_data.csv")
    ap.add_argument("--output", default="data/crime_arrest.csv")
    ap.add_argument("--radius", type=int, default=255)
    ap.add_argument("--kernel", type=str, default="custom", choices=["exp","gaussian","idw","custom"])
    ap.add_argument("--tau", type=float, default=127.5)
    ap.add_argument("--sigma", type=float, default=127.5)
    ap.add_argument("--p", type=float, default=1.0, help="custom kernel에서 d^p의 p 값")
    args = ap.parse_args()
    
    # 기존 범죄 데이터 로드
    print("데이터 로딩 중")
    crime_df = pd.read_csv(args.crime)
    print(f"범죄 데이터: {len(crime_df)}행, {len(crime_df.columns)}열")
    
    # 체포 데이터 로드
    try:
        arrest_df = pd.read_csv(args.arrest)
        print(f"체포 데이터: {len(arrest_df)}행, {len(arrest_df.columns)}열")
        print("체포 데이터 컬럼:", list(arrest_df.columns))
        print("체포 데이터 샘플:")
        print(arrest_df.head(3))
    except Exception as e:
        print(f"체포 데이터 로드 실패: {e}")
        return
    
    # 노드 위치 (범죄 데이터의 위도/경도)
    node_lat = crime_df["latitude"].to_numpy()
    node_lon = crime_df["longitude"].to_numpy()
    
    # 체포 시간대별 soft count 피처 생성
    print(f"\n체포 시간대별 피처 생성 중")
    print(f"kernel={args.kernel}, p={args.p}, tau={args.tau}, radius={args.radius}m")
    
    arrest_soft = arrest_soft_features_time_only(
        node_lat, node_lon, arrest_df,
        radius_m=args.radius, kernel=args.kernel, tau=args.tau, sigma=args.sigma, p=args.p
    )
    
    if not arrest_soft:
        print("체포 피처 생성 실패")
        return
    
    # soft count를 DataFrame으로 변환
    arrest_names = list(arrest_soft.keys())
    arrest_mat = np.column_stack([arrest_soft[n] for n in arrest_names])
    
     # 기존 데이터에 추가
    for j, name in enumerate(arrest_names):
        crime_df[f"{name}_soft"] = arrest_mat[:, j]
    
    print(f"\n체포 시간대별 soft count 평균:")
    for name in arrest_names:
        soft_col = f"{name}_soft"
        mean_val = crime_df[soft_col].mean()
        print(f"{soft_col:25s} mean={mean_val:7.3f}")
    
    # 최종 결과 저장
    crime_df.to_csv(args.output, index=False)
    
    print(f"\n[완료] {args.output}")
    print(f"최종 데이터: {len(crime_df)}행, {len(crime_df.columns)}열")
    print(f"추가된 체포 피처: {len(arrest_names)}개 (soft)")

if __name__ == "__main__":
    main()
