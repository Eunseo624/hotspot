import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# 지구 반지름 (미터 단위)
EARTH_RADIUS_M = 6371000.0
# 계산 시 0으로 나누기 방지를 위한 값
EPS = 1e-12

# ---------- 지리적 좌표 변환 유틸리티 ----------
def to_rad(lat, lon): return np.radians(np.c_[lat, lon])
def radius_rad(m):    return m / EARTH_RADIUS_M

# ---------- 시간 분류 유틸리티 ----------
def get_time_category(time_str):
    """
    시간 문자열을 8개의 시간대 카테고리로 분류
    
    Args:
        time_str: 시간 문자열 (예: "1430", "14:30" 등)
    
    Returns:
        시간대 카테고리 문자열 (예: "time_12_15") 또는 None
    """
    if pd.isna(time_str): return None
    
    # 숫자만 추출하여 시간 파싱
    s = ''.join(ch for ch in str(time_str) if ch.isdigit())
    if not s: return None
    
    try: 
        hour = int(s.zfill(4)[:2])  # 4자리로 맞춘 후 앞 2자리가 시간
    except:  
        return None
    
    # 3시간 단위로 시간대 분류
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
    """
    거리에 따른 가중치를 계산하는 함수
    
    Args:
        d_m: 거리(미터)
        kernel: 커널 타입 ("exp", "gaussian", "idw", "custom")
        tau: 지수/IDW 커널의 감쇠 파라미터
        sigma: 가우시안 커널의 표준편차 -> 안씀
        p: custom 커널에서 d^p의 지수 (default=1)
    
    Returns:
        거리에 따른 가중치 배열
    """
    d_m = np.maximum(d_m, EPS)  # 0으로 나누기 방지
    if kernel == "exp":
        # 지수 감쇠: exp(-d/tau)
        tau = tau or 75.0
        return np.exp(-d_m / tau)
    elif kernel == "gaussian":
        # 가우시안(정규분포): exp(-d²/(2σ²))
        sigma = sigma or 75.0
        return np.exp(-(d_m**2) / (2.0 * sigma**2))
    elif kernel == "idw":
        # 역거리 가중치: 1/(1 + d/tau)
        tau = tau or 75.0
        return 1.0 / (1.0 + d_m / max(tau, EPS))
    elif kernel == "custom":
        # K(d) = 1 / d^p
        k = 1.0 / (d_m ** p)
        # W(x) = x / (1 + x)
        return k / (1.0 + k)
    
    # 기본값: 균등 가중치
    return np.ones_like(d_m)

# ---------- 소프트 카운팅 함수 ----------
def soft_count_points(node_lat, node_lon, pt_lat, pt_lon,
                      radius_m=255, kernel="exp", tau=None, sigma=None, p=1.0):
    """
    각 노드 주변의 포인트들을 거리에 따른 가중치로 소프트 카운팅
    
    Args:
        node_lat, node_lon: 기준 노드들의 위도, 경도
        pt_lat, pt_lon: 카운팅할 포인트들의 위도, 경도
        radius_m: 검색 반경(미터)
        kernel: 가중치 커널 타입
        tau, sigma: 커널 파라미터
    
    Returns:
        각 노드의 소프트 카운트 값 배열
    """
    if len(pt_lat) == 0:
        return np.zeros(len(node_lat), dtype=float)
    
    # BallTree로 이웃 검색 -> 위도 경도 좌표로 haversine으로 구현했는데 euclidean..?
    tree = BallTree(to_rad(pt_lat, pt_lon), metric='haversine')
    nodes_rad = to_rad(node_lat, node_lon)
    
    # 각 노드에서 반경 내 포인트들과 거리 찾기
    ind_list, dist_list = tree.query_radius(
        nodes_rad, r=radius_rad(radius_m),
        return_distance=True, sort_results=True
    )
    
    out = np.zeros(len(node_lat), dtype=float)
    
    # 각 노드별로 가중치 합계 계산
    for i, (nds, d_rad) in enumerate(zip(ind_list, dist_list)):
        if len(nds) == 0: continue
        
        # 라디안 거리를 미터로 변환
        d_m = d_rad * EARTH_RADIUS_M
        # 거리에 따른 가중치 계산
        w = weights_from_dist(d_m, kernel=kernel, tau=tau, sigma=sigma, p=p)
        # 가중치 합계가 해당 노드의 소프트 카운트
        out[i] = w.sum()
    
    return out

# ---------- 교통사고 시간대별 소프트 피처 생성 ----------
def traffic_soft_features_time_only(node_lat, node_lon, traffic_df, radius_m=200,
                                    kernel="exp", tau=None, sigma=None, p=1.0):
    """
    교통사고 데이터를 시간대별로 분류하여 소프트 카운팅
    
    Args:
        node_lat, node_lon: 기준 노드들의 위도, 경도
        traffic_df: 교통사고 데이터프레임 (latitude, longitude, Time_Occurred 컬럼 필요)
        radius_m: 검색 반경(미터)
        kernel: 가중치 커널 타입
        tau, sigma: 커널 파라미터
        p: custom 커널의 지수 (1/d^p)
    
    Returns:
        시간대별 교통사고 소프트 카운트 딕셔너리
    """
    if traffic_df.empty: return {}
    
    # 교통사고 위치로 BallTree 구성
    tree = BallTree(to_rad(traffic_df['latitude'].to_numpy(),
                           traffic_df['longitude'].to_numpy()),
                    metric='haversine')
    nodes_rad = to_rad(node_lat, node_lon)
    
    # 각 노드에서 반경 내 교통사고들과 거리 찾기
    ind_list, dist_list = tree.query_radius(
        nodes_rad, r=radius_rad(radius_m),
        return_distance=True, sort_results=True
    )

    # 8개 시간대 정의
    times = ['time_0_3','time_3_6','time_6_9','time_9_12',
             'time_12_15','time_15_18','time_18_21','time_21_24']
    names = [f"traffic_{t}" for t in times]
    
    # 각 시간대별 소프트 카운트 배열 초기화
    out = {n: np.zeros(len(node_lat), dtype=float) for n in names}

    # 교통사고 발생 시간을 시간대 카테고리로 변환
    time_col = traffic_df['Time_Occurred'].map(get_time_category)

    # 각 노드별로 시간대별 가중치 합계 계산
    for i, (nds, d_rad) in enumerate(zip(ind_list, dist_list)):
        if len(nds) == 0: continue
        
        # 라디안 거리를 미터로 변환
        d_m = d_rad * EARTH_RADIUS_M
        # 거리에 따른 가중치 계산
        w = weights_from_dist(d_m, kernel=kernel, tau=tau, sigma=sigma, p=p)
        # 해당 교통사고들의 시간대 정보
        t = time_col.iloc[nds].to_numpy()
        
        # 각 교통사고별로 해당 시간대에 가중치 추가
        for j in range(len(nds)):
            tt = t[j]
            if tt: 
                out[f"traffic_{tt}"][i] += w[j]
    
    return out

# ---------- 메인 실행 함수 ----------
def main():
    # ========== 1. 명령행 인자 파싱 ==========
    ap = argparse.ArgumentParser()
    
    # 여기에 데이터 주소 넣기!, help는 생략함
    ap.add_argument("--crime", default="data/crime_count_grid.csv")
    ap.add_argument("--alcohol", default="data/alcohol.csv")
    ap.add_argument("--bus", default="data/bus.csv")
    ap.add_argument("--metro_portal", default="data/metro_portal.csv")
    ap.add_argument("--metro", default="data/metro.csv")
    ap.add_argument("--school", default="data/school.csv")
    ap.add_argument("--traffic", default="data/traffic_collision.csv")
    
    ap.add_argument("--radius", type=int, default=255)
    ap.add_argument("--kernel", type=str, default="custom", choices=["exp","gaussian","idw","custom"])
    ap.add_argument("--tau", type=float, default=127.5)
    ap.add_argument("--sigma", type=float, default=127.5)
    ap.add_argument("--p", type=float, default=1.0, help="custom kernel에서 d^p의 p 값")
    
    args = ap.parse_args()


    # ========== 2. 데이터 로딩 ==========
    # 파일 로딩 실패 시 빈 pd 반환 -> 주소 확인
    def load_or_empty(path):
        try: return pd.read_csv(path)
        except: return pd.DataFrame(columns=["latitude","longitude"])
        
    print("데이터로딩중")

    crime = pd.read_csv(args.crime)
    node_lat = crime["latitude"].to_numpy()
    node_lon = crime["longitude"].to_numpy()

    alcohol = load_or_empty(args.alcohol)
    bus     = load_or_empty(args.bus)
    mport   = load_or_empty(args.metro_portal)
    metro   = load_or_empty(args.metro)
    school  = load_or_empty(args.school)
    traffic = load_or_empty(args.traffic)
    
    # 데이터 로딩 상태 출력
    print(f"데이터로딩완료")
    print(f"노드:{len(crime)} | alcohol:{len(alcohol)} bus:{len(bus)} metro_portal:{len(mport)} "
          f"metro:{len(metro)} school:{len(school)} traffic:{len(traffic)}")
    print(f"kernel={args.kernel}, tau={args.tau}, sigma={args.sigma}, radius={args.radius}m")

    # ========== 3. 시설별 소프트 카운팅 피처 생성 ==========
    # 시설 데이터들을 딕셔너리로 정리
    sources = {
        "alcohol": alcohol,
        "bus": bus, 
        "metro_portal": mport,
        "metro": metro,
        "school": school
    }
    
    fac_soft = {}  # 시설별 소프트 카운트 저장
    
    for name, df in sources.items():
        soft = soft_count_points(node_lat, node_lon,
                                 df.get("latitude", pd.Series([])).to_numpy(),
                                 df.get("longitude", pd.Series([])).to_numpy(),
                                 radius_m=args.radius, kernel=args.kernel, tau=args.tau, sigma=args.sigma, p=args.p)
        fac_soft[name] = soft
    
    # 피처 행렬 생성 및 범죄 데이터에 추가
    fac_names = list(fac_soft.keys())
    fac_mat = np.column_stack([fac_soft[n] for n in fac_names]) if fac_names else np.zeros((len(node_lat),0))
    
    for j, n in enumerate(fac_names):
        crime[f"{n}_soft"] = fac_mat[:, j]

    # ========== 4. 교통사고 시간대별 소프트 카운팅 피처 생성 ==========
    traf_soft = traffic_soft_features_time_only(node_lat, node_lon, traffic,
                                                radius_m=args.radius, kernel=args.kernel, tau=args.tau, sigma=args.sigma, p=args.p)
    
    # 교통사고 피처 행렬 생성 및 범죄 데이터에 추가
    tnames = list(traf_soft.keys())
    tmat   = np.column_stack([traf_soft[n] for n in tnames]) if tnames else np.zeros((len(node_lat),0))
    
    for j, n in enumerate(tnames):
        crime[f"{n}_soft"] = tmat[:, j]

    # ========== 5. 생성된 피처 요약 통계 출력 ==========
    print("\n시설 피처(soft) 요약:")
    for n in fac_names:
        x = crime[f"{n}_soft"]
        print(f"{n:15s} min={x.min():7.3f} mean={x.mean():7.3f} max={x.max():7.3f}")

    if tnames:
        print("\n교통사고(시간대) soft 평균:")
        for n in tnames: print(f"{n:22s} mean={crime[n+'_soft'].mean():7.3f}")

    # 주소 지정.
    out_path = "data/crime_without_arrest.csv"
    crime.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")
    print(f"최종 컬럼 수: {len(crime.columns)}")

if __name__ == "__main__":
    main()
