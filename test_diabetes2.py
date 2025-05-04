# 라이브러리 임포트
from sklearn.datasets import load_diabetes  # 당뇨병 예측 데이터셋 로드
import matplotlib.pyplot as plt

# 1. 당뇨병 데이터셋 로드
diabetes = load_diabetes()

# 2. 데이터 크기 확인
# - diabetes.data: (442, 10) 크기의 특성 데이터 (10개의 특성)
# - diabetes.target: (442,) 크기의 타겟(정답) 데이터 (당뇨병 진행 점수)
print(diabetes.data.shape, diabetes.target.shape)

# 3. 첫 3개 샘플의 특성과 타겟 값 출력
print(diabetes.data[0:3])      # 각 행은 환자, 열은 특성(BMI 등)
print(diabetes.target[:3])     # 해당 환자들의 당뇨병 점수

# 4. x와 y 설정
# - x: BMI 값만 사용 (세 번째 특성, 정규화된 값)
# - y: 당뇨병 진행 점수 (정답)
x = diabetes.data[:, 2]
y = diabetes.target

# 5. 가중치(w)와 절편(b) 초기값 설정
w = 1.0
b = 1.0

# 6. 첫 번째 데이터 샘플로 예측값 계산 (1차 함수 적용)
#    y_hat = x * w + b
y_hat = x[0] * w + b

print('실제 데이터 : ', y[0])      # 정답 값
print('방정식으로 나온 데이터 : ', y_hat)  # 예측한 값

# 7. 가중치를 소폭 증가시켜 예측값 다시 계산
w_inc = w + 0.1                     # w 증가
y_hat_inc = x[0] * w_inc + b        # 증가된 w로 예측

print('y_hat_inc : ', y_hat_inc)   # 새로운 예측값

# 8. 예측값 증가율을 이용해 가중치 변화율(기울기) 계산
#    수치 미분 방식 사용: (변화량/변화폭)
w_rate = (y_hat_inc - y_hat) / (w_inc - w)
print('w_rate : ', w_rate)

# 9. 가중치를 기울기만큼 업데이트
#    (이 값은 실제로는 경사 하강법의 한 단계라 볼 수 있음)
w_new = w + w_rate
print('w_new : ', w_new)

# 10. 절편도 같은 방식으로 업데이트 진행
b_inc = b + 0.1
y_hat_inc = x[0] * w + b_inc       # 증가된 b로 다시 예측
print('y_hat_inc : ', y_hat_inc)  # 새로운 예측값

# 11. 절편의 변화율(기울기) 계산
b_rate = (y_hat_inc - y_hat) / (b_inc - b)
print('b_rate : ', b_rate)
