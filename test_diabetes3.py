# 1차 함수 진행하기 (선형 회귀 기본 원리 직접 구현)

from sklearn.datasets import load_diabetes  # 당뇨병 예측 데이터셋 로드
import matplotlib.pyplot as plt             # 그래프 시각화를 위한 라이브러리

# 1. 데이터 로딩
diabetes = load_diabetes()

# 2. 입력(x), 출력(y) 데이터 선택
# x: BMI (Body Mass Index) - 세 번째 특성
# y: 당뇨병 진행 정도
x = diabetes.data[:, 2]
y = diabetes.target

# 3. 초기 가중치(w)와 절편(b) 설정
w = 1.0
b = 1.0

# 4. 첫 번째 데이터 샘플을 이용한 예측값 계산
y_hat = x[0] * w + b     # 예측값
err = y[0] - y_hat       # 실제값과 예측값의 오차

# 5. 오차를 기반으로 w, b를 한 번 업데이트
# - w의 변화율(w_rate)은 x[0] (편미분 개념)
# - b는 항상 1 (편미분 시 도함수는 상수 1)
w_rate_for_update = x[0]
w_new1 = w + w_rate_for_update * err  # 가중치 업데이트
b_new1 = b + 1 * err                  # 절편 업데이트
print('첫번째 값 : ', w_new1, b_new1)

# 6. 두 번째 데이터 샘플을 이용한 반복 업데이트
# - 새로 구한 w_new1, b_new1을 사용하여 다시 예측
y_hat = x[1] * w_new1 + b_new1
err = y[1] - y_hat
w_rate = x[1]
w_new2 = w_new1 + w_rate * err
b_new2 = b_new1 + 1 * err
print('두번째 값 : ', w_new2, b_new2)

# 7. 전체 데이터를 순차적으로 돌면서 w, b를 반복적으로 업데이트
# - 한 번의 에폭(epoch)이며, 온라인 학습 방식 (데이터 1개씩 적용)
w = 1.0  # 다시 초기화
b = 1.0
for x_i, y_i in zip(x, y):
    y_hat = x_i * w + b     # 예측값 계산
    err = y_i - y_hat       # 오차 계산
    w_rate = x_i            # x가 기울기 역할
    w = w + w_rate * err    # w 업데이트
    b = b + 1 * err         # b 업데이트

# 최종적으로 학습된 w, b 출력
print(w, b)

# 8. 결과 시각화
# - 산점도: 실제 데이터 (x vs y)
# - 선형 회귀선: 학습된 w, b를 기반으로한 직선
plt.scatter(x, y)

# 회귀직선을 두 점으로 표현
# x = -0.1일 때와 x = 0.15일 때 예측값을 계산하여 직선으로 그림
pt1 = (-0.1, -0.1*w + b)
pt2 = (0.15, 0.15*w + b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])  # 회귀선 그리기

plt.show()  # 그래프 표시
