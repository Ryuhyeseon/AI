# 필요한 라이브러리 불러오기
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

# --- 데이터 불러오기 및 준비 ---
# sklearn에서 제공하는 당뇨병 데이터셋 로드
diabetes = load_diabetes()

# 10개의 특성 중 2번째 특성만 선택 (보통 'BMI'와 관련된 값)
x = diabetes.data[:, 2]  # 입력 데이터 (특성)
y = diabetes.target      # 목표 데이터 (타겟값)
# ----------------------------------------------------------

# --- 뉴런 클래스 정의 ---
class Neuron:
    def __init__(self):
        # 가중치(w)와 절편(b) 초기화 (모두 1.0으로 시작)
        self.w = 1.0
        self.b = 1.0

    # 정방향 계산 (y = wx + b)
    def forpass(self, x):
        return x * self.w + self.b

    # 역전파: 오차를 이용하여 가중치와 절편의 기울기 계산
    def backprop(self, x, err):
        w_grad = x * err      # 가중치에 대한 기울기
        b_grad = 1 * err      # 절편에 대한 기울기 (항상 err)
        return w_grad, b_grad

    # 학습 함수 (경사하강법 적용)
    def fit(self, x, y, epochs=100):
        for i in range(epochs):  # 지정된 에폭 수만큼 반복
            for x_i, y_i in zip(x, y):  # 각 데이터를 순서대로 학습
                # 예측값 계산 (순전파)
                y_hat = self.forpass(x_i)
                # 오차 계산 (실제값 - 예측값)
                err = -(y_i - y_hat)  # 마이너스를 붙여서 경사하강법처럼 사용
                # 기울기 계산 (역전파)
                w_grad, b_grad = self.backprop(x_i, err)
                # 가중치와 절편 업데이트 (학습률 없이 단순 더하기 방식)
                self.w = self.w + w_grad
                self.b = self.b + b_grad
# ----------------------------------------------------------

# --- 뉴런 생성 및 학습 ---
# 뉴런 객체 생성
neuron = Neuron()

# 뉴런 학습 (특성 x와 타겟 y를 이용해 100 에폭 동안 학습)
neuron.fit(x, y)

# 학습된 직선과 원본 데이터를 시각화
plt.scatter(x, y)  # 원본 데이터 산점도

# 학습된 직선을 그리기 위해 두 점 계산
pt1 = (-0.1, -0.1 * neuron.w + neuron.b)
pt2 = (0.15, 0.15 * neuron.w + neuron.b)

# 예측 직선 시각화 (빨간색 선)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r')

# x축, y축 라벨 설정
plt.xlabel('x (feature)')
plt.ylabel('y (target)')

# 그래프 출력
plt.title("Neuron Linear Fit Result")
plt.show()

# 최종 학습된 가중치와 절편 출력
print(f"최종 가중치 (w): {neuron.w}")
print(f"최종 절편 (b): {neuron.b}")
