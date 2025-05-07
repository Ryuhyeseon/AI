from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- 1. 데이터 로드 및 전처리 ---

# 유방암 데이터셋 로드
cancer = load_breast_cancer()

# 입력 데이터(x)와 타겟 레이블(y) 설정
x = cancer.data  # 569개의 샘플, 30개의 특성
y = cancer.target  # 0: 악성(malignant), 1: 양성(benign)

# 훈련/테스트 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size=0.2, random_state=42
)

# 표준화 (스케일링): 평균 0, 표준편차 1로 변환
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)  # 테스트셋은 fit 없이 transform만

# --- 2. 로지스틱 뉴런 구현 ---

class LogisticNeuron:
    def __init__(self, learning_rate=0.01):
        """
        로지스틱 뉴런 초기화
        - w: 가중치 벡터 (각 특성에 대응)
        - b: 절편 (bias)
        - lr: 학습률
        """
        self.w = None
        self.b = None
        self.lr = learning_rate

    def forpass(self, x_sample):
        """
        순전파(forward pass): 선형 결합 계산
        z = w1*x1 + w2*x2 + ... + wn*xn + b
        """
        return np.sum(x_sample * self.w) + self.b

    def activation(self, z):
        """
        시그모이드 함수 (로지스틱 활성화 함수)
        y_hat = 1 / (1 + e^(-z))
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, x, y, epochs=100):
        """
        학습 함수
        - x: 입력 특성 (2D 배열)
        - y: 타겟 레이블 (1D 배열)
        - epochs: 전체 데이터셋을 몇 번 반복할지 지정
        """
        self.w = np.ones(x.shape[1])  # 특성 개수만큼 가중치 초기화
        self.b = 0  # 절편 초기화

        for _ in range(epochs):  # 에포크 수만큼 반복
            for x_i, y_i in zip(x, y):  # 각 샘플에 대해
                z = self.forpass(x_i)
                a = self.activation(z)
                err = y_i - a  # 오차 계산

                # 가중치와 절편 업데이트 (경사 하강법)
                self.w += self.lr * err * x_i
                self.b += self.lr * err

    def predict(self, x):
        """
        예측 함수
        - 입력 데이터 x에 대해 예측값 반환 (0 또는 1)
        """
        z = [self.forpass(x_i) for x_i in x]  # 각 샘플에 대해 z 계산
        a = self.activation(np.array(z))  # 시그모이드 적용
        return a > 0.5  # 0.5보다 크면 양성(1), 아니면 악성(0)

    def score(self, x, y):
        """
        정확도(Accuracy) 평가
        - 예측 결과와 실제 정답 비교하여 평균 정확도 반환
        """
        return np.mean(self.predict(x) == y)

# --- 3. 모델 학습 및 평가 ---

# 로지스틱 뉴런 객체 생성 및 학습
neuron = LogisticNeuron(learning_rate=0.01)
neuron.fit(x_train_scaled, y_train, epochs=100)

# 훈련 및 테스트 정확도 출력
print(f"Train Accuracy: {neuron.score(x_train_scaled, y_train):.4f}")
print(f"Test Accuracy: {neuron.score(x_test_scaled, y_test):.4f}")