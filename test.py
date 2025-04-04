# 0. 필요한 라이브러리 가져오기
import tensorflow as tf  # 텐서플로우 라이브러리 (딥러닝 프레임워크)
from tensorflow import keras  # 케라스 API (텐서플로우 위에서 동작하는 고수준 딥러닝 API)
import numpy as np  # NumPy 라이브러리 (수치 계산, 배열 처리)
import matplotlib.pyplot as plt  # Matplotlib 라이브러리 (데이터 시각화, 그래프 그리기)
from sklearn.model_selection import cross_validate, train_test_split  # Scikit-learn 라이브러리 (머신러닝 도구)
# cross_validate: 교차 검증 수행 함수
# train_test_split: 데이터를 훈련 세트와 테스트(또는 검증) 세트로 나누는 함수
from sklearn.linear_model import SGDClassifier  # Scikit-learn 라이브러리
# SGDClassifier: 확률적 경사 하강법을 사용하는 선형 분류 모델

# --- 데이터 준비 ---

# 1. 데이터 로드 (Fashion MNIST 데이터셋)
# 케라스에 내장된 Fashion MNIST 데이터셋을 불러옵니다.
# 이 데이터셋은 10 종류의 패션 아이템에 대한 28x28 픽셀 흑백 이미지로 구성됩니다.
# load_data() 함수는 데이터를 (훈련 입력, 훈련 타겟), (테스트 입력, 테스트 타겟) 튜플 형태로 반환합니다.
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

# 데이터 크기(shape) 출력
# train_input: (60000, 28, 28) -> 60000개의 훈련 샘플, 각 샘플은 28x28 크기의 2차원 배열(이미지)
# train_target: (60000,) -> 60000개의 훈련 샘플에 대한 레이블(정답, 0~9 사이의 숫자)
# test_input: (10000, 28, 28) -> 10000개의 테스트 샘플
# test_target: (10000,) -> 10000개의 테스트 샘플에 대한 레이블
print("훈련 데이터 모양:", train_input.shape, train_target.shape)
print("테스트 데이터 모양:", test_input.shape, test_target.shape)

# 2. 샘플 이미지 출력 (처음 10개 데이터 시각화)
# Matplotlib을 사용하여 훈련 데이터의 첫 10개 이미지를 화면에 표시합니다.
# 이를 통해 데이터가 어떻게 생겼는지 눈으로 확인할 수 있습니다.
fig, axs = plt.subplots(1, 10, figsize=(10, 10))  # 1행 10열의 서브플롯(작은 그래프 영역) 생성
for i in range(10):  # 0부터 9까지 반복
    axs[i].imshow(train_input[i], cmap='gray_r')  # i번째 이미지를 흑백 반전(밝은 부분이 0, 어두운 부분이 255에 가깝게)으로 표시
    axs[i].axis('off')  # 이미지 축(눈금)을 표시하지 않음
plt.show()  # 준비된 그래프를 화면에 보여줌

# 3. 첫 10개 데이터의 레이블(정답) 출력
# 위에서 출력한 10개 이미지에 해당하는 실제 정답값(패션 아이템 종류)을 확인합니다.
# 레이블은 0부터 9까지의 숫자로 표현됩니다 (예: 0=T-shirt/top, 1=Trouser, ..., 9=Ankle boot).
print("처음 10개 레이블:", [train_target[i] for i in range(10)])

# 4. 타겟 데이터의 고유값과 개수 확인 (클래스 분포 확인)
# 훈련 데이터의 레이블(train_target)에 어떤 종류의 값들이 있고, 각각 몇 개씩 있는지 확인합니다.
# np.unique 함수는 고유한 값들과 각 값의 개수(return_counts=True)를 반환합니다.
# 이를 통해 각 패션 아이템 클래스가 데이터셋에 얼마나 균등하게 분포되어 있는지 알 수 있습니다.
print("클래스 분포:", np.unique(train_target, return_counts=True))

# 5. 데이터 정규화 (0~1 범위로 조정)
# 이미지 픽셀 값은 보통 0부터 255 사이의 정수입니다.
# 머신러닝/딥러닝 모델은 일반적으로 입력 값의 범위가 작을 때 (예: 0~1 또는 -1~1) 더 잘 작동합니다.
# 따라서 모든 픽셀 값을 255.0 (실수 나눗셈을 위해 .0 추가)으로 나누어 0~1 사이의 값으로 변환합니다.
# 이 과정을 '정규화(Normalization)' 또는 '스케일링(Scaling)'이라고 합니다.
train_scaled = train_input / 255.0

# 6. 데이터 형태 변환 (2차원 이미지 -> 1차원 벡터)
# SGDClassifier 같은 기본 머신러닝 모델이나 완전 연결 신경망(Dense layer)은
# 주로 1차원 형태의 입력을 받습니다.
# 따라서 각 28x28 크기의 2차원 이미지 데이터를 784개(28 * 28 = 784)의 요소를 가진
# 1차원 배열(벡터)로 펼쳐줍니다.
# reshape(-1, 784)는 "전체 데이터 개수(-1, 자동으로 계산됨) x 784개의 특성" 형태의 2차원 배열로 변환하라는 의미입니다.
train_scaled = train_scaled.reshape(-1, 28 * 28)

# 테스트 데이터도 동일하게 전처리합니다. (나중에 신경망 평가 시 사용)
test_scaled = test_input / 255.0
test_scaled = test_scaled.reshape(-1, 28 * 28)

print("정규화 및 1차원 변환 후 훈련 데이터 모양:", train_scaled.shape) # (60000, 784)
print("정규화 및 1차원 변환 후 테스트 데이터 모양:", test_scaled.shape) # (10000, 784)


# --- 머신러닝 (SGDClassifier) ---
print("\n--- SGDClassifier 학습 및 평가 ---")

# 7. 확률적 경사 하강법(SGD) 분류 모델 생성 및 학습 준비
# SGDClassifier 모델 객체를 생성합니다.
# loss='log_loss': 손실 함수로 로지스틱 손실(크로스 엔트로피와 유사)을 사용합니다. 분류 문제에 적합합니다.
# max_iter=100: 훈련 데이터를 최대 100번 반복 학습합니다 (에포크 수). 이전의 5번보다 훨씬 많이 학습하도록 증가시켰습니다.
# random_state=42: 난수 발생 시드를 고정하여 실행할 때마다 동일한 결과를 얻도록 합니다 (재현성).
# tol=1e-3: 학습 중 성능 개선이 이 값보다 작아지면 조기 종료합니다. (성능 개선이 미미하면 멈춤)
sc = SGDClassifier(loss='log_loss', max_iter=100, random_state=42, tol=1e-3)
# 참고: 교차 검증을 사용하면 fit 메서드를 직접 호출하지 않아도 됩니다. cross_validate 함수 내부에서 수행됩니다.
# sc.fit(train_scaled, train_target) # 이 줄은 교차 검증 시 필요 없음

# 8. 교차 검증(Cross-Validation) 수행
# 모델의 일반화 성능을 더 신뢰성 있게 평가하기 위해 교차 검증을 사용합니다.
# cross_validate 함수는 데이터를 여러 개의 폴드(기본값 5개)로 나누고,
# 각 폴드를 한 번씩 테스트 세트로 사용하고 나머지를 훈련 세트로 사용하여 모델을 여러 번 훈련하고 평가합니다.
# sc: 평가할 모델 객체
# train_scaled, train_target: 전체 훈련 데이터와 레이블
# n_jobs=-1: 컴퓨터의 모든 CPU 코어를 사용하여 병렬로 교차 검증을 수행하여 시간을 단축합니다.
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
# scores 딕셔너리에는 각 폴드별 훈련 시간, 테스트 시간, 테스트 점수 등이 저장됩니다.
# 'test_score' 키에 저장된 값들의 평균을 계산하여 평균적인 모델 성능(정확도)을 확인합니다.
print("SGDClassifier 교차 검증 평균 정확도:", np.mean(scores['test_score']))


# --- 인공 신경망 (Keras) ---
print("\n--- 인공 신경망 학습 및 평가 ---")

# 9. 데이터셋을 훈련 세트와 검증 세트로 분리
# 신경망 모델을 훈련하고 평가하기 위해, 기존의 훈련 데이터(train_scaled)를 다시
# 새로운 훈련 세트(train_scaled)와 검증 세트(val_scaled)로 나눕니다.
# 검증 세트는 모델 훈련 중에 매 에포크마다 성능을 측정하여,
# 모델이 과적합(훈련 데이터에만 너무 잘 맞고 새로운 데이터에는 잘 못 맞추는 현상)되는지 확인하거나
# 하이퍼파라미터(예: 에포크 수, 학습률 등)를 튜닝하는 데 사용됩니다.
# test_size=0.2: 전체 데이터 중 20%를 검증 세트로 사용 (나머지 80%는 훈련 세트)
# random_state=42: 데이터를 무작위로 섞어 나눌 때, 항상 동일한 방식으로 나누도록 시드를 고정합니다.
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

# 분리된 데이터의 크기 확인
print("신경망 훈련 데이터 모양:", train_scaled.shape, train_target.shape) # (48000, 784) (48000,)
print("신경망 검증 데이터 모양:", val_scaled.shape, val_target.shape)   # (12000, 784) (12000,)

# 10. 인공 신경망 모델 구성
# 케라스의 Sequential API를 사용하여 모델을 만듭니다. Sequential 모델은 층(layer)을 순서대로 쌓는 방식입니다.
model = keras.Sequential([
    # 첫 번째 층 (은닉층, Hidden Layer): 완전 연결층(Dense)
    # 128: 이 층의 뉴런(노드) 개수. (이 값은 조절 가능한 하이퍼파라미터입니다)
    # activation='relu': 활성화 함수로 ReLU(Rectified Linear Unit)를 사용합니다.
    #                  음수 입력은 0으로, 양수 입력은 그대로 출력하는 비선형 함수로, 딥러닝에서 널리 쓰입니다.
    # input_shape=(784,): 첫 번째 층에는 입력 데이터의 형태를 지정해야 합니다.
    #                     각 샘플이 784개의 특성을 가진 1차원 배열임을 의미합니다.
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),

    # 두 번째 층 (출력층, Output Layer): 완전 연결층(Dense)
    # 10: 출력층의 뉴런 개수. 분류하려는 클래스(패션 아이템 종류)의 개수와 동일해야 합니다 (0~9까지 10개).
    # activation='softmax': 활성화 함수로 Softmax를 사용합니다.
    #                     출력층의 뉴런 값들을 각 클래스에 대한 확률로 변환해줍니다.
    #                     모든 클래스에 대한 확률의 합은 1이 됩니다. 다중 클래스 분류 문제의 출력층에 주로 사용됩니다.
    keras.layers.Dense(10, activation='softmax')
])

# 모델의 구조를 요약하여 출력합니다 (각 층의 이름, 출력 형태, 파라미터 개수 등).
model.summary()

# 11. 모델 컴파일: 학습 과정 설정
# 모델을 훈련하기 전에, 어떤 방식으로 학습할지 설정해야 합니다.
model.compile(
    # optimizer='adam': 옵티마이저(최적화 알고리즘)로 'adam'을 사용합니다.
    #                  Adam은 현재 가장 널리 쓰이고 성능이 좋은 옵티마이저 중 하나입니다. (SGD의 개선된 버전)
    optimizer='adam',
    # loss='sparse_categorical_crossentropy': 손실 함수(모델의 예측이 실제 정답과 얼마나 다른지 측정하는 함수) 설정.
    #                                        'sparse_categorical_crossentropy'는 타겟(정답) 레이블이
    #                                        정수 형태(0, 1, 2, ...)일 때 사용하는 다중 클래스 분류용 손실 함수입니다.
    #                                        (만약 타겟이 원-핫 인코딩 형태라면 'categorical_crossentropy' 사용)
    loss='sparse_categorical_crossentropy',
    # metrics=['accuracy']: 훈련 및 평가 과정에서 측정하고 기록할 지표를 설정합니다.
    #                      'accuracy'는 정확도(전체 예측 중 맞은 예측의 비율)를 의미합니다.
    metrics=['accuracy']
)

# 12. 신경망 모델 학습 (훈련)
# model.fit() 함수를 사용하여 모델을 훈련시킵니다.
print("신경망 모델 학습 시작...")
history = model.fit(
    train_scaled,          # 훈련 입력 데이터 (새로 나눠진 훈련 세트)
    train_target,          # 훈련 타겟 데이터 (레이블)
    epochs=20,             # 에포크 수: 전체 훈련 데이터를 몇 번 반복 학습할지 지정합니다. (이전 5번 -> 20번으로 증가)
    validation_data=(val_scaled, val_target) # 검증 데이터: 매 에포크 학습이 끝날 때마다 이 데이터로 모델 성능을 평가합니다.
                                             # 이를 통해 학습 과정을 모니터링하고 과적합 여부를 판단할 수 있습니다.
)
print("신경망 모델 학습 완료.")

# history 객체에는 에포크별 훈련 손실(loss), 훈련 정확도(accuracy),
# 검증 손실(val_loss), 검증 정확도(val_accuracy)가 기록되어 있습니다.

# 13. 학습 과정 시각화
# Matplotlib을 사용하여 에포크별 손실과 정확도 변화를 그래프로 그립니다.
# 이를 통해 모델이 잘 학습되고 있는지, 과적합이 발생하는지 등을 시각적으로 확인할 수 있습니다.
plt.figure(figsize=(12, 4)) # 그래프 크기 설정

# 손실(Loss) 그래프
plt.subplot(1, 2, 1) # 1행 2열 중 첫 번째 위치에 그래프 생성
plt.plot(history.history['loss'], label='Train Loss') # 훈련 손실 변화
plt.plot(history.history['val_loss'], label='Validation Loss') # 검증 손실 변화
plt.title('Loss Over Epochs') # 그래프 제목
plt.xlabel('Epochs') # x축 레이블
plt.ylabel('Loss') # y축 레이블
plt.legend() # 범례 표시

# 정확도(Accuracy) 그래프
plt.subplot(1, 2, 2) # 1행 2열 중 두 번째 위치에 그래프 생성
plt.plot(history.history['accuracy'], label='Train Accuracy') # 훈련 정확도 변화
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') # 검증 정확도 변화
plt.title('Accuracy Over Epochs') # 그래프 제목
plt.xlabel('Epochs') # x축 레이블
plt.ylabel('Accuracy') # y축 레이블
plt.legend() # 범례 표시

plt.tight_layout() # 서브플롯 간 간격 자동 조절
plt.show() # 그래프 화면에 표시

# 14. 검증 세트로 모델 최종 성능 평가
# 훈련이 완료된 모델을 사용하여 검증 세트에 대한 최종 손실과 정확도를 계산합니다.
print("\n검증 세트 평가 결과:")
# model.evaluate() 함수는 손실과 설정된 metrics(여기서는 accuracy) 값을 반환합니다.
# verbose=0 옵션은 평가 진행 상황 막대를 출력하지 않도록 합니다.
val_loss, val_accuracy = model.evaluate(val_scaled, val_target, verbose=0)
print(f"검증 손실: {val_loss:.4f}") # 소수점 4자리까지 출력
print(f"검증 정확도: {val_accuracy:.4f}") # 소수점 4자리까지 출력

# 15. 테스트 세트로 모델 최종 성능 평가 (일반화 성능 확인)
# 모델 훈련 및 검증 과정에서 전혀 사용되지 않은 '테스트 세트'를 사용하여
# 모델의 최종 성능, 즉 새로운 데이터에 대한 일반화 성능을 평가합니다.
# 이 성능이 모델의 실제 성능에 가장 가깝다고 볼 수 있습니다.
print("\n테스트 세트 평가 결과:")
test_loss, test_accuracy = model.evaluate(test_scaled, test_target, verbose=0)
print(f"테스트 손실: {test_loss:.4f}")
print(f"테스트 정확도: {test_accuracy:.4f}")