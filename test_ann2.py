# -*- coding: utf-8 -*-

# 1. 필수 라이브러리 임포트
import tensorflow as tf                    # TensorFlow 라이브러리
from tensorflow import keras               # TensorFlow 내 Keras API
from keras import layers, models, datasets # Keras 구성 요소 (원본 방식 유지)
import numpy as np                         # NumPy 라이브러리
import matplotlib.pyplot as plt            # Matplotlib 시각화 라이브러리
from sklearn import preprocessing          # Scikit-learn 데이터 전처리 도구

# 2. 회기를 위한 ANN 모델 정의 (Model 서브클래싱)
class ANN(models.Model):
    # 모델 초기화 함수 (생성자)
    def __init__(self, Nin, Nh, Nout):
        # 필요한 레이어 정의
        hidden = layers.Dense(Nh) # 은닉층 (유닛 수: Nh)
        output = layers.Dense(Nout) # 출력층 (유닛 수: Nout)
        relu = layers.Activation('relu') # 활성화 함수: ReLU

        # 모델 구조 정의 (함수형 API 스타일)
        x = layers.Input(shape=(Nin, )) # 입력층
        h = relu(hidden(x)) # 은닉층 = Dense -> ReLU
        y = output(h) # 출력층 = Dense (선형 활성화)

        # 부모 클래스 생성자 호출 (모델 구성 완료)
        super().__init__(inputs=x, outputs=y)

        # 모델 컴파일 (손실함수, 옵티마이저 설정)
        self.compile(loss='mse', optimizer='sgd') # 회귀 문제이므로 MSE 손실, SGD 옵티마이저 사용

# 3. 데이터 전처리 함수
def Data_func():
    # Keras 보스턴 주택 가격 데이터셋 로드
    (X_train, Y_train), (x_test, y_test) = datasets.boston_housing.load_data()
    # MinMaxScaler를 사용하여 데이터 스케일링 (0~1 범위)
    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train) # 훈련 데이터 학습 및 변환
    x_test = scaler.transform(x_test) # 테스트 데이터는 학습된 스케일러로 변환만
    # 전처리된 데이터 반환
    return (X_train, Y_train), (x_test, y_test)

# 4. 학습 결과 시각화 함수 (손실 그래프)
def plot_loss(history, title=None):
    # history 객체에서 loss, val_loss 추출하여 그래프 그리기
    if not isinstance(history, dict):
        history = history.history # Keras History 객체에서 딕셔너리 추출
    plt.plot(history['loss']) # 훈련 손실 플롯
    if 'val_loss' in history: # 검증 손실이 있으면 플롯
        plt.plot(history['val_loss'])
    if title is not None: # 제목이 주어지면 설정
        plt.title(title)
    plt.ylabel('Loss (MSE)') # Y축 레이블
    plt.xlabel('Epochs') # X축 레이블
    legend_items = ['Training Loss'] # 범례 항목 초기화
    if 'val_loss' in history:
        legend_items.append('Validation Loss') # 검증 손실 항목 추가
    plt.legend(legend_items, loc='upper right') # 범례 표시

# 5. 메인 실행 함수
def main():
    # 신경망 파라미터 설정
    Nin = 13 # 입력 뉴런 수 (보스턴 데이터 특성 수)
    Nh = 5 # 은닉층 뉴런 수
    Nout = 1 # 출력 뉴런 수 (주택 가격)

    # 모델 생성
    model = ANN(Nin, Nh, Nout)
    model.summary() # 모델 구조 출력

    # 데이터 가져오기 및 전처리
    (X_train, Y_train), (x_test, y_test) = Data_func()

    # 모델 학습
    history = model.fit(X_train, Y_train, epochs=100, batch_size=100, validation_split=0.2, verbose=2)

    # 모델 평가 (테스트 데이터)
    performance_test = model.evaluate(x_test, y_test, batch_size=100, verbose=0)
    # 테스트 손실(MSE) 출력
    print('\nTest Loss (MSE) -> {:.4f}'.format(performance_test)) # evaluate 반환값 직접 사용

    # 학습 손실 그래프 그리기
    plot_loss(history, title='Loss (MSE) Trend')
    plt.show() # 그래프 화면 표시

# 6. 스크립트 실행 지점
if __name__ == '__main__':
    main() # 메인 함수 호출