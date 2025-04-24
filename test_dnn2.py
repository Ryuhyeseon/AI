# -*- coding: utf-8 -*- # 파일 인코딩 UTF-8 명시

# 1. 필수 라이브러리 임포트
import sys                      # 시스템 관련 모듈
import os                       # 운영체제 인터페이스 모듈
import numpy as np              # 수치 계산 라이브러리 (별칭 np)
import tensorflow as tf             # TensorFlow 라이브러리
from tensorflow import keras          # TensorFlow 내 Keras API
from keras import layers, models, datasets # Keras 구성 요소
from tensorflow.python.keras.utils import np_utils    # Keras 유틸리티 (to_categorical 등)
import matplotlib.pyplot as plt     # 데이터 시각화 라이브러리 (별칭 plt)
from keraspp.skeras import plot_loss, plot_acc # 학습 결과 시각화 함수 임포트

# MKL 관련 환경 변수 설정 (특정 환경 오류 방지용)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 2. DNN 모델 정의 (Sequential 상속, Dropout 포함)
class DNN(models.Sequential):
    """
    Sequential 모델을 상속받아 Dropout을 포함한 DNN 모델을 정의하는 클래스입니다.
    """
    # 모델 초기화 함수 (생성자)
    def __init__(self, Nin, Nh_l, Pd_l, Nout):
        # 부모 클래스(Sequential) 생성자 호출
        super().__init__(name='DNN_with_Dropout')
        # 입력층 역할 겸 첫 번째 은닉층 추가 (Dense + ReLU)
        self.add(layers.Dense(Nh_l[0], activation='relu', input_shape=(Nin,), name='Hidden-1'))
        # 첫 번째 Dropout 층 추가 (과적합 방지 목적)
        self.add(layers.Dropout(Pd_l[0]))
        # 두 번째 은닉층 추가 (Dense + ReLU)
        self.add(layers.Dense(Nh_l[1], activation='relu', name='Hidden-2'))
        # 두 번째 Dropout 층 추가 (과적합 방지 목적)
        self.add(layers.Dropout(Pd_l[1]))
        # 출력층 추가 (Dense + Softmax - 다중 클래스 분류)
        self.add(layers.Dense(Nout, activation='softmax'))
        # 모델 컴파일 (손실함수, 옵티마이저, 평가 지표 설정)
        self.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])

# 3. 데이터 전처리 함수 (CIFAR-10)
# 데이터 전처리를 위한 함수
def Data_func():
    """
    Keras 내장 CIFAR-10 데이터셋을 로드하고 신경망 학습에 맞게 전처리합니다.
    """
    # CIFAR-10 데이터셋 로드 (32x32 컬러 이미지, 10개 클래스)
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # 레이블(0-9 정수)을 원-핫 인코딩 벡터로 변환
    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    # 입력 데이터(이미지) 형태 확인 및 전처리
    L, W, H, C = X_train.shape # L: 샘플 수, W: 너비, H: 높이, C: 채널 수
    # 형태 정보 출력
    print(X_train.shape)

    # 이미지를 1D 벡터로 펼치기 (W*H*C 크기)
    X_train = X_train.reshape(-1, W*H*C)
    X_test = X_test.reshape(-1, W*H*C)

    # 픽셀 값을 [0, 1] 범위로 정규화하고 float32 타입으로 변환
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # 전처리된 데이터를 튜플로 묶어 반환
    return (X_train, Y_train), (X_test, Y_test)

# 4. 메인 실행 함수
def main():
    """ 메인 로직: 파라미터 설정, 데이터 로드, 모델 생성, 학습, 평가, 시각화 """
    # 하이퍼파라미터 설정
    Nh_l = [100, 50]     # 각 은닉층의 뉴런 수 리스트
    Pd_l = [0.0, 0.0]    # 각 은닉층 뒤 Dropout 비율 리스트
    number_of_class = 10 # 분류할 클래스 수
    Nout = number_of_class # 출력층 뉴런 수

    # 데이터 로드 및 전처리 함수 호출
    (X_train, Y_train), (X_test, Y_test) = Data_func()
    # 입력층 뉴런 수 계산 (데이터 로드 후 결정)
    Nin = X_train.shape[1]

    # DNN 모델 생성
    model = DNN(Nin, Nh_l, Pd_l, Nout)
    # model.summary() # 모델 구조 출력 (주석 처리됨)

    # 모델 학습 실행
    history = model.fit(X_train, Y_train, epochs=10, batch_size=100, validation_split=0.2, verbose=2)

    # 학습된 모델 평가 (테스트 데이터 사용)
    performace_test = model.evaluate(X_test, Y_test, batch_size=100)
    # 테스트 결과(손실, 정확도) 출력
    print('Test Loss and Accuracy :', performace_test)

    # 그래프 그리기
    # 손실(Loss) 변화 그래프 표시
    plot_loss.plot_loss(history, title='(a) 손실 추이')
    plt.show() # 그래프 창을 보여줌
    # 정확도(Accuracy) 변화 그래프 표시
    plot_acc.plot_acc(history, title='(b) 정확도 추이')
    plt.show() # 그래프 창을 보여줌

# 5. 스크립트 실행 지점
# 이 파일이 직접 실행될 때 main() 함수 호출
if __name__ == '__main__':
    main()