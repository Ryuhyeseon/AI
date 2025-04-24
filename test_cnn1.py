# -*- coding: utf-8 -*-

# 1. 필수 라이브러리 임포트 (이미지 조합 기준)
import tensorflow as tf             # TensorFlow 라이브러리
from tensorflow import keras          # TensorFlow 내 Keras API
from keras import layers, models, datasets, backend # Keras 구성 요소 및 백엔드
from tensorflow.python.keras.utils import np_utils    # Keras 유틸리티 (to_categorical 등)
import numpy as np                  # NumPy 라이브러리
import matplotlib.pyplot as plt     # Matplotlib 시각화 라이브러리
import os                           # 운영체제 인터페이스 모듈
from keraspp.skeras import plot_loss, plot_acc # 학습 결과 시각화 함수 임포트

# MKL 관련 오류 방지 설정 (이미지에 있음)
# os.environ['KMP_DUPLICATE_LIB_OK']='True' # 필요시 주석 해제

# 2. CNN 모델 정의 (Sequential 상속 - 이미지 기준)
class CNN(models.Sequential):
    # 모델 초기화 함수 (생성자)
    def __init__(self, input_shape, num_classes):
        # 부모 클래스(Sequential) 생성자 호출
        super().__init__()
        # 첫 번째 합성곱 블록: Conv2D(필터32,커널3x3,활성화relu) -> MaxPooling2D(2x2) -> Dropout(0.25)
        self.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        self.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.add(layers.Dropout(0.25))
        # Flatten 층: 다차원 특징맵을 1차원 벡터로 변환
        self.add(layers.Flatten())
        # 완전 연결 블록: Dense(128,활성화relu) -> Dropout(0.5)
        self.add(layers.Dense(128, activation='relu'))
        self.add(layers.Dropout(0.5))
        # 출력층: Dense(클래스수, 활성화softmax)
        self.add(layers.Dense(num_classes, activation='softmax'))

        # 모델 컴파일 설정 (손실함수, 옵티마이저, 평가지표)
        # self.compile(loss=keras.losses.categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy']) # 주석 처리된 원본 라인
        self.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']) # 이미지상의 활성 라인 (Keras 2 기준일 수 있음)

# 3. 데이터 처리 클래스 (이미지 기준)
# 데이터 전처리 하기
class DATA():
    # 클래스 초기화 함수 (생성자)
    def __init__(self):
        # 클래스 수 정의 (MNIST 기준)
        num_classes = 10
        # MNIST 데이터셋 로드
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data() # 원본: datasets -> keras.datasets
        # 이미지 차원(행, 열) 가져오기
        img_rows, img_cols = X_train.shape[1:] # 첫 번째 이미지의 높이, 너비

        # 이미지 데이터 포맷 확인 및 형태 변환 (CNN 입력에 맞게 채널 차원 추가)
        if backend.image_data_format() == 'channels_first': # 채널 우선순서 확인
            # (샘플수, 채널수, 높이, 너비) 형태로 변환
            X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
            X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols) # CNN 입력 형태 설정
        else: # 'channels_last' (TensorFlow 기본값)
            # (샘플수, 높이, 너비, 채널수) 형태로 변환 (흑백=1)
            X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
            X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1) # CNN 입력 형태 설정

        # 데이터 타입 변환 (정수 -> 부동소수점)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        # 픽셀 값 정규화 (0~255 -> 0.0~1.0)
        X_train /= 255.0
        X_test /= 255.0

        # 레이블(정수)을 원-핫 인코딩 벡터로 변환
        Y_train = np_utils.to_categorical(y_train, num_classes)
        Y_test = np_utils.to_categorical(y_test, num_classes)

        # 클래스 인스턴스 변수에 전처리된 데이터와 정보 저장
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.X_train, self.Y_train = X_train, Y_train
        self.X_test, self.Y_test = X_test, Y_test

# 4. 메인 실행 함수 (이미지 기준)
def main():
    # 학습 파라미터 설정
    batch_size = 128 # 배치 크기
    epochs = 10      # 학습 에포크 수

    # 데이터 객체 생성하여 데이터 로드 및 전처리 수행
    data = DATA()

    # CNN 모델 생성 (DATA 객체로부터 입력 형태와 클래스 수 전달)
    model = CNN(data.input_shape, data.num_classes)
    # model.summary() # 원본 코드에는 summary 호출 없음

    # 모델 학습
    history = model.fit(data.X_train, data.Y_train,
                        batch_size=batch_size, epochs=epochs,
                        validation_split=0.2, # 검증 데이터 비율
                        verbose=1) # 학습 로그 출력 설정 (원본 이미지 기준)

    # 모델 평가 (테스트 데이터 사용)
    score = model.evaluate(data.X_test, data.Y_test, batch_size=batch_size, verbose=0) # verbose=0 추가
    # 테스트 결과(손실, 정확도) 출력
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # 그래프 그리기
    # 손실 그래프 표시
    plot_loss.plot_loss(history, title='(a) Loss Trend') # 외부 라이브러리 함수 호출
    plt.show() # 그래프 창을 보여줌
    # 정확도 그래프 표시
    plot_acc.plot_acc(history, title='(b) Accuracy Trend') # 외부 라이브러리 함수 호출
    plt.show() # 그래프 창을 보여줌

# 5. 스크립트 실행 지점
# 이 파일이 직접 실행될 때 main() 함수 호출
if __name__ == '__main__':
    main()