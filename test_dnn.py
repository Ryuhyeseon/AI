# -*- coding: utf-8 -*- # 파일 인코딩을 UTF-8로 명시 (주석 등에 한글 사용 시 권장)

# 1. 필수 라이브러리 임포트
import sys                                         # 시스템 관련 파라미터 및 함수 접근 (예: 최대 출력 크기 설정)
import os                                          # 운영체제 기능 접근 (이 코드에서는 직접 사용되지 않음)
import numpy as np                                 # 수치 계산, 배열/행렬 연산 라이브러리 (별칭 np)
from tensorflow import keras                       # TensorFlow 백엔드를 사용하는 Keras API
from keras import layers, models                   # Keras 모델 구축을 위한 층(layers) 및 모델(models) 클래스/함수
from keras import datasets                         # Keras에 내장된 데이터셋 (예: MNIST)
from tensorflow.python.keras.utils import np_utils # One-hot 인코딩 등 유틸 함수 제공
import matplotlib.pyplot as plt                    # 데이터 시각화 라이브러리 (별칭 plt)

# 특정 환경(주로 macOS의 Jupyter 환경)에서 Intel MKL 관련 오류 방지 설정
# 여러 버전의 MKL 라이브러리가 로드될 때 발생하는 충돌을 허용합니다.
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 2. NumPy 출력 옵션 설정
# 배열을 출력할 때 생략 없이 모든 요소를 표시하도록 설정
np.set_printoptions(threshold=sys.maxsize)

# 3. 모델 정의 (4가지 방식)

# 방식 1: 함수형 API (함수 정의)
def ANN_model_fun(Nin, Nh, Nout):
    """
    함수형 API를 사용하여 간단한 ANN 모델(입력-은닉-출력)을 생성하고 컴파일합니다.

    Args:
        Nin (int): 입력층 뉴런 수 (입력 데이터의 특성 수)
        Nh (int): 은닉층 뉴런 수
        Nout (int): 출력층 뉴런 수 (분류할 클래스 수)

    Returns:
        keras.models.Model: 컴파일된 Keras 모델 객체 (함수형 API 기반)
    """
    x = layers.Input(shape=(Nin,), name='input_layer')
    h = layers.Activation('relu', name='hidden_activation')(layers.Dense(Nh, name='hidden_dense')(x))
    y = layers.Activation('softmax', name='output_activation')(layers.Dense(Nout, name='output_dense')(h))
    model = models.Model(inputs=x, outputs=y, name='Functional_ANN_Model')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 방식 2: Sequential API (함수 정의)
def ANN_seq_func(Nin, Nh, Nout):
    """
    Sequential API를 사용하여 간단한 ANN 모델(입력-은닉-출력)을 생성하고 컴파일합니다.

    Args:
        Nin (int): 입력층 뉴런 수 (input_shape으로 지정)
        Nh (int): 은닉층 뉴런 수
        Nout (int): 출력층 뉴런 수

    Returns:
        keras.models.Sequential: 컴파일된 Keras 모델 객체 (Sequential API 기반)
    """
    model = models.Sequential(name='Sequential_ANN_Function')
    model.add(layers.Dense(Nh, activation='relu', input_shape=(Nin,), name='hidden_dense_relu'))
    model.add(layers.Dense(Nout, activation='softmax', name='output_dense_softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 방식 3: 함수형 API (클래스 정의 - Model 서브클래싱)
class ANN_models_class(models.Model):
    """
    Model 클래스를 상속받아 함수형 API 스타일로 ANN 모델을 정의하는 클래스입니다.
    """
    def __init__(self, Nin, Nh, Nout):
        super().__init__(name='Functional_ANN_Class')
        self.hidden_layer = layers.Dense(Nh, name='hidden_dense')
        self.output_layer = layers.Dense(Nout, name='output_dense')
        self.relu_activation = layers.Activation('relu', name='hidden_activation')
        self.softmax_activation = layers.Activation('softmax', name='output_activation')

        # 모델 구조 정의 (함수형 API 스타일)
        x = layers.Input(shape=(Nin,), name='input_layer')
        h = self.relu_activation(self.hidden_layer(x))
        y = self.softmax_activation(self.output_layer(h))

        # Model 생성자 재호출 (입력과 출력 텐서 전달)
        super().__init__(inputs=x, outputs=y)
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 방식 4: Sequential API (클래스 정의 - Sequential 서브클래싱)
class ANN_seq_class(models.Sequential):
    """
    Sequential 클래스를 상속받아 Sequential API 스타일로 ANN 모델을 정의하는 클래스입니다.
    """
    def __init__(self, Nin, Nh, Nout):
        super().__init__(name='Sequential_ANN_Class')
        self.add(layers.Dense(Nh, activation='relu', input_shape=(Nin,), name='hidden_dense_relu'))
        self.add(layers.Dense(Nout, activation='softmax', name='output_dense_softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 데이터 로드 및 전처리 함수
def Data_func():
    """
    Keras 내장 MNIST 데이터셋을 로드하고 신경망 학습에 맞게 전처리합니다.
    """
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    # 레이블 원-핫 인코딩 (keras.utils.to_categorical 사용 권장)
    Y_train = np_utils.to_categorical(y_train) # 원본 코드 유지
    Y_test = np_utils.to_categorical(y_test)   # 원본 코드 유지

    print("Shape of Y_train:", Y_train.shape)

    img_width, img_height = X_train.shape[1], X_train.shape[2]
    num_pixels = img_width * img_height

    # 이미지 1D 변환 및 정규화
    X_train = X_train.reshape(-1, num_pixels).astype('float32') / 255.0
    X_test = X_test.reshape(-1, num_pixels).astype('float32') / 255.0

    return (X_train, Y_train), (X_test, Y_test)

# 5. 학습 결과 시각화 함수

# 정확도 그래프 함수
def plot_acc(history, title=None):
    """ 모델 학습 history에서 훈련/검증 정확도를 시각화합니다. """
    if not isinstance(history, dict):
        history = history.history
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')

# 손실 그래프 함수
def plot_loss(history, title=None):
    """ 모델 학습 history에서 훈련/검증 손실을 시각화합니다. """
    if not isinstance(history, dict):
        history = history.history
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')

# 6. 메인 실행 함수
def main():
    """ 데이터 로드, 모델 생성/학습/평가, 결과 시각화 메인 로직. """
    # 하이퍼파라미터 설정
    Nin = 784
    Nh = 100 # 단일 은닉층 뉴런 수 (ANN_seq_class는 현재 단일 값만 처리)
            # 만약 여러 은닉층을 원하면 ANN_seq_class 수정 또는 다른 모델 사용 필요
    number_of_class = 10
    Nout = number_of_class

    # 사용할 모델 선택 (예: ANN_seq_class)
    # 여러 은닉층을 사용하려면 Nh를 리스트로 전달하고 클래스 수정 필요
    # model = ANN_seq_class(Nin, Nh, Nout) # 단일 은닉층 모델
    model = ANN_model_fun(Nin, Nh, Nout) # 예시로 함수형 API 사용

    print("Model Summary:")
    model.summary()
    print("-" * 30)

    print("Loading and preprocessing data...")
    (X_train, Y_train), (X_test, Y_test) = Data_func()
    print("Data loaded and preprocessed.")
    print("-" * 30)

    print("Starting model training...")
    history = model.fit(X_train, Y_train, epochs=5, batch_size=100, validation_split=0.2, verbose=1)
    print("Model training finished.")
    print("-" * 30)

    print("Evaluating model performance...")
    performance_test = model.evaluate(X_test, Y_test, batch_size=100, verbose=0)
    print(f'Test Loss: {performance_test[0]:.4f}, Test Accuracy: {performance_test[1]:.4f}')
    print("-" * 30)

    print("Plotting training history...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plot_loss(history, title='Model Loss during Training')

    plt.subplot(1, 2, 2)
    plot_acc(history, title='Model Accuracy during Training')

    plt.tight_layout()
    plt.show()
    print("Plots displayed.")

# 7. 스크립트 실행 지점
if __name__ == '__main__':
    main()
