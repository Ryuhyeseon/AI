# -*- coding: utf-8 -*- # 파일 인코딩을 UTF-8로 명시 (주석 등에 한글 사용 시 권장)

# 1. 필수 라이브러리 임포트
import sys                      # 시스템 관련 파라미터 및 함수 접근 (예: 최대 출력 크기 설정)
import numpy as np              # 수치 계산, 배열/행렬 연산 라이브러리 (별칭 np)
from tensorflow import keras    # TensorFlow 백엔드를 사용하는 Keras API
from keras import layers, models      # Keras 모델 구축을 위한 층(layers) 및 모델(models) 클래스/함수
from keras import datasets # Keras에 내장된 데이터셋 (예: MNIST)
from tensorflow.keras.utils import to_categorical # 레이블 원-핫 인코딩 함수 (표준 방식)
import matplotlib.pyplot as plt # 데이터 시각화 라이브러리 (별칭 plt)

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
    # 입력층 정의: 입력 형태(shape) 지정
    x = layers.Input(shape=(Nin,), name='input_layer')
    # 은닉층 정의: Dense(완전 연결) + ReLU 활성화 함수
    h = layers.Activation('relu', name='hidden_activation')(layers.Dense(Nh, name='hidden_dense')(x))
    # 출력층 정의: Dense(완전 연결) + Softmax 활성화 함수 (다중 클래스 분류용)
    y = layers.Activation('softmax', name='output_activation')(layers.Dense(Nout, name='output_dense')(h))

    # 모델 구성: 입력과 출력을 연결하여 모델 생성
    model = models.Model(inputs=x, outputs=y, name='Functional_ANN_Model') # inputs 인자 명시
    # 모델 컴파일: 손실 함수, 옵티마이저, 평가 지표 설정
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
    # Sequential 모델 객체 생성
    model = models.Sequential(name='Sequential_ANN_Function')
    # 은닉층 추가: 입력 형태(input_shape)는 첫 번째 레이어에만 지정
    model.add(layers.Dense(Nh, activation='relu', input_shape=(Nin,), name='hidden_dense_relu'))
    # 출력층 추가
    model.add(layers.Dense(Nout, activation='softmax', name='output_dense_softmax'))
    # 모델 컴파일
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 방식 3: 함수형 API (클래스 정의 - Model 서브클래싱)
class ANN_models_class(models.Model):
    """
    Model 클래스를 상속받아 함수형 API 스타일로 ANN 모델을 정의하는 클래스입니다.
    __init__에서 레이어를 정의하고, Model 생성자에 input/output 텐서를 전달합니다.

    Args:
        Nin (int): 입력층 뉴런 수
        Nh (int): 은닉층 뉴런 수
        Nout (int): 출력층 뉴런 수
    """
    def __init__(self, Nin, Nh, Nout):
        # Model 클래스의 생성자 호출 (이름 지정)
        super().__init__(name='Functional_ANN_Class')
        # 필요한 레이어들을 인스턴스 변수로 정의
        self.hidden_layer = layers.Dense(Nh, name='hidden_dense')
        self.output_layer = layers.Dense(Nout, name='output_dense')
        self.relu_activation = layers.Activation('relu', name='hidden_activation')
        self.softmax_activation = layers.Activation('softmax', name='output_activation')

        # 모델 구조 정의 (함수형 API 스타일) - 입력을 정의하고 연결
        x = layers.Input(shape=(Nin,), name='input_layer')
        h = self.relu_activation(self.hidden_layer(x))
        y = self.softmax_activation(self.output_layer(h))

        # Model 생성자 재호출 (입력과 출력 텐서 전달) - 이 부분은 Keras의 일반적인
        # 서브클래싱 방식과 약간 다릅니다. 보통 call 메서드에서 로직을 정의하지만,
        # 이 방식은 생성자 내에서 함수형 API처럼 구성 후 super().__init__을 다시 호출합니다.
        # 명확성을 위해 call 메서드를 사용하는 것이 더 일반적입니다.
        # 하지만 원본 구조를 유지하기 위해 이대로 둡니다.
        # (주의: 이 super().__init__ 호출은 모델을 실제로 '빌드'합니다.)
        super().__init__(inputs=x, outputs=y)

        # 모델 컴파일
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # (참고) 일반적인 Keras 서브클래싱 방식에서는 call 메서드를 정의합니다:
    # def call(self, inputs):
    #     h = self.relu_activation(self.hidden_layer(inputs))
    #     y = self.softmax_activation(self.output_layer(h))
    #     return y

# 방식 4: Sequential API (클래스 정의 - Sequential 서브클래싱)
class ANN_seq_class(models.Sequential):
    """
    Sequential 클래스를 상속받아 Sequential API 스타일로 ANN 모델을 정의하는 클래스입니다.
    생성자에서 add() 메소드를 사용하여 레이어를 추가합니다.

    Args:
        Nin (int): 입력층 뉴런 수 (input_shape으로 지정)
        Nh (int): 은닉층 뉴런 수
        Nout (int): 출력층 뉴런 수
    """
    def __init__(self, Nin, Nh, Nout):
        # Sequential 클래스의 생성자 호출
        super().__init__(name='Sequential_ANN_Class')
        # add() 메소드를 사용하여 레이어 순차적으로 추가
        self.add(layers.Dense(Nh, activation='relu', input_shape=(Nin,), name='hidden_dense_relu'))
        self.add(layers.Dense(Nout, activation='softmax', name='output_dense_softmax'))
        # 모델 컴파일
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 데이터 로드 및 전처리 함수
def Data_func():
    """
    Keras 내장 MNIST 데이터셋을 로드하고 신경망 학습에 맞게 전처리합니다.
    - 레이블 원-핫 인코딩
    - 이미지 데이터 1차원 변환
    - 이미지 픽셀 값 [0, 1] 정규화

    Returns:
        tuple: ((X_train, Y_train), (X_test, Y_test)) 형태의 전처리된 데이터 튜플
    """
    # MNIST 데이터 로드 (훈련 데이터 60000개, 테스트 데이터 10000개)
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    # 레이블(0-9 정수)을 원-핫 인코딩 벡터로 변환 (예: 5 -> [0,0,0,0,0,1,0,0,0,0])
    Y_train = to_categorical(y_train)
    Y_test = to_categorical(y_test)

    # (선택적) 변환된 레이블 확인
    print("Shape of Y_train:", Y_train.shape) # (60000, 10)

    # 이미지 데이터 형태 변환 및 정규화
    img_width, img_height = X_train.shape[1], X_train.shape[2]
    num_pixels = img_width * img_height # 28 * 28 = 784

    # 이미지를 2D(28x28)에서 1D(784) 벡터로 펼치기(reshape) 및 타입 변환/정규화
    X_train = X_train.reshape(-1, num_pixels).astype('float32') / 255.0
    X_test = X_test.reshape(-1, num_pixels).astype('float32') / 255.0

    # 전처리된 데이터를 튜플로 묶어 반환
    return (X_train, Y_train), (X_test, Y_test)

# 5. 학습 결과 시각화 함수

# 정확도 그래프 함수
def plot_acc(history, title=None):
    """
    모델 학습 history에서 훈련/검증 정확도를 추출하여 그래프로 시각화합니다.

    Args:
        history (keras.callbacks.History or dict): model.fit() 결과 또는 관련 딕셔너리
        title (str, optional): 그래프 제목
    """
    # history가 Keras History 객체인 경우 실제 데이터는 .history 속성에 있음
    if not isinstance(history, dict):
        history = history.history

    # 훈련 정확도 (accuracy) 플롯
    plt.plot(history['accuracy'])
    # 검증 정확도 (val_accuracy) 플롯
    plt.plot(history['val_accuracy'])
    # 그래프 제목 설정 (주어진 경우)
    if title is not None:
        plt.title(title)
    # Y축 레이블 설정
    plt.ylabel('Accuracy')
    # X축 레이블 설정
    plt.xlabel('Epochs')
    # 범례 추가
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right') # 위치 조정 가능

# 손실 그래프 함수
def plot_loss(history, title=None):
    """
    모델 학습 history에서 훈련/검증 손실을 추출하여 그래프로 시각화합니다.

    Args:
        history (keras.callbacks.History or dict): model.fit() 결과 또는 관련 딕셔너리
        title (str, optional): 그래프 제목
    """
    # history가 Keras History 객체인 경우 실제 데이터는 .history 속성에 있음
    if not isinstance(history, dict):
        history = history.history

    # 훈련 손실 (loss) 플롯
    plt.plot(history['loss'])
    # 검증 손실 (val_loss) 플롯
    plt.plot(history['val_loss'])
    # 그래프 제목 설정 (주어진 경우)
    if title is not None:
        plt.title(title)
    # Y축 레이블 설정
    plt.ylabel('Loss')
    # X축 레이블 설정
    plt.xlabel('Epochs')
    # 범례 추가
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right') # 위치 조정 가능

# 6. 메인 실행 함수
def main():
    """
    데이터 로드, 모델 생성/학습/평가, 결과 시각화를 수행하는 메인 로직.
    """
    # 신경망 하이퍼파라미터 설정
    Nin = 784           # 입력 특성 수 (MNIST 28x28 픽셀)
    Nh = 100            # 은닉층 뉴런 수
    number_of_class = 10 # 출력 클래스 수 (0-9 숫자)
    Nout = number_of_class # 출력층 뉴런 수

    # 사용할 모델 구현 방식 선택 (아래 중 하나 주석 해제)
    # model = ANN_model_fun(Nin, Nh, Nout)
    # model = ANN_seq_func(Nin, Nh, Nout)
    # model = ANN_models_class(Nin, Nh, Nout)
    model = ANN_seq_class(Nin, Nh, Nout) # 원본 코드에서 사용된 클래스 방식 유지

    # 모델 구조 요약 출력
    print("Model Summary:")
    model.summary()
    print("-" * 30)

    # 데이터 로드 및 전처리
    print("Loading and preprocessing data...")
    (X_train, Y_train), (X_test, Y_test) = Data_func()
    print("Data loaded and preprocessed.")
    print("-" * 30)

    # 모델 학습
    print("Starting model training...")
    history = model.fit(X_train, Y_train, epochs=5, batch_size=100, validation_split=0.2, verbose=1) # verbose=1로 진행 상황 표시
    print("Model training finished.")
    print("-" * 30)

    # 모델 평가 (테스트 데이터셋 사용)
    print("Evaluating model performance...")
    performance_test = model.evaluate(X_test, Y_test, batch_size=100, verbose=0) # 평가는 진행 상황 생략 가능
    print(f'Test Loss: {performance_test[0]:.4f}, Test Accuracy: {performance_test[1]:.4f}')
    print("-" * 30)

    # 학습 과정 시각화 (손실 및 정확도 그래프)
    print("Plotting training history...")
    plt.figure(figsize=(12, 5)) # 전체 그림 크기 설정

    plt.subplot(1, 2, 1) # 1행 2열 중 1번째 서브플롯
    plot_loss(history, title='Model Loss during Training')

    plt.subplot(1, 2, 2) # 1행 2열 중 2번째 서브플롯
    plot_acc(history, title='Model Accuracy during Training')

    plt.tight_layout() # 서브플롯 간 간격 자동 조절
    plt.show() # 그래프 창 표시
    print("Plots displayed.")

# 7. 스크립트 실행 지점
# 이 파일이 직접 실행될 경우 (__name__이 '__main__'일 경우) main() 함수 호출
if __name__ == '__main__':
    main()