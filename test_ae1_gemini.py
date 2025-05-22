import tensorflow as tf
from tensorflow import keras # TensorFlow의 고수준 API인 Keras를 임포트합니다.
from keras import layers, models, datasets # Keras에서 모델 구성 및 데이터셋 로드를 위한 모듈을 임포트합니다.
# from keras.datasets import mnist # MNIST 데이터셋을 직접 임포트할 수도 있습니다. (위 datasets에 포함되어 있음)
import matplotlib.pyplot as plt # 데이터 시각화를 위한 라이브러리입니다.
import os # 운영 체제와 상호 작용하기 위한 모듈입니다. (KMP_DUPLICATE_LIB_OK 설정에 사용)

# Intel MKL 라이브러리 중복 로드 오류 해결 (주로 Windows 환경에서 발생)
# 여러 버전의 MKL(Math Kernel Library)이 로드될 때 발생하는 충돌을 방지하기 위한 설정입니다.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 오토인코더(Autoencoder, AE) 모델 정의
# models.Model을 상속받아 사용자 정의 모델을 만듭니다.
class AE(models.Model):
    def __init__(self, x_nodes, z_dim):
        # x_nodes: 입력 및 출력 레이어의 노드 수 (원본 데이터의 차원, 예: 28*28=784 for MNIST)
        # z_dim: 잠재 공간(은닉층)의 차원 (압축된 데이터의 차원)

        # 입력 데이터의 형태를 정의합니다. (x_nodes,)는 1차원 벡터를 의미합니다.
        x_shape = (x_nodes,)

        # 함수형 API를 사용하여 모델의 레이어를 정의합니다.
        # 입력 레이어: 입력 데이터의 형태(x_shape)를 받습니다.
        x_input = layers.Input(shape=x_shape, name='encoder_input') # 명확성을 위해 변수명 변경 및 이름 지정
        # 은닉 레이어 (인코더의 출력, 잠재 변수 z):
        # z_dim개의 노드를 가지며, 활성화 함수로 'relu'를 사용합니다.
        # 입력으로 x_input을 받습니다.
        z_latent = layers.Dense(z_dim, activation='relu', name='latent_space')(x_input) # 명확성을 위해 변수명 변경 및 이름 지정
        # 출력 레이어 (디코더의 출력, 복원된 데이터 y):
        # x_nodes개의 노드를 가지며, 활성화 함수로 'sigmoid'를 사용합니다.
        # (MNIST 픽셀 값이 0~1 사이이므로 sigmoid가 적합)
        # 입력으로 z_latent를 받습니다.
        y_output = layers.Dense(x_nodes, activation='sigmoid', name='decoder_output')(z_latent) # 명확성을 위해 변수명 변경 및 이름 지정

        # models.Model의 생성자를 호출하여 모델을 초기화합니다.
        # 입력으로 인코더의 입력(x_input)과 디코더의 출력(y_output)을 전달합니다.
        super().__init__(inputs=x_input, outputs=y_output) # inputs와 outputs 인자명 명시

        # 모델을 컴파일합니다.
        # optimizer: 최적화 알고리즘 ('adadelta' 사용)
        # loss: 손실 함수 ('binary_crossentropy'는 픽셀별 복원 오차 계산에 적합)
        # metrics: 평가 지표 (여기서는 정확도 'accuracy'를 사용하지만, AE에서는 주로 손실 값을 봄)
        self.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy']) # metrics 추가

        # 모델의 주요 부분을 인스턴스 변수로 저장하여 나중에 쉽게 접근할 수 있도록 합니다.
        self.x_input = x_input # 인코더의 입력 레이어
        self.z_latent = z_latent # 잠재 공간 레이어 (인코더의 출력)
        self.z_dim = z_dim # 잠재 공간의 차원

    # 인코더 부분을 별도의 모델로 반환하는 메서드
    def Encoder(self):
        # 입력으로 원본 데이터(self.x_input)를 받고, 출력으로 잠재 변수(self.z_latent)를 반환하는 모델을 생성합니다.
        return models.Model(inputs=self.x_input, outputs=self.z_latent) # inputs와 outputs 인자명 명시

    # 디코더 부분을 별도의 모델로 반환하는 메서드
    def Decoder(self):
        # 잠재 공간의 형태를 정의합니다. (z_dim,)는 1차원 벡터입니다.
        z_input_shape = (self.z_dim,) # 튜플로 만들기 위해 쉼표(,) 추가
        # 디코더의 입력 레이어를 정의합니다. (잠재 변수 z를 입력으로 받음)
        z_input_for_decoder = layers.Input(shape=z_input_shape, name='decoder_input_z')
        # AE 모델의 마지막 레이어(출력 레이어)를 가져옵니다.
        # self.layers[-1]은 AE 모델의 y_output에 해당하는 Dense 레이어입니다.
        output_layer_of_ae = self.layers[-1]
        # 디코더의 입력(z_input_for_decoder)을 AE의 출력 레이어에 연결하여 디코더의 출력을 얻습니다.
        y_decoded = output_layer_of_ae(z_input_for_decoder)
        # 입력으로 잠재 변수(z_input_for_decoder)를 받고, 출력으로 복원된 데이터(y_decoded)를 반환하는 모델을 생성합니다.
        return models.Model(inputs=z_input_for_decoder, outputs=y_decoded) # inputs와 outputs 인자명 명시

# --- 데이터 준비 ---
# MNIST 데이터셋을 로드합니다. x_train, x_test에는 이미지 데이터가, _에는 레이블 데이터가 들어갑니다.
# 오토인코더는 비지도 학습의 일종으로 레이블(y_train, y_test)을 사용하지 않으므로 _로 받습니다.
(x_train_orig, _), (x_test_orig, _) = datasets.mnist.load_data() # 원본 데이터 변수명 변경

# 데이터 정규화: 이미지의 픽셀 값을 0~1 사이로 조정합니다.
# astype('float32')는 나눗셈 연산을 위해 데이터 타입을 실수형으로 변경합니다.
x_train_normalized = x_train_orig.astype('float32') / 255.0 # .0 추가하여 float 나눗셈 명시
x_test_normalized = x_test_orig.astype('float32') / 255.0

# 데이터 형태 변경: 2차원 이미지 데이터(28x28)를 1차원 벡터(784)로 변환합니다.
# len(x_train_normalized)는 샘플의 수, -1은 나머지 차원을 자동으로 계산하라는 의미입니다.
x_train = x_train_normalized.reshape((len(x_train_normalized), -1))
x_test = x_test_normalized.reshape((len(x_test_normalized), -1))


# --- 시각화 함수 정의 ---
# kerasspp.skeras 라이브러리가 설치되어 있거나, plot_acc, plot_loss 함수가 현재 스코프에 정의되어 있어야 합니다.
# 만약 해당 라이브러리가 없다면, 직접 matplotlib을 사용하여 구현해야 합니다.
# 여기서는 해당 함수가 있다고 가정합니다.
try:
    from kerasspp.skeras import plot_acc, plot_loss
except ImportError:
    print("Warning: kerasspp.skeras library not found. plot_acc and plot_loss will not be available.")
    # 대체 함수 (간단한 예시)
    def plot_acc(history, **kwargs):
        if 'accuracy' in history.history:
            plt.plot(history.history['accuracy'], label='train_accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='val_accuracy')
        if 'accuracy' in history.history or 'val_accuracy' in history.history:
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
        else:
            print("Accuracy data not found in history.")

    def plot_loss(history, **kwargs):
        if 'loss' in history.history:
            plt.plot(history.history['loss'], label='train_loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='val_loss')
        if 'loss' in history.history or 'val_loss' in history.history:
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
        else:
            print("Loss data not found in history.")


# 오토인코더의 원본, 인코딩된 표현, 복원된 이미지를 시각화하는 함수
def show_ae_results(autoencoder_model, data_to_show): # 함수명 및 인자명 변경
    encoder_model = autoencoder_model.Encoder()
    decoder_model = autoencoder_model.Decoder()

    # 테스트 데이터로 인코딩 및 디코딩을 수행합니다.
    encoded_images = encoder_model.predict(data_to_show)
    decoded_images = decoder_model.predict(encoded_images)

    n = 10  # 보여줄 이미지의 개수
    plt.figure(figsize=(20, 6)) # Figure 크기 설정

    for i in range(n):
        # 원본 이미지 표시
        ax = plt.subplot(3, n, i + 1) # 3행 n열 중 i+1번째 위치
        plt.imshow(data_to_show[i].reshape(28, 28)) # 1D 벡터를 28x28 이미지로 변환하여 표시
        plt.gray() # 흑백으로 표시
        ax.get_xaxis().set_visible(False) # x축 눈금 숨기기
        ax.get_yaxis().set_visible(False) # y축 눈금 숨기기
        if i == 0:
            ax.set_title("Original", loc='left', fontsize=10, pad=10)


        # 인코딩된 이미지(잠재 변수) 표시
        # z_dim이 36이므로 6x6 이미지로 시각화합니다. 다른 z_dim 값이라면 이 부분을 수정해야 합니다.
        ax = plt.subplot(3, n, i + 1 + n) # 3행 n열 중 i+1+n번째 위치
        if autoencoder_model.z_dim == 36: # z_dim에 따라 reshape 형태 변경
            plt.imshow(encoded_images[i].reshape(6, 6))
        else:
            # 일반적인 경우 1차원 벡터를 그대로 막대 그래프 등으로 시각화하거나,
            # z_dim에 맞는 적절한 2D 형태로 reshape해야 합니다.
            # 여기서는 간단히 1D 벡터의 첫 부분만 보여주는 예시 (실제 시각화는 개선 필요)
            plt.imshow(encoded_images[i].reshape(-1, 1)[:min(28, autoencoder_model.z_dim)], aspect='auto', cmap='viridis')


        plt.gray() # 인코딩된 표현도 흑백으로 시도 (만약 reshape(6,6)등으로 2D 표현 시)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0:
            ax.set_title("Encoded", loc='left', fontsize=10, pad=10)


        # 복원된 이미지 표시
        ax = plt.subplot(3, n, i + 1 + n + n) # 3행 n열 중 i+1+n+n번째 위치
        plt.imshow(decoded_images[i].reshape(28, 28)) # 1D 벡터를 28x28 이미지로 변환하여 표시
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0:
            ax.set_title("Reconstructed", loc='left', fontsize=10, pad=10)

    plt.tight_layout() # 서브플롯 간 간격 자동 조절
    plt.show()


# 메인 실행 함수
def main():
    # 오토인코더의 파라미터 설정
    input_nodes = 784  # MNIST 이미지의 픽셀 수 (28*28)
    latent_dim = 36   # 잠재 공간의 차원 (예: 6*6 이미지로 표현 가능)

    # AE 모델 생성
    autoencoder = AE(input_nodes, latent_dim) # 변수명 일관성 있게 수정

    # 모델 훈련
    # 입력과 타겟으로 모두 x_train을 사용하여 원본 이미지를 복원하도록 학습합니다.
    history = autoencoder.fit(x_train, x_train, # 모델 변수명 수정
                              epochs=10,        # 전체 데이터셋에 대한 반복 학습 횟수
                              batch_size=256,   # 한 번의 가중치 업데이트에 사용될 샘플 수
                              shuffle=True,     # 각 에포크 시작 전에 훈련 데이터를 섞을지 여부
                              validation_data=(x_test, x_test)) # 검증 데이터로 x_test 사용

    # 학습 과정 시각화 (정확도 및 손실)
    plot_acc(history, version=2) # kerasspp.skeras 함수가 있다면 version 인자 사용 가능
    plt.show()
    plot_loss(history, version=2)
    plt.show()

    # 학습된 오토인코더로 결과 시각화 (x_test 데이터 중 처음 10개 사용)
    show_ae_results(autoencoder, x_test[:10]) # 함수명 및 모델 변수명 수정, 데이터 일부만 전달


# 이 스크립트가 직접 실행될 때 main() 함수를 호출합니다.
if __name__ == '__main__':
    main()