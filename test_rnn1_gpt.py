import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence
import matplotlib.pyplot as plt

# --- 1. 데이터 전처리 클래스 ---
class Data:
    def __init__(self, max_features=20000, maxlen=80):
        """
        IMDB 데이터셋을 로딩하고, 각 리뷰를 동일한 길이로 패딩 처리합니다.

        Args:
            max_features (int): 사용할 최대 단어 수 (빈도 기준).
            maxlen (int): 시퀀스 길이 제한 (긴 문장은 자르고, 짧은 문장은 패딩).
        """
        print(f"IMDB 데이터셋 로딩 중... max_features={max_features}, maxlen={maxlen}")
        (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)

        print(f"원본 시퀀스 길이 (예시): {len(x_train[0])}")
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

        print(f"패딩 후 x_train.shape: {x_train.shape}, x_test.shape: {x_test.shape}")

        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.max_features = max_features
        self.maxlen = maxlen


# --- 2. LSTM을 사용한 RNN 모델 클래스 ---
class RNN_LSTM(keras.Model):  # 사용자 정의 모델 구축을 위해 Keras Model 클래스 상속
    def __init__(self, max_features, maxlen):
        """
        LSTM 레이어를 사용하여 RNN 모델 아키텍처를 정의합니다.

        Args:
            max_features (int): 어휘 크기 (Embedding 레이어의 input_dim).
            maxlen (int): 입력 시퀀스의 길이.
        """
        super().__init__()

        # Embedding: 정수 시퀀스를 임베딩 벡터 시퀀스로 변환
        self.embedding = layers.Embedding(input_dim=max_features, output_dim=128, input_length=maxlen)

        # LSTM: 순환 신경망, 시퀀스의 순서를 반영한 정보 처리
        self.lstm = layers.LSTM(128)  # 출력 차원: 128

        # Dropout: 과적합 방지
        self.dropout = layers.Dropout(0.2)

        # 출력: 이진 분류 -> sigmoid 활성화
        self.dense = layers.Dense(1, activation='sigmoid')

    def call(self, x):
        """
        순전파(Forward) 과정 정의. 입력 -> Embedding -> LSTM -> Dropout -> Dense
        """
        x = self.embedding(x)   # (batch, maxlen) -> (batch, maxlen, 128)
        x = self.lstm(x)        # (batch, 128)
        x = self.dropout(x)     # (batch, 128)
        return self.dense(x)    # (batch, 1)


# --- 3. 모델 학습 및 평가 루틴 ---
def train_and_evaluate():
    # 데이터셋 준비
    data = Data()

    # 모델 인스턴스 생성
    model = RNN_LSTM(data.max_features, data.maxlen)

    # 모델 빌드 후 summary 출력
    model.build(input_shape=(None, data.maxlen))
    model.summary()

    # 모델 컴파일
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # 모델 학습
    history = model.fit(
        data.x_train, data.y_train,
        batch_size=32,
        epochs=5,
        validation_split=0.2
    )

    # 테스트셋 평가
    loss, acc = model.evaluate(data.x_test, data.y_test)
    print(f"\n테스트 손실: {loss:.4f}, 테스트 정확도: {acc:.4f}")

    return history


# --- 4. 학습 결과 시각화 함수 ---
def plot_history(history):
    """
    훈련 과정에서 기록된 손실과 정확도를 그래프로 시각화합니다.
    """
    plt.figure(figsize=(12, 4))

    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("[Loss]")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title("[Accuracy]")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


# --- 5. 전체 실행 ---
if __name__ == '__main__':
    history = train_and_evaluate()
    plot_history(history)
