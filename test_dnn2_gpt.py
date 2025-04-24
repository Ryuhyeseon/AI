# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


# 1. 데이터 로드 및 전처리
def load_cifar10_data():
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0

    Y_train = to_categorical(y_train, 10)
    Y_test = to_categorical(y_test, 10)

    return (X_train, Y_train), (X_test, Y_test)


# 2. DNN 모델 정의
class DNN(models.Sequential):
    def __init__(self, Nin: int, Nh_l: list[int], Pd_l: list[float], Nout: int):
        super().__init__(name='DNN_with_Dropout')
        self.add(layers.Dense(Nh_l[0], activation='relu', input_shape=(Nin,), name='Hidden-1'))
        self.add(layers.Dropout(Pd_l[0]))
        self.add(layers.Dense(Nh_l[1], activation='relu', name='Hidden-2'))
        self.add(layers.Dropout(Pd_l[1]))
        self.add(layers.Dense(Nout, activation='softmax'))

        self.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])


# 3. 학습 결과 시각화 함수
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


# 4. 메인 실행
def main():
    Nh_l = [100, 50]
    Pd_l = [0.2, 0.2]  # Dropout 비율 설정
    Nout = 10

    (X_train, Y_train), (X_test, Y_test) = load_cifar10_data()
    Nin = X_train.shape[1]

    model = DNN(Nin, Nh_l, Pd_l, Nout)
    history = model.fit(X_train, Y_train,
                        epochs=10,
                        batch_size=100,
                        validation_split=0.2,
                        verbose=2)

    test_loss, test_acc = model.evaluate(X_test, Y_test, batch_size=100)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    plot_training_history(history)


if __name__ == "__main__":
    main()