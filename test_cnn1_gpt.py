# -*- coding: utf-8 -*-

# 1. 필수 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist

# 2. CNN 모델 정의 (클래스 기반)
class CNN(models.Sequential):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        self.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.add(layers.Dropout(0.25))
        self.add(layers.Flatten())
        self.add(layers.Dense(128, activation='relu'))
        self.add(layers.Dropout(0.5))
        self.add(layers.Dense(num_classes, activation='softmax'))

        self.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# 3. 데이터 처리 클래스
class DATA():
    def __init__(self):
        num_classes = 10
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        img_rows, img_cols = X_train.shape[1:]

        # 채널 포맷 설정
        if K.image_data_format() == 'channels_first':
            X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
            X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
            X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        # 데이터 정규화 및 형 변환
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0

        # 레이블 원-핫 인코딩
        Y_train = to_categorical(y_train, num_classes)
        Y_test = to_categorical(y_test, num_classes)

        # 멤버 변수로 저장
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.X_train, self.Y_train = X_train, Y_train
        self.X_test, self.Y_test = X_test, Y_test

# 4. 학습 결과 시각화 함수
def plot_history(history):
    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Val loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train acc')
    plt.plot(history.history['val_accuracy'], label='Val acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 5. 메인 함수
def main():
    batch_size = 128
    epochs = 10

    data = DATA()
    model = CNN(data.input_shape, data.num_classes)

    history = model.fit(
        data.X_train, data.Y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(data.X_test, data.Y_test)
    )

    plot_history(history)

    # 모델 평가
    score = model.evaluate(data.X_test, data.Y_test, verbose=0)
    print(f"Test loss: {score[0]:.4f}")
    print(f"Test accuracy: {score[1]*100:.2f}%")

# 6. 실행
if __name__ == '__main__':
    main()