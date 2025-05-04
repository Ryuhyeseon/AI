# TensorFlow 및 필수 라이브러리 불러오기
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import models, layers, datasets, utils
import os
from sklearn.model_selection import train_test_split

# 1. Fashion MNIST 데이터셋 로드
# - 28x28 크기의 흑백 이미지로 구성된 10개의 패션 카테고리(예: 티셔츠, 신발 등)
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

# 2. 입력 데이터 전처리
# - CNN은 4차원 입력을 필요로 하므로 reshape 수행
# - 픽셀 값(0~255)을 0~1로 정규화하여 학습 안정성 향상
train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0

# 3. 훈련 데이터를 훈련/검증 세트로 나누기 (8:2 비율)
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

# 4. 테스트 데이터도 동일한 방식으로 전처리
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0

# 5. CNN 모델 구성 (Sequential 방식)
model = keras.Sequential()

# 첫 번째 합성곱 층: 32개의 3x3 필터, relu 활성화, same 패딩
model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=(28, 28, 1)))
# 최대 풀링: 특성 맵 크기를 절반으로 줄임
model.add(layers.MaxPooling2D(2))

# 두 번째 합성곱 층: 64개의 3x3 필터
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(2))

# 2차원 feature map → 1차원 벡터로 변환
model.add(layers.Flatten())

# 완전 연결층 (은닉층): 뉴런 100개, relu 활성화
model.add(layers.Dense(units=100, activation='relu'))

# 과적합 방지를 위한 Dropout (40% 확률로 뉴런 끔)
model.add(layers.Dropout(0.4))

# 출력층: 10개의 클래스, softmax 사용
model.add(layers.Dense(units=10, activation='softmax'))

# 모델 요약 출력 (구조 확인)
model.summary()

# 6. 모델 컴파일
# - 손실 함수: sparse_categorical_crossentropy (레이블이 정수일 때)
# - 옵티마이저: Adam (학습률 자동 조정)
# - 평가 지표: 정확도
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 7. 콜백 함수 정의
# - ModelCheckpoint: 검증 성능이 가장 좋은 모델만 저장
# - EarlyStopping: 2 에폭 동안 개선이 없으면 조기 종료, 가장 성능 좋은 모델 복원
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)

# 8. 모델 학습
# - 훈련 데이터에 대해 fit 수행, 검증 세트 지정
# - 콜백 함수를 통해 과적합 방지 및 베스트 모델 저장
history = model.fit(train_scaled, train_target, epochs=5,
                    validation_data=(val_scaled, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])

# 9. 학습 결과 시각화
plt.figure(figsize=(12, 4))

# (a) 손실 추이
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('(a) 손실 추이')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# (b) 정확도 추이
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('(a) 정확도 추이')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
