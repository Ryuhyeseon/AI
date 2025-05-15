# 0. 기본 설정 및 라이브러리 임포트
import os
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow 및 Keras 관련 모듈 임포트
from tensorflow import keras
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# --- 1. 데이터 로드 및 기본 탐색 ---
print("--- 1. Data Loading and Basic Exploration ---")
(train_input_full, train_target_full), (test_input, test_target) = datasets.imdb.load_data(num_words=500)
print(f"Initial full training data shape: {train_input_full.shape}, Initial full training target shape: {train_target_full.shape}")
print(f"Initial test data shape: {test_input.shape}, Initial test target shape: {test_target.shape}")

# --- 2. 데이터 탐색: 리뷰 길이 분포 확인 ---
print("\n--- 2. Data Exploration: Checking Review Length Distribution ---")
lengths = np.array([len(x) for x in train_input_full])
print(f"Review length - Mean: {np.mean(lengths):.2f}, Median: {np.median(lengths)}")

# --- 3. 훈련 데이터와 검증 데이터 분리 ---
print("\n--- 3. Splitting Training and Validation Data ---")
train_input, val_input, train_target, val_target = train_test_split(
    train_input_full, train_target_full, test_size=0.2, random_state=42
)
print(f"Training input data shape after split: {train_input.shape}")
print(f"Validation input data shape after split: {val_input.shape}")

# --- 4. 데이터 전처리: 패딩 및 원-핫 인코딩 ---
print("\n--- 4. Data Preprocessing: Padding and One-Hot Encoding ---")
maxlen = 100
vocab_size = 500 # imdb.load_data에서 설정한 num_words와 동일

# 가. 시퀀스 패딩 (Padding) - model2에서 사용
train_seq = pad_sequences(train_input, maxlen=maxlen)
val_seq = pad_sequences(val_input, maxlen=maxlen)
test_seq = pad_sequences(test_input, maxlen=maxlen) # 테스트 세트도 동일하게 전처리
print(f"Training sequence shape after padding (for model2): {train_seq.shape}")

# 나. 원-핫 인코딩 (One-Hot Encoding) - model1에서 사용
train_oh = to_categorical(train_seq, num_classes=vocab_size)
val_oh = to_categorical(val_seq, num_classes=vocab_size)
test_oh = to_categorical(test_seq, num_classes=vocab_size) # 테스트 세트도 동일하게 전처리
print(f"Training data shape after one-hot encoding (for model1): {train_oh.shape}")


# --- 5. 모델 구성 ---
print("\n--- 5. Building Models ---")

# 모델 1: SimpleRNN (원-핫 인코딩 입력)
print("--- Model 1: SimpleRNN with One-Hot Encoded Input ---")
model1 = models.Sequential()
model1.add(layers.SimpleRNN(units=8, input_shape=(maxlen, vocab_size))) # 입력 형태: (시퀀스 길이, 어휘 크기)
model1.add(layers.Dense(units=1, activation='sigmoid'))
model1.summary()

# 모델 2: Embedding Layer + SimpleRNN
print("\n--- Model 2: SimpleRNN with Embedding Layer ---")
model2 = models.Sequential()
# Embedding 레이어:
# input_dim: 어휘 사전의 크기 (500)
# output_dim: 임베딩 벡터의 차원 (여기서는 16으로 설정)
# input_length: 입력 시퀀스의 길이 (maxlen, 100)
model2.add(layers.Embedding(input_dim=vocab_size, output_dim=16, input_length=maxlen))
model2.add(layers.SimpleRNN(units=8)) # Embedding 레이어 뒤에는 input_shape을 명시할 필요 없음
model2.add(layers.Dense(units=1, activation='sigmoid'))
model2.summary()


# --- 6. 모델 컴파일 ---
print("\n--- 6. Compiling Models ---")
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4) # 0.0001

# model1 컴파일
model1.compile(optimizer=rmsprop,
               loss='binary_crossentropy',
               metrics=['accuracy'])

# model2 컴파일 (optimizer를 새로 생성하거나, rmsprop 변수를 공유해도 됩니다)
# 이미지에서는 rmsprop 변수를 공유하고 있으므로 그대로 사용합니다.
model2.compile(optimizer=rmsprop, #동일한 rmsprop 옵티마이저 사용
               loss='binary_crossentropy',
               metrics=['accuracy'])

# --- 7. 모델 훈련 (model2를 훈련하고 시각화) ---
print("\n--- 7. Training Model 2 (with Embedding Layer) ---")
# 콜백 함수 정의
# ModelCheckpoint: 현재 설정으로는 model1과 model2 훈련 시 동일한 파일을 덮어쓰게 됩니다.
# 각 모델의 최적 가중치를 별도로 저장하려면 파일명을 다르게 하거나,
# 각 모델 훈련 시점에 ModelCheckpoint 객체를 새로 생성해야 합니다.
# 여기서는 이미지에 나온 대로 하나의 checkpoint_cb를 사용합니다.
checkpoint_cb = keras.callbacks.ModelCheckpoint('best_rnn_model.h5', # 파일명 변경 (예시)
                                                save_best_only=True,
                                                monitor='val_loss') # val_loss를 기준으로 저장
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True,
                                                  monitor='val_loss') # val_loss를 기준으로 조기 종료

# model2 훈련 시작
# 입력 데이터로 패딩된 정수 시퀀스(train_seq, val_seq)를 사용합니다.
history = model2.fit(train_seq, train_target, # model2는 train_seq 사용
                     epochs=10, # 이미지에서는 epochs=10
                     batch_size=64,
                     validation_data=(val_seq, val_target), # model2는 val_seq 사용
                     callbacks=[checkpoint_cb, early_stopping_cb])

# (선택 사항) model1을 훈련하고 싶다면 아래 주석을 해제하고 실행할 수 있습니다.
# print("\n--- (Optional) Training Model 1 (with One-Hot Encoding) ---")
# checkpoint_cb_model1 = keras.callbacks.ModelCheckpoint('best_simplernn_onehot_model.h5', save_best_only=True, monitor='val_loss')
# early_stopping_cb_model1 = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss')
# history_model1 = model1.fit(train_oh, train_target,
#                             epochs=10, # 에포크는 동일하게 또는 다르게 설정 가능
#                             batch_size=64,
#                             validation_data=(val_oh, val_target),
#                             callbacks=[checkpoint_cb_model1, early_stopping_cb_model1])


# --- 8. 훈련 과정 시각화 (model2의 결과) ---
print("\n--- 8. Visualizing Training Process for Model 2 (English Labels) ---")

# 손실 그래프 (model2)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss') # model2의 훈련 손실
plt.plot(history.history['val_loss'], label='Validation Loss') # model2의 검증 손실
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model 2: Training and Validation Loss (with Embedding)')
plt.grid(True)
plt.show()

# 정확도 그래프 (model2)
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy') # model2의 훈련 정확도
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') # model2의 검증 정확도
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model 2: Training and Validation Accuracy (with Embedding)')
plt.grid(True)
plt.show()

# (선택 사항) model1의 훈련 결과를 시각화하고 싶다면 아래 주석을 해제하고,
# history_model1 변수를 사용해야 합니다.
# print("\n--- (Optional) Visualizing Training Process for Model 1 ---")
# if 'history_model1' in locals(): # history_model1이 정의되었는지 확인
#     # 손실 그래프 (model1)
#     plt.figure(figsize=(10, 6))
#     plt.plot(history_model1.history['loss'], label='Train Loss (Model 1)')
#     plt.plot(history_model1.history['val_loss'], label='Validation Loss (Model 1)')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title('Model 1: Training and Validation Loss (One-Hot)')
#     plt.grid(True)
#     plt.show()

#     # 정확도 그래프 (model1)
#     plt.figure(figsize=(10, 6))
#     plt.plot(history_model1.history['accuracy'], label='Train Accuracy (Model 1)')
#     plt.plot(history_model1.history['val_accuracy'], label='Validation Accuracy (Model 1)')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.title('Model 1: Training and Validation Accuracy (One-Hot)')
#     plt.grid(True)
#     plt.show()

print("\n--- Script execution completed ---")

# 모델 평가 (테스트 세트 사용) - 선택 사항
# 저장된 최적 모델 로드 (model2 기준)
# best_model_m2 = keras.models.load_model('best_rnn_model.h5')
# test_loss_m2, test_acc_m2 = best_model_m2.evaluate(test_seq, test_target, verbose=0) # model2는 test_seq 사용
# print(f"\nModel 2 - Test set loss: {test_loss_m2:.4f}")
# print(f"Model 2 - Test set accuracy: {test_acc_m2:.4f}")

# (선택 사항) model1의 테스트 평가
# if os.path.exists('best_simplernn_onehot_model.h5'):
#     best_model_m1 = keras.models.load_model('best_simplernn_onehot_model.h5')
#     test_loss_m1, test_acc_m1 = best_model_m1.evaluate(test_oh, test_target, verbose=0) # model1은 test_oh 사용
#     print(f"\nModel 1 - Test set loss: {test_loss_m1:.4f}")
#     print(f"Model 1 - Test set accuracy: {test_acc_m1:.4f}")