import numpy as np
import matplotlib.pyplot as plt

# 랜덤 시드 설정 (재현 가능성 위해)
np.random.seed(42)

# 데이터 생성
x = np.random.randn(1000)
y = np.random.randn(1000)

# 그림 크기 지정
plt.figure(figsize=(8, 6))

# 산점도 그리기
plt.scatter(x, y, alpha=0.6, c='royalblue', edgecolor='k', s=40, label='샘플 데이터')

# 축 라벨 및 제목 추가
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Random Scatter Plot Example')

# 그리드 추가
plt.grid(True, linestyle='--', alpha=0.5)

# 범례 추가
plt.legend()

# 출력
plt.show()
