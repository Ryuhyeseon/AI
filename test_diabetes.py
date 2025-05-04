# 당뇨병 데이터셋 로드 (scikit-learn에 내장된 예제 데이터)
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

# 1. 당뇨병 데이터셋 불러오기
diabetes = load_diabetes()

# 2. 데이터의 크기 출력
# diabetes.data는 입력 데이터 (442명 × 10개의 특성)
# diabetes.target은 출력 데이터 (당뇨병 진행 정도 점수), 442개
print(diabetes.data.shape, diabetes.target.shape)  # 출력: (442, 10), (442,)

# 3. 데이터 일부 확인 (앞에서 3개 행만 출력)
# 각각의 행은 한 명의 환자, 열은 10가지 특성(예: 나이, BMI, 혈압 등)
print(diabetes.data[0:3])      # 첫 3명 환자의 특성 데이터
print(diabetes.target[:3])     # 첫 3명 환자의 당뇨병 진행 점수

# 4. 특성 중 BMI(BODY MASS INDEX, 체질량지수)를 x축으로 하여 산점도 그리기
# diabetes.data[:, 2]는 3번째 특성(BMI)만 추출한 것
plt.scatter(diabetes.data[:, 2], diabetes.target)

# x축 라벨: BMI
plt.xlabel('Body Mass Index (BMI)')

# y축 라벨: 당뇨병 진행 점수
plt.ylabel('Diabetes Progression')

# 산점도 출력
plt.show()

# 5. BMI 데이터 배열 출력 (확인용)
print(diabetes.data[:, 2])  # 전체 환자의 BMI 값(정규화된 값)
