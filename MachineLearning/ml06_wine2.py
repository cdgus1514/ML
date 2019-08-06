import matplotlib.pyplot as plt
import pandas as pd


# 1. 데이터 로드
wine = pd.read_csv("/content/winequality-white.csv", sep=";", encoding="utf-8")


# 품질 데이터로 나누고 count
count_data = wine.groupby("quality")["quality"].count()
print(count_data)


# 그래프
count_data.plot()
plt.savefig("wine-count-plt.png")
plt.show()