import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# CSV読み込み（区切り自動）
df = pd.read_csv("wine.csv")

print(df.columns)  # 確認用

# 目的変数
y = df["quality"]

# 説明変数
X = df[["alcohol", "sulphates", "volatile acidity", "citric acid"]]

# 分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# モデル
model = LinearRegression()
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 評価
print("R2 Score:", r2_score(y_test, y_pred))

# 図
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Linear Regression Result")

plt.plot([3, 9], [3, 9], color="red")

# 保存（指定通り）
plt.savefig("linear_regression_result.png")

plt.show()
