import pandas as pd
import numpy as np
import joblib  # 用于保存/加载pkl文件
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 1. 导入你已有的数据处理模块（data.py），获取干净数据
from data import process_student_data  # 调用data.py中的数据处理函数

# 2. 加载处理后的学生数据
print("正在从data.py加载处理后的学生数据...")
processed_data = process_student_data()  # 依赖你data.py中的process_student_data函数

# 3. 准备模型输入特征与目标变量
# 定义特征（影响期末成绩的因素）和目标（期末成绩）
feature_cols = ["study_hours", "attendance", "midterm_score", "homework_rate"]  # 需与data.py输出的字段一致
target_col = "final_score"

# 验证字段是否存在（避免数据处理后字段缺失）
missing_cols = [col for col in feature_cols + [target_col] if col not in processed_data.columns]
if missing_cols:
    raise ValueError(f"数据中缺失必要字段：{missing_cols}，请检查data.py的数据处理逻辑")

X = processed_data[feature_cols].values  # 特征矩阵
y = processed_data[target_col].values    # 目标变量（期末成绩）

# 4. 特征标准化（消除量纲影响，提升模型精度）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 标准化后的特征

# 5. 划分训练集与测试集（8:2，保证模型泛化能力）
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42  # 固定random_state确保结果可复现
)

# 6. 训练线性回归模型
print("正在训练期末成绩预测模型...")
model = LinearRegression()
model.fit(X_train, y_train)  # 用训练集训练模型

# 7. 模型评估（验证模型效果，避免无效模型）
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)  # 决定系数（越接近1越好）
test_mse = mean_squared_error(y_test, y_test_pred)  # 均方误差（越小越好）

print(f"\n模型评估结果：")
print(f"测试集R²（决定系数）：{test_r2:.3f}")
print(f"测试集MSE（均方误差）：{test_mse:.3f}")

# 仅当模型效果合格时才保存（R²≥0.4，可根据需求调整阈值）
if test_r2 < 0.4:
    print("警告：模型预测效果较差（R²<0.4），建议优化数据或特征后再保存！")
else:
    # 8. 保存模型和标准化器为pkl文件
    # 模型保存路径（与data.py、当前脚本同目录）
    model_save_path = "score_prediction_model.pkl"
    scaler_save_path = "scaler.pkl"

    # 保存模型
    joblib.dump(model, model_save_path)
    print(f"\n模型已保存为：{model_save_path}")

    # 保存特征标准化器（预测时需用同一标准化器处理输入特征）
    joblib.dump(scaler, scaler_save_path)
    print(f"特征标准化器已保存为：{scaler_save_path}")

    # 验证加载（确保保存成功）
    loaded_model = joblib.load(model_save_path)
    loaded_scaler = joblib.load(scaler_save_path)
    print(f"\n验证：pkl文件加载成功！")
    print(f"加载后的模型测试预测值：{loaded_model.predict(X_test[:1])[0]:.2f}（原始测试值：{y_test[0]:.2f}）")
