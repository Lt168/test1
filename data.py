import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ---------------------- 1. 配置参数（根据实际CSV字段调整） ----------------------
# 文件路径（需与CSV实际存放路径一致）
FILE_PATH = "D:/streamlit_env/student_data_adjusted_rounded.csv"

# 字段映射：CSV原字段名 → 代码内部标准化字段名（根据CSV实际字段修改）
FIELD_MAPPING = {
    "学号": "student_id",          # 学生唯一标识
    "专业": "major",               # 专业名称
    "性别": "gender",              # 性别
    "每周学习时长（小时）": "study_hours",  # 每周学习时长
    "上课出勤率": "attendance",    # 上课出勤率（%）
    "期中考试分数": "midterm_score",  # 期中考试分数
    "作业完成率": "homework_rate",  # 作业完成率（%）
    "期末考试分数": "final_score"   # 期末考试分数（预测目标）
}

# 核心字段列表（确保后续模块依赖字段不缺失）
REQUIRED_FIELDS = list(FIELD_MAPPING.values())


# ---------------------- 2. 数据加载（含路径容错，无侧边栏输出） ----------------------
def load_raw_data(file_path: str) -> pd.DataFrame:
    """加载CSV原始数据，路径错误时生成模拟数据"""
    try:
        # 读取CSV（支持中文编码）
        raw_df = pd.read_csv(file_path, encoding="utf-8")
        return raw_df
    except FileNotFoundError:
        # 生成100条模拟数据，结构与真实数据一致
        mock_data = {
            "学号": [f"2024{str(i).zfill(4)}" for i in range(1, 101)],
            "专业": np.random.choice(["信息系统", "计算机科学", "数据科学", "大数据管理", "人工智能"], 100),
            "性别": np.random.choice(["男", "女"], 100, p=[0.52, 0.48]),
            "每周学习时长（小时）": np.random.randint(5, 18, 100),
            "上课出勤率": np.random.uniform(75, 98, 100).round(1),
            "期中考试分数": np.random.randint(40, 90, 100),
            "作业完成率": np.random.uniform(60, 100, 100).round(1),
            "期末考试分数": np.random.randint(45, 95, 100)
        }
        return pd.DataFrame(mock_data)
    except Exception as e:
        raise  # 抛出未知错误，便于调试


# ---------------------- 3. 字段标准化（统一字段名，无侧边栏输出） ----------------------
def standardize_fields(raw_df: pd.DataFrame) -> pd.DataFrame:
    """将CSV原字段名标准化为统一字段名，删除无关字段"""
    # 清理原字段名（去除空格、特殊字符）
    raw_df.columns = [col.strip().replace("\u3000", "").replace(" ", "") for col in raw_df.columns]
    
    # 保留存在的核心字段，映射为标准化名称
    existing_mapping = {f: FIELD_MAPPING[f] for f in FIELD_MAPPING.keys() if f in raw_df.columns}
    standardized_df = raw_df[list(existing_mapping.keys())].rename(columns=existing_mapping)
    
    return standardized_df


# ---------------------- 4. 缺失值处理（保证数据完整性，无侧边栏输出） ----------------------
def handle_missing_values(standardized_df: pd.DataFrame) -> pd.DataFrame:
    """数值型字段用均值填充，分类型字段用众数填充"""
    clean_df = standardized_df.copy()
    
    # 数值型字段处理
    numeric_fields = ["study_hours", "attendance", "midterm_score", "homework_rate", "final_score"]
    for field in numeric_fields:
        if field in clean_df.columns and clean_df[field].isnull().sum() > 0:
            clean_df[field].fillna(clean_df[field].mean(), inplace=True)
    
    # 分类型字段处理
    categorical_fields = ["student_id", "major", "gender"]
    for field in categorical_fields:
        if field in clean_df.columns and clean_df[field].isnull().sum() > 0:
            clean_df[field].fillna(clean_df[field].mode()[0], inplace=True)
    
    return clean_df


# ---------------------- 5. 补充缺失核心字段（无侧边栏输出） ----------------------
def supplement_required_fields(clean_df: pd.DataFrame) -> pd.DataFrame:
    """补充缺失的核心字段，用合理模拟数据填充"""
    full_df = clean_df.copy()
    data_size = len(full_df)
    
    for field in REQUIRED_FIELDS:
        if field not in full_df.columns:
            if field == "student_id":
                full_df[field] = [f"2024{str(i).zfill(4)}" for i in range(1, data_size + 1)]
            elif field == "major":
                full_df[field] = np.random.choice(["信息系统", "计算机科学", "数据科学", "大数据管理", "人工智能"], data_size)
            elif field == "gender":
                full_df[field] = np.random.choice(["男", "女"], data_size, p=[0.52, 0.48])
            elif field == "study_hours":
                full_df[field] = np.random.randint(5, 18, data_size).astype(float)
            elif field in ["attendance", "homework_rate"]:
                full_df[field] = np.random.uniform(70, 100, data_size).round(1).astype(float)
            elif field in ["midterm_score", "final_score"]:
                full_df[field] = np.random.randint(30, 100, data_size).astype(float)
    
    return full_df


# ---------------------- 6. 数据类型转换（适配建模/可视化，无侧边栏输出） ----------------------
def convert_data_types(full_df: pd.DataFrame) -> pd.DataFrame:
    """统一数据类型：数值型转float，分类型转str"""
    final_df = full_df.copy()
    
    # 数值型字段转float
    numeric_fields = ["study_hours", "attendance", "midterm_score", "homework_rate", "final_score"]
    for field in numeric_fields:
        if field in final_df.columns:
            final_df[field] = pd.to_numeric(final_df[field], errors="coerce").astype(float)
    
    # 分类型字段转str
    categorical_fields = ["student_id", "major", "gender"]
    for field in categorical_fields:
        if field in final_df.columns:
            final_df[field] = final_df[field].astype(str)
    
    return final_df


# ---------------------- 7. 数据处理主函数（外部模块调用入口） ----------------------
def process_student_data() -> pd.DataFrame:
    """完整数据处理流程：加载→标准化→缺失值处理→补充字段→类型转换"""
    raw_df = load_raw_data(FILE_PATH)
    standardized_df = standardize_fields(raw_df)
    clean_df = handle_missing_values(standardized_df)
    full_df = supplement_required_fields(clean_df)
    final_df = convert_data_types(full_df)
    
    return final_df


# ---------------------- 8. 测试代码（单独运行时验证，无侧边栏输出） ----------------------
if __name__ == "__main__":
    # 执行数据处理并打印结果（无Streamlit界面输出）
    processed_data = process_student_data()
    print("数据处理完成，前5条数据预览：")
    print(processed_data.head())
    print(f"\n数据规模：{processed_data.shape[0]}行 × {processed_data.shape[1]}列")
    print(f"核心字段：{list(processed_data.columns)}")
