import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ---------------------- 模拟数据处理模块（替代原有data.py） ----------------------
def process_student_data():
    """模拟学生数据生成（用于测试，实际使用时替换为真实数据加载逻辑）"""
    majors = ["大数据管理", "计算机科学", "人工智能", "软件工程", "信息管理"]
    genders = ["男", "女"]
    
    # 生成模拟数据
    np.random.seed(42)
    data = []
    for major in majors:
        # 大数据管理专业出勤率稍低，模拟颜色淡的场景
        att_base = 0.76 if major == "大数据管理" else np.random.uniform(0.78, 0.85)
        for gender in genders:
            n_students = np.random.randint(30, 50)
            for _ in range(n_students):
                data.append({
                    "major": major,
                    "gender": gender,
                    "midterm_score": np.random.uniform(40, 95),
                    "final_score": np.random.uniform(45, 98),
                    "study_hours": np.random.uniform(5, 25),
                    "attendance": np.random.uniform(att_base - 0.02, att_base + 0.02),
                    "homework_rate": np.random.uniform(0.7, 1.0)
                })
    return pd.DataFrame(data)

# ---------------------- 全局配置 ----------------------
st.set_page_config(page_title="学生成绩分析与预测系统", layout="wide")
# 加载数据
processed_data = process_student_data()
# 加载模型（确保pkl文件存在）
try:
    model = joblib.load("score_prediction_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    model = None

# ---------------------- 侧边栏导航 ----------------------
st.sidebar.title("导航菜单")
page = st.sidebar.radio("", ["项目介绍", "专业数据分析", "成绩预测"])

# ---------------------- 1. 项目介绍界面 ----------------------
if page == "项目介绍":
    st.title("学生成绩分析与预测系统")
    st.divider()

    # 项目概述+预览图
    col_overview, col_preview = st.columns([2, 1])
    with col_overview:
        st.subheader("📋 项目概述")
        st.write("本项目是一个基于streamlit的学生成绩分析平台，通过数据可视化和机器学习技术，帮助教育工作者和学生深入了解学业表现，并预测期末考试成绩。")

        st.subheader("✨ 主要特点")
        st.markdown("""
        - 📊 **数据可视化**：多维度展示学生学业数据
        - 🎓 **专业分析**：按专业分类的详细统计分析
        - 🤖 **智能预测**：基于机器学习模型的成绩预测
        - 💡 **学习建议**：根据预测结果提供个性化反馈
        """)
    with col_preview:
        st.subheader("专业数据分析")
        st.write("1.各专业男女性别比例")
        # 仅修改此处：替换为你的截图路径
        st.image("D:/streamlit_env/jietu.png", use_container_width=True)
        st.caption("学生数据分析示意图")

    st.divider()

    # 项目目标
    st.subheader("🎯 项目目标")
    col_target1, col_target2, col_target3 = st.columns(3)
    with col_target1:
        st.markdown("#### 目标一：分析影响因素")
        st.markdown("""
        - 识别关键学习指标
        - 探索成绩相关因素
        - 提供数据支持决策
        """)
    with col_target2:
        st.markdown("#### 目标二：可视化展示")
        st.markdown("""
        - 专业对比分析
        - 性别差异研究
        - 学习模式识别
        """)
    with col_target3:
        st.markdown("#### 目标三：成绩预测")
        st.markdown("""
        - 机器学习模型
        - 个性化预测
        - 及时干预预警
        """)

    st.divider()

    # 技术架构
    st.subheader("🔧 技术架构")
    col_tech1, col_tech2, col_tech3, col_tech4 = st.columns(4)
    with col_tech1:
        st.write("前端框架：Streamlit")
    with col_tech2:
        st.write("数据处理：Pandas、NumPy")
    with col_tech3:
        st.write("可视化：Plotly、Matplotlib")
    with col_tech4:
        st.write("机器学习：Scikit-learn")

# ---------------------- 2. 专业数据分析界面（最终优化版） ----------------------
elif page == "专业数据分析":
    st.title("专业数据分析")
    st.divider()

    # 1. 各专业男女性别比例
    st.subheader("1. 各专业男女性别比例")
    col_gender_chart, col_gender_table = st.columns([2, 1])
    with col_gender_chart:
        gender_ratio = processed_data.groupby(["major", "gender"]).size().unstack(fill_value=0)
        gender_ratio = gender_ratio.div(gender_ratio.sum(axis=1), axis=0).reset_index()
        fig_gender = px.bar(
            gender_ratio.melt(id_vars="major", value_vars=["男", "女"], var_name="性别", value_name="比例"),
            x="major", y="比例", color="性别", barmode="group",
            color_discrete_map={"男": "#1f77b4", "女": "#aec7e8"},
            labels={"比例": "比例", "major": "专业"},
            height=300
        )
        fig_gender.update_layout(
            xaxis_title="专业",  # 补充x轴标题
            legend_title="性别", 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_gender, use_container_width=True)
    with col_gender_table:
        gender_table = gender_ratio[["major", "男", "女"]].rename(columns={"男": "男性比例", "女": "女性比例"}).round(4)
        st.dataframe(gender_table.set_index("major"), use_container_width=True)

    st.divider()

    # 2. 各专业学习指标对比（核心修改：补充x轴标题+全中文标签）
    st.subheader("2. 各专业学习指标对比")
    col_score_chart, col_score_table = st.columns([2, 1])
    with col_score_chart:
        score_data = processed_data.groupby("major").agg({
            "midterm_score": "mean",
            "final_score": "mean",
            "study_hours": "mean"
        }).reset_index()
        # 多折线图（双轴）
        fig_score = go.Figure()
        fig_score.add_trace(go.Scatter(
            x=score_data["major"], 
            y=score_data["midterm_score"], 
            name="期中考试分数", 
            mode="lines+markers", 
            line=dict(color="#1f77b4")
        ))
        fig_score.add_trace(go.Scatter(
            x=score_data["major"], 
            y=score_data["final_score"], 
            name="期末考试分数", 
            mode="lines+markers", 
            line=dict(color="#ff7f0e")
        ))
        fig_score.add_trace(go.Scatter(
            x=score_data["major"], 
            y=score_data["study_hours"], 
            name="每周学习时长", 
            mode="lines+markers", 
            line=dict(color="#2ca02c"), 
            yaxis="y2"
        ))
        fig_score.update_layout(
            xaxis_title="专业",  # 补充major对应的标题
            yaxis=dict(title="分数"),
            yaxis2=dict(title="每周学习时长（小时）", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=300
        )
        st.plotly_chart(fig_score, use_container_width=True)
    with col_score_table:
        score_table = score_data.rename(columns={
            "midterm_score": "期中考试分数",
            "final_score": "期末考试分数",
            "study_hours": "每周学习时长（小时）"
        }).round(4)
        st.dataframe(score_table.set_index("major"), use_container_width=True)

    st.divider()

    # 3. 各专业出勤率分析（优化颜色）
    st.subheader("3. 各专业出勤率分析")
    col_att_chart, col_att_table = st.columns([2, 1])
    with col_att_chart:
        att_data = processed_data.groupby("major")["attendance"].mean().reset_index()
        
        # 优化1：固定颜色范围 + 更鲜明的绿色渐变
        fig_att = px.bar(
            att_data, x="major", y="attendance", color="attendance",
            color_continuous_scale=["#d9f0a3", "#78c679", "#238443"],  # 更鲜明的绿色渐变
            range_color=[att_data["attendance"].min() - 0.01, att_data["attendance"].max() + 0.01],  # 自适应颜色范围
            color_continuous_midpoint=att_data["attendance"].mean(),  # 以平均值为中点
            labels={"attendance": "平均上课出勤率", "major": "专业"},
            height=300
        )
        # 优化2：调整颜色条显示，增强可读性
        fig_att.update_layout(
            xaxis_title="专业",  # 补充x轴标题
            coloraxis_showscale=True, 
            coloraxis_colorbar=dict(
                title="平均出勤率",
                tickformat=".1%",  # 百分比显示
                len=0.8  # 缩短颜色条，更美观
            ),
            legend=None
        )
        # 优化3：给柱子添加数值标签
        fig_att.add_trace(go.Bar(
            x=att_data["major"], 
            y=att_data["attendance"],
            text=[f"{x:.1%}" for x in att_data["attendance"]],
            textposition="auto",
            showlegend=False,
            marker=dict(color="rgba(0,0,0,0)")  # 透明柱子，只显示文字
        ))
        st.plotly_chart(fig_att, use_container_width=True)
    with col_att_table:
        att_table = att_data.rename(columns={"attendance": "平均出勤率", "major": "专业"}).round(4)
        st.dataframe(att_table.set_index("专业"), use_container_width=True)

    st.divider()

    # 4. 大数据管理专业专项分析（全中文展示期末成绩）
    st.subheader("4. 大数据管理专业专项分析")
    target_major = "大数据管理" if "大数据管理" in processed_data["major"].unique() else processed_data["major"].unique()[0]
    df_target = processed_data[processed_data["major"] == target_major]
    
    # 核心指标卡片
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    with col_metric1:
        st.metric("平均出勤率", f"{df_target['attendance'].mean():.1%}")
    with col_metric2:
        st.metric("平均期末成绩", f"{df_target['final_score'].mean():.1f}分")  # 改为“期末成绩”
    with col_metric3:
        st.metric("通过率", f"{(df_target['final_score'] >= 60).mean():.1%}")
    with col_metric4:
        st.metric("平均学习时长", f"{df_target['study_hours'].mean():.1f}小时")

    # 成绩分布+学习时长分布（全中文标题/标签）
    col_dist1, col_dist2 = st.columns(2)
    with col_dist1:
        st.subheader(f"{target_major}专业期末成绩分布")  # 改为“期末成绩”
        fig_score_dist = px.histogram(
            df_target, 
            x="final_score", 
            nbins=20, 
            color_discrete_sequence=["#1f77b4"], 
            height=300,
            labels={"final_score": "期末成绩"}  # x轴标签改为中文
        )
        fig_score_dist.update_layout(xaxis_title="期末成绩", yaxis_title="人数")
        st.plotly_chart(fig_score_dist, use_container_width=True)
    with col_dist2:
        st.subheader(f"{target_major}专业学习时长分布")
        fig_study_box = px.box(
            df_target, 
            y="study_hours", 
            color_discrete_sequence=["#1f77b4"], 
            height=300,
            labels={"study_hours": "学习时长（小时）"}  # y轴标签改为中文
        )
        fig_study_box.update_layout(yaxis_title="学习时长（小时）")
        st.plotly_chart(fig_study_box, use_container_width=True)

# ---------------------- 3. 成绩预测界面 ----------------------
elif page == "成绩预测":
    st.title("期末成绩预测")
    st.write("请输入学生的学习信息，系统将预测其期末成绩并提供学习建议")
    st.divider()

    # 输入表单（左右分栏）
    with st.form("pred_form"):
        col_left, col_right = st.columns(2)
        with col_left:
            student_id = st.text_input("学号", "1231231")
            gender = st.selectbox("性别", processed_data["gender"].unique())
            major = st.selectbox("专业", processed_data["major"].unique())
            submit_btn = st.form_submit_button("预测期末成绩", type="primary")
        with col_right:
            study_hours = st.slider("每周学习时长（小时）", 0, 30, 10)
            attendance = st.slider("上课出勤率（%）", 0, 100, 30) / 100
            midterm_score = st.slider("期中考试分数", 0, 100, 18)
            homework_rate = st.slider("作业完成率（%）", 0, 100, 93) / 100

    # 预测逻辑+结果展示
    if submit_btn and model is not None:
        # 模型预测
        input_feat = np.array([[study_hours, attendance, midterm_score, homework_rate]])
        input_scaled = scaler.transform(input_feat)
        pred_score = model.predict(input_scaled)[0].round(1)
        pred_score = max(0, min(100, pred_score))

        # 展示结果（中文）
        st.subheader("预测结果")
        st.write(f"预测期末成绩：{pred_score} 分")
        
        # 学习建议+图片
        if pred_score >= 60:
            st.success("🎉 预测成绩及格~建议保持当前学习状态，巩固薄弱知识点！")
            st.image("D:/streamlit_env/congratulations.jpg", width=400)
        else:
            st.warning("💪 预测成绩未及格~建议增加学习时长、提高出勤率，及时向老师和同学请教！")
            st.image("D:/streamlit_env/sad.jpg", width=400)
    elif submit_btn:
        st.error("❌ 模型未加载，无法预测")
