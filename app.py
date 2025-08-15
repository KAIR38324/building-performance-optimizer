import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser # 正确的导入路径
import json

# --- 0. 全局设置与资源加载 ---

st.set_page_config(page_title="建筑性能优化与决策支持系统", layout="wide")

# --- 1. 修正后的模型定义 ---

# 为 UDI 和 DGI 定义一个模型类
# 这个结构与您的 UDI_train.py 和 DGI_train.py 完全一致
class UDI_DGI_Model(nn.Module):
    def __init__(self, input_features=10):
        super().__init__()
        self.layer1 = nn.Linear(input_features, 256)  
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 256)
        self.output_layer = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.output_layer(x)
        # 移除 .squeeze() 以便 .item() 可以直接使用
        return x

# 为 CEUI 定义一个独立的、更复杂的模型类
# 这个结构与您的 CEUI_train.py 完全一致
class CEUI_Model(nn.Module):
    def __init__(self, input_features=10):
        super().__init__()
        self.layer1 = nn.Linear(input_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)
        self.layer3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.output_layer = nn.Linear(256, 1)

    def forward(self, x):
        # 确保在推理时，输入是二维的 [batch_size, features]
        # BatchNorm1d 层期望至少二维的输入
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = self.dropout1(torch.relu(self.bn1(self.layer1(x))))
        x = self.dropout2(torch.relu(self.bn2(self.layer2(x))))
        x = torch.relu(self.bn3(self.layer3(x)))
        x = self.output_layer(x)
        # 移除 .squeeze() 以便 .item() 可以直接使用
        return x

@st.cache_resource
def load_models_and_scaler():
    """加载所有PyTorch模型和数据标准化Scaler"""
    try:
        # 使用正确的类来实例化每个模型
        udi_model = UDI_DGI_Model(input_features=10)
        dgi_model = UDI_DGI_Model(input_features=10)
        ceui_model = CEUI_Model(input_features=10)

        # 确定设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.sidebar.success(f"模型将运行在: {str(device).upper()}")

        # 加载模型权重，并映射到正确的设备
        udi_model.load_state_dict(torch.load("UDI_trained_model.pth", map_location=device))
        dgi_model.load_state_dict(torch.load("DGI_trained_model.pth", map_location=device))
        ceui_model.load_state_dict(torch.load("CEUI_trained_model.pth", map_location=device))

        # 将模型本身也移动到正确的设备
        udi_model.to(device)
        dgi_model.to(device)
        ceui_model.to(device)

        # 切换到评估模式 (对于有Dropout和BatchNorm的模型至关重要)
        udi_model.eval()
        dgi_model.eval()
        ceui_model.eval()

        # 加载Scaler对象
        scaler = joblib.load('scaler.pkl') 
        
        return udi_model, dgi_model, ceui_model, scaler, device
    except FileNotFoundError as e:
        st.error(f"加载文件时出错: {e}。请确保所有模型文件 (UDI_trained_model.pth, DGI_trained_model.pth, CEUI_trained_model.pth) 和 scaler.pkl 都在应用根目录下。")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"加载资源时发生未知错误: {e}")
        return None, None, None, None, None

@st.cache_data
def load_pareto_data():
    """加载帕累托解集数据"""
    try:
        df = pd.read_csv("帕累托解集.csv")
        return df
    except FileNotFoundError as e:
        st.error(f"加载数据文件时出错: {e}。请确保 '帕累-托解集.csv' 文件存在。")
        return pd.DataFrame()

# 加载所有资源
udi_model, dgi_model, ceui_model, scaler, device = load_models_and_scaler()
df_solutions = load_pareto_data()

# --- 2. 场景一：快速性能预测页面 ---

def prediction_page():
    st.header("场景一：快速性能预测")
    st.markdown("通过滑块输入单个设计方案的参数，系统将调用预测模型计算其性能。")
    
    feature_names = ['开间', '进深', '层高', '北侧窗台高', '北侧窗高', '北侧窗墙比', '南侧窗墙比', '南侧窗高', '南侧窗台高', '南侧窗间距']
    input_vars = {}
    
    st.info("请拖动下面的滑块来设置设计参数值:")
    cols = st.columns(5)
    # 设定一些更合理的默认值和范围
    default_values = {'开间': 7.2, '进深': 9.0, '层高': 3.6, '北侧窗台高': 0.9, '北侧窗高': 1.8, '北侧窗墙比': 0.3, '南侧窗墙比': 0.5, '南侧窗高': 1.8, '南侧窗台高': 0.9, '南侧窗间距': 1.2}
    min_max_step = {'开间': (3.0, 10.0, 0.1), '进深': (5.0, 12.0, 0.1), '层高': (2.8, 5.0, 0.1), '北侧窗台高': (0.5, 1.5, 0.1), '北侧窗高': (1.0, 3.0, 0.1), '北侧窗墙比': (0.1, 0.8, 0.05), '南侧窗墙比': (0.1, 0.8, 0.05), '南侧窗高': (1.0, 3.0, 0.1), '南侧窗台高': (0.5, 1.5, 0.1), '南侧窗间距': (0.5, 3.0, 0.1)}

    for i, name in enumerate(feature_names):
        with cols[i % 5]:
            min_val, max_val, step_val = min_max_step[name]
            input_vars[name] = st.slider(f'{name}', min_value=min_val, max_value=max_val, value=default_values[name], step=step_val)

    if st.button("开始预测", key="predict_button", use_container_width=True):
        if not all([udi_model, dgi_model, ceui_model, scaler, device]):
            st.error("模型或Scaler未成功加载，无法进行预测。请检查文件是否存在并刷新页面。")
            return

        with st.spinner('正在进行标准化处理和模型预测...'):
            try:
                # 1. 整理输入数据并确保顺序正确
                input_data = np.array([[input_vars[name] for name in feature_names]])
                
                # 2. 使用加载的scaler进行标准化
                scaled_data = scaler.transform(input_data)
                
                # 3. 转换为PyTorch张量，并移动到正确的设备
                input_tensor = torch.FloatTensor(scaled_data).to(device)

                # 4. 调用模型进行预测
                with torch.no_grad():
                    pred_udi = udi_model(input_tensor).item()
                    pred_dgi = dgi_model(input_tensor).item()
                    pred_ceui = ceui_model(input_tensor).item()
                
                st.subheader("🚀 性能预测结果:")
                c1, c2, c3 = st.columns(3)
                c1.metric("有效采光照度 (UDI)", f"{pred_udi:.2f} %")
                c2.metric("不舒适眩光指数 (DGI)", f"{pred_dgi:.2f}")
                c3.metric("制冷能耗强度 (CEUI)", f"{pred_ceui:.2f} kWh/m²")

            except Exception as e:
                st.error(f"预测过程中发生错误: {e}")

# --- 2. 场景二：多方案智能决策页面 (优化后版本，无硬性筛选) ---
def decision_page():
    st.header("场景二：多方案智能决策")
    st.markdown("用自然语言描述您的设计偏好，LLM将为您从帕累托最优方案中智能推荐。")
    
    # 预设的帕累托解集列名与用户友好名称的映射
    feature_names = ['开间', '进深', '层高', '北侧窗台高', '北侧窗高', '北侧窗墙比', '南侧窗墙比', '南侧窗高', '南侧窗台高', '南侧窗间距']

    user_preference = st.text_area(
        "请输入您的设计偏好:",
        placeholder="例如：我希望教室能尽可能地节能，同时严格控制眩光，采光达到良好水平即可。",
        height=150
    )

    if st.button("开始智能推荐", key="decision_button", use_container_width=True):
        if df_solutions.empty:
            st.error("帕累托解集数据 ('帕累托解集.csv') 未成功加载，无法进行决策。")
            return
        if not user_preference:
            st.warning("请输入您的设计偏好！")
            return

        with st.spinner('AI专家正在分析您的偏好并进行决策...'):
            parser = JsonOutputParser()
            
            # --- 这是修改的核心部分：新的Prompt模板 ---
            prompt_template = """
            你是一位顶尖的建筑设计与性能优化专家。你的任务是分析用户的自然语言设计偏好，并将其转化为一个包含'weights'和'reasoning'的JSON对象。
            
            # 指标含义说明：
            - UDI (有效采光照度): 代表室内“采光”水平，数值越高越好。
            - CEUI (制冷能耗强度): 代表建筑“能耗”，数值越低越好。
            - DGI (不舒适眩光指数): 代表室内“眩光”程度，数值越低越好。

            # 你的任务：
            1.  仔细阅读用户的偏好描述: '{preference}'
            2.  **不要使用任何硬性筛选条件。**
            3.  生成一个'weights'字典，表达三个指标的相对重要性。权重越高的指标，代表用户越关心。三个权重的总和必须为1.0。
            4.  生成一个'reasoning'字符串，用一两句话简要解释你为什么这样分配权重。

            # 示例：
            如果用户说："我最关心的是节能，眩光其次，采光差不多就行。"
            你的输出应该是类似这样的JSON:
            {{
                "weights": {{
                    "UDI": 0.2,
                    "CEUI": 0.5,
                    "DGI": 0.3
                }},
                "reasoning": "根据用户的描述，节能(CEUI)是首要目标，因此分配了最高权重。眩光(DGI)为次要目标，采光(UDI)要求不高，权重最低。"
            }}

            {format_instructions}
            """

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["preference"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )
            
            # 使用您指定的deepseek-rl模型
            llm = Ollama(model="deepseek-r1:7b")
            chain = prompt | llm | parser

            try:
                # 调用LLM获取决策策略
                decision_strategy = chain.invoke({"preference": user_preference})
                
                st.subheader("AI专家对您偏好的解读:")
                st.info(f"推荐理由: {decision_strategy.get('reasoning', '无')}")
                st.json({"weights": decision_strategy.get('weights')}) # 只显示权重部分

                df_processed = df_solutions.copy()

                # --- 核心排序逻辑 (对所有方案进行操作) ---
                
                # 1. 归一化 (Normalization)
                # DGI和CEUI是成本型指标（越小越好），UDI是效益型指标（越大越好）
                # 我们的目标是将所有指标都转换为“越大越好”的得分
                for col in ['UDI', 'DGI', 'CEUI']:
                    min_val, max_val = df_processed[col].min(), df_processed[col].max()
                    if (max_val - min_val) > 0:
                        if col == 'UDI': # 效益型指标，直接归一化
                            df_processed[f'{col}_norm'] = (df_processed[col] - min_val) / (max_val - min_val)
                        else: # 成本型指标 (DGI, CEUI)，反向归一化
                            df_processed[f'{col}_norm'] = (max_val - df_processed[col]) / (max_val - min_val)
                    else:
                        # 如果一个指标在所有方案中都一样，那么它的归一化得分为1（或0.5），不影响排序
                        df_processed[f'{col}_norm'] = 1.0

                # 2. 加权评分 (Weighted Scoring)
                weights = decision_strategy.get('weights', {})
                w_udi = weights.get('UDI', 1/3) # 如果LLM没给，就用默认等权重
                w_dgi = weights.get('DGI', 1/3)
                w_ceui = weights.get('CEUI', 1/3)
                
                # 所有归一化后的指标都是“越大越好”，所以直接加权求和
                df_processed['score'] = (
                    df_processed['UDI_norm'] * w_udi + 
                    df_processed['DGI_norm'] * w_dgi + 
                    df_processed['CEUI_norm'] * w_ceui
                )
                
                # 3. 排序和展示 (score 越大越好)
                final_recommendations = df_processed.sort_values(by='score', ascending=False).head(5)
                
                st.subheader("为您推荐的最佳5个方案:")
                # 确保只展示数据集中存在的列
                display_cols = ['UDI', 'DGI', 'CEUI', 'score'] + [col for col in feature_names if col in final_recommendations.columns]
                st.dataframe(final_recommendations[display_cols].style.format("{:.2f}"))

            except Exception as e:
                st.error(f"处理过程中发生错误，请检查Ollama服务是否开启且模型名称'deepseek-rl:7b'是否正确: {e}")
                st.exception(e) # 打印详细的错误堆栈，方便调试

# --- 4. 主程序与导航 ---
def main():
    st.sidebar.title("功能导航")
    
    # 检查资源是否加载成功
    if not all([udi_model, dgi_model, ceui_model, scaler, device]):
        st.sidebar.error("核心资源加载失败，请检查文件或刷新。")
        st.header("系统初始化失败")
        st.write("无法加载必要的模型或数据文件，请确保以下文件与`app.py`在同一目录下，并检查终端错误信息：")
        st.code("""
- UDI_trained_model.pth
- DGI_trained_model.pth
- CEUI_trained_model.pth
- scaler.pkl
- 帕累托解集.csv
        """)
        return

    page_options = ["快速性能预测", "多方案智能决策"]
    page = st.sidebar.radio("请选择一个功能场景", page_options)

    if page == "快速性能预测":
        prediction_page()
    elif page == "多方案智能决策":
        decision_page()

if __name__ == "__main__":
    main()