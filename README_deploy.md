# 建筑性能预测系统 - 云部署版

这是建筑性能预测系统的云部署版本，专为Streamlit Community Cloud优化。

## 🌟 主要特性

- **智能建筑性能预测**：基于深度学习模型的UDI、DGI、CEUI指标预测
- **多AI服务支持**：支持OpenAI、Google Gemini、DeepSeek三种AI服务提供商
- **灵活配置**：通过侧边栏选择AI服务提供商，只需填写对应API密钥
- **多方案智能决策**：结合AI的智能决策支持系统
- **云端部署**：无需本地环境，直接在浏览器中使用
- **实时交互**：Streamlit提供的现代化Web界面

## 📁 文件说明

### 核心文件
- `app_deploy.py` - 主应用程序（云部署版本）
- `requirements_deploy.txt` - Python依赖包列表
- `UDI_trained_model.pth` - UDI预测模型
- `DGI_trained_model.pth` - DGI预测模型
- `CEUI_trained_model.pth` - CEUI预测模型
- `scaler.pkl` - 数据标准化器
- `帕累托解集.csv` - 帕累托最优解集数据

### 配置文件
- `.streamlit/config.toml` - Streamlit应用配置
- `.streamlit/secrets.toml.example` - API密钥配置模板

## 🔧 本地部署

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements_deploy.txt
```

### 2. 配置API密钥
```bash
# 复制配置模板
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# 编辑配置文件，填入您的Google Gemini API密钥
# GOOGLE_API_KEY = "your-actual-api-key-here"
```

### 3. 运行应用
```bash
streamlit run app_deploy.py
```

## ☁️ Streamlit Community Cloud 部署

### 1. 准备GitHub仓库
1. 创建新的GitHub仓库
2. 上传所有文件到仓库
3. 确保包含以下文件：
   - `app_deploy.py`
   - `requirements_deploy.txt`
   - 所有模型文件（.pth）
   - `scaler.pkl`
   - `帕累托解集.csv`
   - `.streamlit/config.toml`

### 2. 配置Streamlit Cloud
1. 访问 [share.streamlit.io](https://share.streamlit.io)
2. 使用GitHub账户登录
3. 点击 "New app"
4. 选择您的仓库和分支
5. 设置主文件路径为 `app_deploy.py`

### 3. 配置环境变量
在Streamlit Cloud的应用设置中添加以下secrets：
```toml
GOOGLE_API_KEY = "your-google-api-key-here"
```

### 4. 部署完成
- 应用将自动构建和部署
- 获得公网访问地址
- 支持自动更新（推送到GitHub时触发）

## 🔑 获取Google Gemini API密钥

1. 访问 [Google AI Studio](https://aistudio.google.com/app/apikey)
2. 登录您的Google账户
3. 点击 "Create API Key"
4. 复制生成的密钥
5. 将密钥配置到应用中

## 📊 系统架构

### 模型架构
- **UDI/DGI模型**: 3层全连接神经网络（256-512-256-1）
- **CEUI模型**: 带BatchNorm和Dropout的深度网络
- **输入特征**: 10个建筑设计参数
- **预测指标**: UDI、DGI、CEUI三个性能指标

### 决策算法
1. **偏好分析**: 使用Google Gemini模型解析自然语言偏好
2. **权重分配**: 自动生成三个指标的权重分配
3. **方案排序**: 基于加权评分对帕累托解集排序
4. **智能推荐**: 返回最优的5个设计方案

## 🛠️ 技术栈

- **前端**: Streamlit
- **机器学习**: PyTorch
- **数据处理**: Pandas, NumPy, Scikit-learn
- **AI服务**: Google Gemini API
- **部署**: Streamlit Community Cloud

## 📝 使用说明

### 选择AI服务提供商
1. 在侧边栏选择您偏好的AI服务（OpenAI、Google Gemini或DeepSeek）
2. 点击对应的API密钥获取链接
3. 在输入框中填写您的API密钥

### 快速性能预测
1. 在侧边栏选择"快速性能预测"
2. 使用滑块调整设计参数
3. 点击"开始预测"查看结果（UDI、DGI、CEUI指标）

### 智能决策推荐
1. 在侧边栏选择"多方案智能决策"
2. 在文本框中描述您的设计偏好和需求
3. AI将分析并推荐最优的建筑设计方案
4. 基于帕累托最优解集的科学决策支持

## ⚠️ 注意事项

1. **API费用**: Google Gemini API有免费额度，超出后按使用量计费
2. **模型文件**: 确保所有.pth文件正确上传到仓库
3. **数据文件**: 确保CSV和PKL文件编码正确
4. **网络连接**: 智能决策功能需要稳定的网络连接

## 🔍 故障排除

### 常见问题
1. **模型加载失败**: 检查.pth文件是否完整上传
2. **API调用失败**: 验证Google Gemini API密钥是否正确配置
3. **数据加载错误**: 确认CSV文件格式和编码
4. **预测结果异常**: 检查输入参数是否在合理范围内

### 联系支持
如遇到技术问题，请检查：
1. 应用日志中的错误信息
2. GitHub仓库中的文件完整性
3. Streamlit Cloud的构建日志

## 📄 许可证

本项目仅供学术研究使用。

---

**版本**: 1.0.0 (云部署版)  
**更新时间**: 2024年  
**技术支持**: 建筑性能优化研究团队