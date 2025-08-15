# 🚀 快速部署指南

## 📋 部署文件清单

### ✅ 已创建的部署文件
- `app_deploy.py` - 云部署版应用（支持OpenAI、Google Gemini、DeepSeek三种AI服务）
- `requirements_deploy.txt` - 云部署依赖包（已修复torch问题）
- `requirements_streamlit_cloud.txt` - Streamlit Cloud专用依赖配置
- `STREAMLIT_CLOUD_DEPLOYMENT_GUIDE.md` - Streamlit Cloud部署最佳实践
- `.streamlit/config.toml` - Streamlit配置
- `.streamlit/secrets.toml.example` - API密钥配置模板
- `README_deploy.md` - 详细部署文档
- `.gitignore` - Git忽略文件配置
- `deploy_test.py` - 本地测试脚本

### 📁 原始文件（保持不变）
- `app.py` - 原始应用（使用Ollama）
- `requirements.txt` - 原始依赖包
- `*.pth` - 训练好的模型文件
- `scaler.pkl` - 数据标准化器
- `帕累托解集.csv` - 帕累托解集数据

## 🔧 本地测试部署版本

### 1. 安装依赖
```bash
pip install -r requirements_deploy.txt
```

### 2. 配置API密钥
```bash
# 复制配置模板
copy .streamlit\secrets.toml.example .streamlit\secrets.toml

# 编辑 .streamlit/secrets.toml，将以下内容：
# GOOGLE_API_KEY = "your-google-api-key-here"
# 替换为您的实际Google API密钥
```

### 3. 运行测试脚本
```bash
python deploy_test.py
```

### 4. 手动运行（可选）
```bash
streamlit run app_deploy.py
```

## ☁️ Streamlit Community Cloud 部署

### 步骤1: 准备GitHub仓库
1. 在GitHub创建新仓库
2. 上传以下文件到仓库：
   ```
   app_deploy.py
   requirements_deploy.txt
   UDI_trained_model.pth
   DGI_trained_model.pth
   CEUI_trained_model.pth
   scaler.pkl
   帕累托解集.csv
   .streamlit/config.toml
   README_deploy.md
   .gitignore
   ```

### 步骤2: 部署到Streamlit Cloud
1. 访问 [share.streamlit.io](https://share.streamlit.io)
2. 使用GitHub账户登录
3. 点击 "New app"
4. 选择您的仓库
5. 设置主文件为 `app_deploy.py`
6. 点击 "Deploy!"

### 步骤3: 配置API密钥
在Streamlit Cloud应用设置中添加：
```toml
GOOGLE_API_KEY = "your-actual-google-api-key-here"
```

## 🔑 AI服务API密钥获取

### OpenAI API密钥
1. **访问OpenAI平台**
   - 打开 https://platform.openai.com/api-keys
   - 使用您的OpenAI账户登录

2. **创建API密钥**
   - 点击 "Create new secret key" 按钮
   - 为密钥命名并复制生成的密钥

### Google AI Studio API密钥
1. **访问Google AI Studio**
   - 打开 https://aistudio.google.com/app/apikey
   - 使用您的Google账户登录

2. **创建API密钥**
   - 点击 "Create API Key" 按钮
   - 选择一个现有项目或创建新项目
   - 复制生成的API密钥

### DeepSeek API密钥
1. **访问DeepSeek平台**
   - 打开 https://platform.deepseek.com/api_keys
   - 注册并登录您的DeepSeek账户

2. **创建API密钥**
   - 点击 "Create API Key" 按钮
   - 复制生成的API密钥

### 配置说明
- **您只需要配置其中一种AI服务即可**
- 在应用中可以通过侧边栏选择要使用的AI服务提供商
- 在Streamlit Cloud部署时，根据您选择的服务在 "Secrets" 部分添加对应的API密钥

## 📊 功能对比

| 功能 | 原版 (app.py) | 部署版 (app_deploy.py) |
|------|---------------|------------------------|
| 性能预测 | ✅ | ✅ |
| 智能决策 | Ollama (本地) | Google Gemini API (云端) |
| GPU支持 | ✅ | ❌ (云端CPU) |
| 部署难度 | 高 | 低 |
| 运行成本 | 免费 | API费用 |

## ⚠️ 重要提醒

1. **保留原始文件**: 所有原始文件（app.py等）都已保留，不受影响
2. **API费用**: Google Gemini API有免费额度，超出后按使用量计费
3. **模型文件**: 确保所有.pth文件正确上传（文件较大，上传可能需要时间）
4. **编码问题**: 如果CSV文件包含中文，确保使用UTF-8编码

## 🔍 故障排除

### 本地测试问题
- 运行 `python deploy_test.py` 检查环境
- 确认所有依赖包已安装
- 检查API密钥配置

### 云部署问题
- 检查GitHub仓库文件完整性
- 查看Streamlit Cloud构建日志
- 确认API密钥正确配置

### 常见错误
1. **ModuleNotFoundError: torch**: 使用修复后的 `requirements_deploy.txt` 或 `requirements_streamlit_cloud.txt`
2. **模型加载失败**: 检查.pth文件是否完整上传
3. **API调用失败**: 验证API密钥配置
4. **编码错误**: 确保CSV文件使用UTF-8编码

### 🚨 Torch依赖问题解决
如果遇到 `ModuleNotFoundError: No module named 'torch'` 错误：
1. 使用修复后的 `requirements_deploy.txt`（已包含CPU版本torch）
2. 或使用 `requirements_streamlit_cloud.txt`（完整优化配置）
3. 详细解决方案请参考 `STREAMLIT_CLOUD_DEPLOYMENT_GUIDE.md`

## 📞 技术支持

如遇问题，请检查：
1. `README_deploy.md` - 详细文档
2. 应用日志中的错误信息
3. GitHub仓库文件完整性

---

**🎉 部署完成后，您将拥有一个可公网访问的建筑性能优化系统！**