# GitHub 上传完整指南

## ✅ 已完成的步骤

1. **Git 安装和配置** ✅
   - Git 已成功安装并配置
   - PATH 环境变量已修复
   - 用户信息已配置（需要您自己修改为真实信息）

2. **本地 Git 仓库准备** ✅
   - Git 仓库已初始化
   - 部署文件已添加到仓库
   - 首次提交已完成

## 🔄 接下来需要您完成的步骤

### 步骤 1：修改 Git 用户信息（重要）

当前使用的是临时用户信息，请修改为您的真实信息：

```powershell
# 在终端中执行以下命令，替换为您的真实信息
git config --global user.name "您的真实姓名"
git config --global user.email "您的GitHub邮箱地址"
```

### 步骤 2：在 GitHub 上创建新仓库

1. 访问 GitHub：https://github.com
2. 登录您的 GitHub 账户
3. 点击右上角的 "+" 按钮
4. 选择 "New repository"
5. 填写仓库信息：
   - **Repository name**：`streamlit-building-performance-app`（或您喜欢的名称）
   - **Description**：`Building Performance Optimization App with Multi-AI Provider Support`
   - **Visibility**：选择 "Public"（公开仓库，免费用户必须选择此项）
   - **不要**勾选 "Add a README file"、"Add .gitignore" 或 "Choose a license"
6. 点击 "Create repository"

### 步骤 3：连接本地仓库到 GitHub

创建 GitHub 仓库后，您会看到一个页面显示如何推送现有仓库。执行以下命令：

```powershell
# 添加远程仓库（替换为您的实际仓库地址）
git remote add origin https://github.com/您的用户名/您的仓库名.git

# 设置主分支名称
git branch -M main

# 推送到 GitHub
git push -u origin main
```

**示例**（假设您的 GitHub 用户名是 `yourname`，仓库名是 `streamlit-building-performance-app`）：
```powershell
git remote add origin https://github.com/yourname/streamlit-building-performance-app.git
git branch -M main
git push -u origin main
```

### 步骤 4：验证上传成功

1. 刷新您的 GitHub 仓库页面
2. 确认以下文件已成功上传：
   - `app_deploy.py`
   - `requirements_deploy.txt`
   - `.streamlit/config.toml`
   - `.streamlit/secrets.toml.example`
   - `README_deploy.md`
   - `DEPLOY_GUIDE.md`
   - 模型文件（`.pth` 文件）
   - 数据文件（`.csv` 文件）

## 🚀 下一步：部署到 Streamlit Cloud

### 步骤 1：访问 Streamlit Cloud

1. 访问：https://share.streamlit.io/
2. 使用您的 GitHub 账户登录

### 步骤 2：创建新应用

1. 点击 "New app"
2. 选择您刚刚创建的仓库
3. 选择分支：`main`
4. 选择主文件：`app_deploy.py`
5. 点击 "Deploy!"

### 步骤 3：配置环境变量（API 密钥）

在 Streamlit Cloud 的应用设置中，添加以下环境变量：

#### 如果使用 OpenAI：
- **Key**: `OPENAI_API_KEY`
- **Value**: 您的 OpenAI API 密钥

#### 如果使用 Google AI Studio：
- **Key**: `GOOGLE_API_KEY`
- **Value**: 您的 Google AI Studio API 密钥

#### 如果使用 DeepSeek：
- **Key**: `DEEPSEEK_API_KEY`
- **Value**: 您的 DeepSeek API 密钥

### 步骤 4：获取 API 密钥

#### OpenAI API 密钥：
1. 访问：https://platform.openai.com/api-keys
2. 登录并创建新的 API 密钥

#### Google AI Studio API 密钥：
1. 访问：https://aistudio.google.com/app/apikey
2. 登录并创建新的 API 密钥

#### DeepSeek API 密钥：
1. 访问：https://platform.deepseek.com/api_keys
2. 注册并创建新的 API 密钥

## 📋 当前项目文件清单

✅ **已准备的部署文件**：
- `app_deploy.py` - 主应用文件（支持多AI提供商）
- `requirements_deploy.txt` - Python依赖包
- `.streamlit/config.toml` - Streamlit配置
- `.streamlit/secrets.toml.example` - 环境变量模板
- `README_deploy.md` - 部署说明文档
- `DEPLOY_GUIDE.md` - 详细部署指南
- 模型文件：`CEUI_trained_model.pth`、`DGI_trained_model.pth`、`UDI_trained_model.pth`
- 数据文件：`帕累托解集.csv`

## 🔧 故障排除

### 如果推送到 GitHub 时遇到认证问题：

1. **使用 Personal Access Token**：
   - 访问 GitHub Settings → Developer settings → Personal access tokens
   - 创建新的 token
   - 在推送时使用 token 作为密码

2. **或者使用 GitHub CLI**：
   ```powershell
   # 安装 GitHub CLI
   winget install GitHub.cli
   
   # 登录
   gh auth login
   
   # 推送
   git push -u origin main
   ```

### 如果 Streamlit Cloud 部署失败：

1. 检查 `requirements_deploy.txt` 中的依赖包版本
2. 确保所有必要的文件都已上传
3. 检查应用日志中的错误信息
4. 确认 API 密钥已正确配置

## 🎉 部署成功后

部署成功后，您将获得一个公网地址，类似于：
`https://your-app-name.streamlit.app`

您的建筑性能优化应用将可以通过这个地址在全球范围内访问！

## 📞 需要帮助？

如果在任何步骤中遇到问题，请：
1. 检查错误信息
2. 参考相关文档
3. 寻求技术支持

---

**重要提醒**：
- 确保 API 密钥的安全性，不要在代码中硬编码
- 定期检查 API 使用量和费用
- 保持依赖包的更新