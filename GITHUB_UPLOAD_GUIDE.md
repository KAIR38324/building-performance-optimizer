# GitHub 上传指南

## 📋 当前状态

✅ **已完成的步骤：**
- Git 仓库已初始化
- 部署文件已添加到 Git
- 首次提交已完成

## 🚀 接下来的步骤

### 1. 创建 GitHub 仓库

1. 打开 [GitHub](https://github.com)
2. 登录你的 GitHub 账户
3. 点击右上角的 "+" 按钮，选择 "New repository"
4. 填写仓库信息：
   - **Repository name**: `streamlit-building-performance-app`（或你喜欢的名称）
   - **Description**: `Building Performance Prediction App with Multi-AI Provider Support`
   - **Visibility**: 选择 Public（Streamlit Community Cloud 需要公开仓库）
   - **不要**勾选 "Add a README file"、"Add .gitignore"、"Choose a license"（我们已经有了这些文件）
5. 点击 "Create repository"

### 2. 连接本地仓库到 GitHub

创建仓库后，GitHub 会显示连接指令。复制以下命令并在终端中执行：

```bash
# 添加远程仓库（替换 YOUR_USERNAME 为你的 GitHub 用户名）
git remote add origin https://github.com/YOUR_USERNAME/streamlit-building-performance-app.git

# 推送代码到 GitHub
git branch -M main
git push -u origin main
```

### 3. 验证上传

上传完成后，你应该能在 GitHub 仓库中看到以下文件：
- `app_deploy.py` - 部署版本的主应用
- `requirements_deploy.txt` - 部署依赖
- `.streamlit/config.toml` - Streamlit 配置
- `.streamlit/secrets.toml.example` - 环境变量示例
- `DEPLOY_GUIDE.md` - 部署指南
- `README_deploy.md` - 项目说明
- `test_deploy.py` - 测试文件
- `.gitignore` - Git 忽略文件

## 🔧 使用终端命令

如果你想使用终端完成上传，可以执行以下命令：

```powershell
# 添加远程仓库（记得替换 YOUR_USERNAME）
& "C:\Program Files\Git\bin\git.exe" remote add origin https://github.com/YOUR_USERNAME/streamlit-building-performance-app.git

# 重命名分支为 main
& "C:\Program Files\Git\bin\git.exe" branch -M main

# 推送到 GitHub
& "C:\Program Files\Git\bin\git.exe" push -u origin main
```

## 📝 重要提醒

1. **替换用户名**: 确保将 `YOUR_USERNAME` 替换为你的实际 GitHub 用户名
2. **仓库名称**: 可以使用建议的名称或自定义名称
3. **公开仓库**: Streamlit Community Cloud 只支持公开仓库
4. **文件完整性**: 确保所有部署文件都已上传

## 🎯 下一步：Streamlit Cloud 部署

上传到 GitHub 后，你就可以：
1. 访问 [Streamlit Community Cloud](https://share.streamlit.io/)
2. 连接你的 GitHub 账户
3. 选择刚上传的仓库
4. 指定 `app_deploy.py` 作为主文件
5. 配置环境变量（API 密钥）
6. 部署应用

## 🆘 遇到问题？

- **Git 命令不识别**: 重启终端或使用完整路径
- **推送失败**: 检查网络连接和 GitHub 凭据
- **仓库不存在**: 确认仓库已在 GitHub 上创建
- **权限问题**: 确保你有仓库的写入权限

---

**准备好了吗？** 现在就去创建你的 GitHub 仓库并上传代码吧！ 🚀