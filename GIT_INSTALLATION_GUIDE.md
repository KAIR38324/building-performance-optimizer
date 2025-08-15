# Git 安装指南

## 问题诊断
✅ **已确认：您的系统未安装 Git**

当您执行 `git push` 命令时出现 "无法将'git'项识别为 cmdlet" 错误，这表明系统中没有安装 Git 或 Git 不在 PATH 环境变量中。

## 解决方案

### 方案一：安装 Git（推荐）

#### 1. 下载 Git
- 访问 Git 官方网站：https://git-scm.com/download/win
- 点击 "Download for Windows" 下载最新版本
- 选择 64-bit Git for Windows Setup（推荐）

#### 2. 安装 Git
1. 运行下载的安装程序
2. 安装过程中的重要选项：
   - **Adjusting your PATH environment**：选择 "Git from the command line and also from 3rd-party software"（推荐）
   - **Choosing the default editor**：可以选择您喜欢的编辑器
   - **Configuring the line ending conversions**：选择 "Checkout Windows-style, commit Unix-style line endings"（推荐）
   - 其他选项可以保持默认设置

#### 3. 验证安装
安装完成后，重新打开 PowerShell 并执行：
```powershell
git --version
```

#### 4. 配置 Git（首次使用）
```powershell
git config --global user.name "您的用户名"
git config --global user.email "您的邮箱@example.com"
```

### 方案二：使用 GitHub Desktop（图形化界面）

如果您不想使用命令行，可以使用 GitHub Desktop：

#### 1. 下载 GitHub Desktop
- 访问：https://desktop.github.com/
- 下载并安装 GitHub Desktop

#### 2. 使用 GitHub Desktop 上传项目
1. 打开 GitHub Desktop
2. 登录您的 GitHub 账户
3. 点击 "File" → "Add Local Repository"
4. 选择您的项目文件夹
5. 点击 "Publish repository" 发布到 GitHub

### 方案三：使用 Web 界面上传

#### 1. 创建 GitHub 仓库
1. 登录 GitHub：https://github.com
2. 点击右上角的 "+" → "New repository"
3. 填写仓库名称（如：streamlit-building-performance-app）
4. 选择 "Public"（公开仓库）
5. 点击 "Create repository"

#### 2. 上传文件
1. 在新创建的仓库页面，点击 "uploading an existing file"
2. 将以下文件拖拽到页面中：
   - `app_deploy.py`
   - `requirements_deploy.txt`
   - `.streamlit/config.toml`
   - `.streamlit/secrets.toml.example`
   - `README_deploy.md`
   - `DEPLOY_GUIDE.md`
   - `models/` 文件夹中的所有文件
   - `data/` 文件夹中的所有文件
3. 添加提交信息："Initial commit for Streamlit deployment"
4. 点击 "Commit changes"

## 推荐的完整流程

### 如果选择安装 Git（推荐）：
1. 按照上述步骤安装 Git
2. 重新打开 PowerShell
3. 导航到项目目录
4. 执行以下命令：
```powershell
# 初始化 Git 仓库
git init

# 添加文件
git add app_deploy.py requirements_deploy.txt .streamlit/ README_deploy.md DEPLOY_GUIDE.md models/ data/

# 提交更改
git commit -m "Initial commit for Streamlit deployment"

# 添加远程仓库（替换为您的仓库地址）
git remote add origin https://github.com/您的用户名/您的仓库名.git

# 推送到 GitHub
git branch -M main
git push -u origin main
```

### 如果选择 GitHub Desktop：
1. 安装 GitHub Desktop
2. 使用图形界面添加本地仓库
3. 发布到 GitHub

### 如果选择 Web 上传：
1. 直接在 GitHub 网站创建仓库
2. 通过拖拽方式上传文件

## 下一步：Streamlit Cloud 部署

无论您选择哪种方式上传到 GitHub，完成后都可以：

1. 访问 https://share.streamlit.io/
2. 使用 GitHub 账户登录
3. 点击 "New app"
4. 选择您的仓库和 `app_deploy.py` 文件
5. 配置环境变量（API 密钥）
6. 部署应用

## 故障排除

### 如果 Git 安装后仍然无法识别：
1. 重启 PowerShell
2. 检查环境变量 PATH 是否包含 Git 安装路径
3. 手动添加 Git 到 PATH（通常是 `C:\Program Files\Git\cmd`）

### 如果遇到权限问题：
1. 以管理员身份运行 PowerShell
2. 或者使用 GitHub Desktop 等图形化工具

---

**建议**：如果您是 Git 新手，推荐使用 GitHub Desktop，它提供了友好的图形界面，避免了命令行的复杂性。