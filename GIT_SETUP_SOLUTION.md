# Git 安装和配置解决方案

## 问题诊断
✅ **确认问题**: Git 命令无法识别，说明 Git 未安装或未正确配置环境变量。

## 解决方案

### 方案一：安装 Git（推荐）

1. **下载 Git**
   - 访问官方网站：https://git-scm.com/download/windows
   - 下载最新版本的 Git for Windows

2. **安装 Git**
   - 运行下载的安装程序
   - **重要**：在安装过程中，确保选择 "Add Git to PATH" 选项
   - 使用默认设置完成安装

3. **重启终端**
   - 安装完成后，**必须重启所有终端窗口**
   - 或者重启整个 IDE

### 方案二：手动配置环境变量（如果已安装但无法识别）

1. **查找 Git 安装路径**
   - 通常在：`C:\Program Files\Git\bin`
   - 或：`C:\Program Files (x86)\Git\bin`

2. **添加到 PATH 环境变量**
   - 右键 "此电脑" → "属性" → "高级系统设置"
   - 点击 "环境变量"
   - 在 "系统变量" 中找到 "Path"
   - 点击 "编辑" → "新建"
   - 添加 Git 的 bin 目录路径
   - 确定保存

3. **重启终端**
   - 配置完成后重启所有终端窗口

### 方案三：使用 GitHub Desktop（图形化替代）

如果命令行 Git 配置困难，可以使用 GitHub Desktop：

1. **下载 GitHub Desktop**
   - 访问：https://desktop.github.com/
   - 下载并安装

2. **登录 GitHub 账户**
   - 打开 GitHub Desktop
   - 使用您的 GitHub 账户登录

3. **克隆仓库**
   - File → Clone repository
   - 选择您的 `building-performance-optimizer` 仓库

4. **同步文件**
   - 将修复后的文件复制到克隆的本地仓库
   - 在 GitHub Desktop 中提交并推送更改

##