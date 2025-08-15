# Git 环境变量永久配置指南

## 问题描述
Git 已安装但未添加到系统 PATH 环境变量中，导致无法在命令行中直接使用 `git` 命令。

## 临时解决方案（已完成）
✅ 在当前 PowerShell 会话中临时添加 Git 路径：
```powershell
$env:PATH += ';C:\Program Files\Git\bin'
```

## 永久解决方案

### 方法一：通过系统设置（推荐）
1. 右键点击「此电脑」→「属性」
2. 点击「高级系统设置」
3. 在「系统属性」窗口中点击「环境变量」
4. 在「系统变量」区域找到并选择「Path」
5. 点击「编辑」
6. 点击「新建」
7. 添加以下路径：
   ```
   C:\Program Files\Git\bin
   C:\Program Files\Git\cmd
   ```
8. 点击「确定」保存所有更改
9. 重启命令行或 PowerShell

### 方法二：通过 PowerShell（管理员权限）
```powershell
# 获取当前系统 PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")

# 添加 Git 路径
$newPath = $currentPath + ";C:\Program Files\Git\bin;C:\Program Files\Git\cmd"

# 设置新的 PATH
[Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
```

## 验证配置
重启命令行后，运行以下命令验证：
```bash
git --version
```

应该显示类似输出：
```
git version 2.50.1.windows.1
```

## 当前状态
✅ **Git 环境变量问题已解决**
✅ **修复的 requirements 文件已推送到 GitHub**

### 已推送的文件：
- `requirements_deploy.txt` - 修复了 torch 依赖版本
- `requirements_streamlit_cloud.txt` - 专门为 Streamlit Cloud 优化的依赖文件

## 下一步：Streamlit Cloud 重新部署
1. 访问 [Streamlit Cloud](https://share.streamlit.io/)
2. 找到你的应用并点击「Settings」
3. 在「Advanced settings」中更新：
   - **Main file path**: `app_deploy.py`
   - **Requirements file**: `requirements_streamlit_cloud.txt`
4. 保存设置并重新部署

## 注意事项
- 环境变量更改需要重启命令行才能生效
- 如果使用 IDE，可能需要重启 IDE
- 建议使用 `requirements_streamlit_cloud.txt` 进行云部署，因为它专门针对 Streamlit Cloud 的 CPU 环境优化