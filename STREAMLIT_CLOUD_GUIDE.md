# Streamlit Community Cloud 部署指南

## 🌟 概述

本指南将帮助你将建筑性能预测应用部署到 Streamlit Community Cloud，让全世界都能访问你的应用！

## 📋 前置条件

✅ **确保已完成：**
- GitHub 仓库已创建并上传代码
- 所有部署文件已准备就绪
- 至少一个 AI 服务提供商的 API 密钥

## 🚀 部署步骤

### 1. 访问 Streamlit Community Cloud

1. 打开 [Streamlit Community Cloud](https://share.streamlit.io/)
2. 点击 "Sign up" 或 "Sign in"
3. 选择 "Continue with GitHub" 登录

### 2. 连接 GitHub 仓库

1. 登录后，点击 "New app"
2. 选择你的 GitHub 仓库：
   - **Repository**: 选择你刚创建的仓库（如 `streamlit-building-performance-app`）
   - **Branch**: 选择 `main`
   - **Main file path**: 输入 `app_deploy.py`
3. 点击 "Deploy!"

### 3. 配置环境变量（API 密钥）

部署后，你需要配置 API 密钥：

1. 在应用页面，点击右下角的 "⚙️ Settings"
2. 选择 "Secrets" 标签
3. 根据你选择的 AI 服务提供商，添加相应的密钥：

#### OpenAI API 配置
```toml
OPENAI_API_KEY = "your-openai-api-key-here"
```

#### Google AI Studio Gemini 配置
```toml
GOOGLE_API_KEY = "your-google-api-key-here"
```

#### DeepSeek API 配置
```toml
DEEPSEEK_API_KEY = "your-deepseek-api-key-here"
```

#### 多服务提供商配置（推荐）
```toml
OPENAI_API_KEY = "your-openai-api-key-here"
GOOGLE_API_KEY = "your-google-api-key-here"
DEEPSEEK_API_KEY = "your-deepseek-api-key-here"
```

4. 点击 "Save" 保存配置
5. 应用会自动重启

## 🔑 获取 API 密钥

### OpenAI API 密钥
1. 访问 [OpenAI API Keys](https://platform.openai.com/api-keys)
2. 登录你的 OpenAI 账户
3. 点击 "Create new secret key"
4. 复制生成的密钥

### Google AI Studio API 密钥
1. 访问 [Google AI Studio](https://aistudio.google.com/app/apikey)
2. 登录你的 Google 账户
3. 点击 "Create API Key"
4. 复制生成的密钥

### DeepSeek API 密钥
1. 访问 [DeepSeek Platform](https://platform.deepseek.com/api_keys)
2. 注册并登录账户
3. 创建新的 API 密钥
4. 复制生成的密钥

## 🎯 应用功能

部署成功后，你的应用将具备以下功能：

### 🏗️ 建筑性能预测
- 输入建筑参数（面积、高度、朝向等）
- 选择建筑类型和气候区域
- 获取能耗和舒适度预测

### 🤖 AI 智能分析
- 多 AI 服务提供商支持
- 智能建议和优化方案
- 自然语言交互

### 📊 数据可视化
- 交互式图表展示
- 性能对比分析
- 导出功能

## 🔧 管理你的应用

### 查看应用状态
- 在 Streamlit Cloud 控制台查看应用运行状态
- 监控访问量和性能指标
- 查看错误日志

### 更新应用
1. 在本地修改代码
2. 提交并推送到 GitHub
3. Streamlit Cloud 会自动重新部署

### 重启应用
- 在应用设置中点击 "Reboot app"
- 或修改任何文件触发自动重启

## 🌐 分享你的应用

部署成功后，你会获得一个公网地址，格式如下：
```
https://your-app-name-random-string.streamlit.app/
```

你可以：
- 分享给同事和朋友
- 嵌入到网站中
- 用于演示和展示

## 🆘 故障排除

### 常见问题

**应用启动失败**
- 检查 `requirements_deploy.txt` 中的依赖
- 确认 `app_deploy.py` 文件路径正确
- 查看错误日志

**API 调用失败**
- 验证 API 密钥是否正确
- 检查 API 配额和余额
- 确认网络连接

**模型加载错误**
- 确认 `scaler.pkl` 文件存在
- 检查文件路径和权限
- 查看详细错误信息

### 获取帮助

- [Streamlit Community Cloud 文档](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit 论坛](https://discuss.streamlit.io/)
- [GitHub Issues](https://github.com/streamlit/streamlit/issues)

## 🎉 恭喜！

如果一切顺利，你的建筑性能预测应用现在已经在云端运行了！

**下一步你可以：**
- 测试所有功能
- 分享给用户
- 收集反馈
- 持续改进

---

**享受你的云端应用吧！** 🌟