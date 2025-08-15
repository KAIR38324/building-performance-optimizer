# Streamlit Cloud 部署最佳实践指南

## 🚨 ModuleNotFoundError: torch 问题解决方案

### 问题描述
在Streamlit Cloud部署时遇到 `ModuleNotFoundError: No module named 'torch'` 错误，这是因为:
1. Streamlit Cloud默认不支持GPU版本的PyTorch
2. 需要明确指定CPU版本的torch
3. 依赖版本兼容性问题

### ✅ 解决方案

#### 方案1: 使用修复后的 requirements_deploy.txt
```txt
streamlit>=1.28.0
--find-links https://download.pytorch.org/whl/torch_stable.html
torch==2.0.1+cpu
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
google-generativeai>=0.3.0
openai>=0.28.0
requests>=2.31.0
```

#### 方案2: 使用专用的 requirements_streamlit_cloud.txt
更完整的依赖配置，包含版本锁定和兼容性保证。

## 🔧 部署步骤

### 1. 更新GitHub仓库
```bash
# 提交修复后的requirements文件
git add requirements_deploy.txt requirements_streamlit_cloud.txt
git commit -m "Fix torch dependency for Streamlit Cloud deployment"
git push origin main
```

### 2. 在Streamlit Cloud中重新部署
1. 访问 [Streamlit Cloud](https://share.streamlit.io/)
2. 找到您的应用
3. 点击 "Reboot app" 或 "Delete app" 后重新创建
4. 确保使用正确的requirements文件路径

### 3. 配置文件选择
在Streamlit Cloud部署设置中:
- **Main file path**: `app_deploy.py`
- **Requirements file**: `requirements_deploy.txt` 或 `requirements_streamlit_cloud.txt`
- **Python version**: 3.9 或 3.10 (推荐)

## 🎯 关键配置说明

### PyTorch CPU版本
```txt
--find-links https://download.pytorch.org/whl/torch_stable.html
torch==2.0.1+cpu
```
- `--find-links`: 指定PyTorch官方CPU版本下载源
- `+cpu`: 明确指定CPU版本，避免GPU依赖

### 版本锁定策略
- **严格版本**: `torch==2.0.1+cpu` (推荐用于生产)
- **兼容版本**: `torch>=2.0.0,<2.1.0` (允许小版本更新)

### 必要依赖
- `joblib`: 用于加载 `scaler.pkl` 文件
- `numpy`: 数据处理基础库
- `pandas`: 数据框操作
- `scikit-learn`: 机器学习工具

## 🔍 故障排除

### 常见错误及解决方案

#### 1. "No module named 'torch'"
**解决**: 使用修复后的requirements文件，确保包含CPU版本的torch

#### 2. "Could not find a version that satisfies the requirement torch"
**解决**: 添加 `--find-links` 指向PyTorch官方源

#### 3. "No module named 'joblib'"
**解决**: 在requirements中添加 `joblib>=1.3.0`

#### 4. 模型文件加载失败
**解决**: 确保所有 `.pth` 和 `.pkl` 文件都已提交到GitHub

### 部署日志检查
在Streamlit Cloud中:
1. 点击应用右下角的 "Manage app"
2. 查看 "Logs" 标签页
3. 检查具体的错误信息

## 📋 部署检查清单

- [ ] ✅ 使用修复后的 `requirements_deploy.txt`
- [ ] ✅ 确保所有模型文件 (`.pth`, `.pkl`) 已上传
- [ ] ✅ 确保数据文件 (`帕累托解集.csv`) 已上传
- [ ] ✅ 配置环境变量 (API密钥)
- [ ] ✅ 选择正确的Python版本 (3.9/3.10)
- [ ] ✅ 指定正确的主文件路径 (`app_deploy.py`)

## 🚀 性能优化建议

### 1. 模型加载优化
- 使用 `@st.cache_resource` 缓存模型加载
- 在云端强制使用CPU: `device = torch.device('cpu')`

### 2. 内存管理
- 模型推理时使用 `torch.no_grad()`
- 及时释放不需要的张量

### 3. 启动时间优化
- 将大文件放在缓存函数中
- 避免在主程序中进行重复计算

## 📞 技术支持

如果仍然遇到问题:
1. 检查Streamlit Cloud的系统状态
2. 确认GitHub仓库的文件完整性
3. 尝试使用 `requirements_streamlit_cloud.txt` 替代默认配置
4. 联系Streamlit社区获取帮助

---

**最后更新**: 2024年1月
**适用版本**: Streamlit Cloud, PyTorch 2.0+, Python 3.9+