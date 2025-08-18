
# 修复后的UDI预测函数
# 缩放因子: 1.119380

def predict_udi_fixed(input_data, scaler, model):
    """修复后的UDI预测函数"""
    # 标准化输入
    scaled_data = scaler.transform(input_data)
    
    # 模型预测
    with torch.no_grad():
        input_tensor = torch.FloatTensor(scaled_data)
        raw_prediction = model(input_tensor).item()
    
    # 应用缩放因子修复
    fixed_prediction = raw_prediction * 1.119380
    
    return fixed_prediction
