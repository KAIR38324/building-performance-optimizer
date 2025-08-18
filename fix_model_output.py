#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复模型输出范围问题
分析当前模型输出与实际数据范围的差异，并提供修复方案
"""

import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 定义模型类
class UDI_DGI_Model(nn.Module):
    def __init__(self, input_features=10):
        super().__init__()
        self.layer1 = nn.Linear(input_features, 256)  
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 256)
        self.output_layer = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return self.output_layer(x)

def analyze_model_output():
    """分析模型输出与实际数据的差异"""
    try:
        print("🔍 分析模型输出问题...")
        
        # 1. 加载实际数据
        df = pd.read_csv('帕累托解集.csv', encoding='utf-8')
        print(f"📊 实际UDI数据范围: {df['UDI'].min():.2f} - {df['UDI'].max():.2f}")
        print(f"📊 实际UDI均值: {df['UDI'].mean():.2f}")
        
        # 2. 加载模型和scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        udi_model = UDI_DGI_Model(input_features=10)
        device = torch.device('cpu')
        udi_model.load_state_dict(torch.load('UDI_trained_model.pth', map_location=device))
        udi_model.eval()
        
        # 3. 使用实际数据进行预测
        feature_names = [
            '开间', '进深', '层高', '北侧窗台高', 
            '北侧窗高', '北侧窗墙比', '南侧窗墙比',
            '南侧窗高', '南侧窗台高', '南侧窗间距'
        ]
        
        X = df[feature_names].values
        y_actual = df['UDI'].values
        
        # 标准化输入数据
        X_scaled = scaler.transform(X)
        
        # 模型预测
        with torch.no_grad():
            input_tensor = torch.FloatTensor(X_scaled)
            y_pred = udi_model(input_tensor).numpy().flatten()
        
        print(f"🤖 模型预测范围: {y_pred.min():.2f} - {y_pred.max():.2f}")
        print(f"🤖 模型预测均值: {y_pred.mean():.2f}")
        
        # 4. 计算缩放因子
        scale_factor = np.mean(y_actual) / np.mean(y_pred)
        print(f"📐 建议缩放因子: {scale_factor:.6f}")
        
        # 5. 测试缩放后的效果
        y_pred_scaled = y_pred * scale_factor
        print(f"✅ 缩放后预测范围: {y_pred_scaled.min():.2f} - {y_pred_scaled.max():.2f}")
        print(f"✅ 缩放后预测均值: {y_pred_scaled.mean():.2f}")
        
        # 6. 计算误差
        mae_original = np.mean(np.abs(y_pred - y_actual))
        mae_scaled = np.mean(np.abs(y_pred_scaled - y_actual))
        
        print(f"\n📊 误差分析:")
        print(f"原始预测MAE: {mae_original:.2f}")
        print(f"缩放后预测MAE: {mae_scaled:.2f}")
        print(f"误差改善: {((mae_original - mae_scaled) / mae_original * 100):.1f}%")
        
        return scale_factor
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_fixed_prediction_function():
    """创建修复后的预测函数"""
    scale_factor = analyze_model_output()
    
    if scale_factor is None:
        print("❌ 无法确定缩放因子")
        return
    
    print(f"\n🔧 创建修复后的预测函数...")
    
    # 创建修复脚本
    fix_code = f'''
# 修复后的UDI预测函数
# 缩放因子: {scale_factor:.6f}

def predict_udi_fixed(input_data, scaler, model):
    """修复后的UDI预测函数"""
    # 标准化输入
    scaled_data = scaler.transform(input_data)
    
    # 模型预测
    with torch.no_grad():
        input_tensor = torch.FloatTensor(scaled_data)
        raw_prediction = model(input_tensor).item()
    
    # 应用缩放因子修复
    fixed_prediction = raw_prediction * {scale_factor:.6f}
    
    return fixed_prediction
'''
    
    with open('udi_prediction_fix.py', 'w', encoding='utf-8') as f:
        f.write(fix_code)
    
    print("✅ 修复代码已保存到 udi_prediction_fix.py")
    print(f"💡 在app.py中使用缩放因子 {scale_factor:.6f} 来修复UDI预测")

def test_fixed_prediction():
    """测试修复后的预测"""
    scale_factor = 0.107  # 基于分析得出的大概值
    
    try:
        # 加载模型和scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        udi_model = UDI_DGI_Model(input_features=10)
        device = torch.device('cpu')
        udi_model.load_state_dict(torch.load('UDI_trained_model.pth', map_location=device))
        udi_model.eval()
        
        # 测试数据
        test_data = np.array([[
            8.0, 8.0, 3.2, 0.9, 1.5, 0.3, 0.4, 1.8, 0.9, 1.0
        ]])
        
        # 预测
        scaled_data = scaler.transform(test_data)
        with torch.no_grad():
            input_tensor = torch.FloatTensor(scaled_data)
            raw_pred = udi_model(input_tensor).item()
        
        fixed_pred = raw_pred * scale_factor
        
        print(f"\n🧪 修复测试:")
        print(f"原始预测: {raw_pred:.2f}")
        print(f"修复后预测: {fixed_pred:.2f}")
        print(f"是否在合理范围: {'✅' if 50 <= fixed_pred <= 65 else '❌'}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    print("🚀 开始分析和修复模型输出问题...")
    create_fixed_prediction_function()
    print("\n🧪 测试修复效果...")
    test_fixed_prediction()