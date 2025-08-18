#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试预测功能，验证UDI值是否正常
"""

import pickle
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

# 定义模型类（与app.py中一致）
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

def test_prediction():
    """测试预测功能"""
    try:
        # 1. 加载scaler
        print("📊 加载scaler...")
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print(f"✅ scaler加载成功，特征数量: {scaler.n_features_in_}")
        
        # 2. 加载模型
        print("🤖 加载UDI模型...")
        udi_model = UDI_DGI_Model(input_features=10)
        device = torch.device('cpu')
        udi_model.load_state_dict(torch.load('UDI_trained_model.pth', map_location=device))
        udi_model.eval()
        print("✅ UDI模型加载成功")
        
        # 3. 准备测试数据（使用合理的建筑参数）
        test_data = np.array([[
            8.0,   # 开间
            8.0,   # 进深  
            3.2,   # 层高
            0.9,   # 北侧窗台高
            1.5,   # 北侧窗高
            0.3,   # 北侧窗墙比
            0.4,   # 南侧窗墙比
            1.8,   # 南侧窗高
            0.9,   # 南侧窗台高
            1.0    # 南侧窗间距
        ]])
        
        print(f"🏠 测试数据: {test_data[0]}")
        
        # 4. 数据标准化
        print("🔄 数据标准化...")
        scaled_data = scaler.transform(test_data)
        print(f"📊 标准化后数据: {scaled_data[0][:3]}...")
        
        # 5. 预测
        print("🔮 进行UDI预测...")
        with torch.no_grad():
            input_tensor = torch.FloatTensor(scaled_data)
            udi_pred = udi_model(input_tensor).item()
        
        # 6. 显示结果
        print(f"\n📈 预测结果:")
        print(f"UDI值: {udi_pred:.4f}")
        print(f"UDI百分比: {udi_pred * 100:.2f}%")
        
        # 7. 判断结果是否合理
        if 0 <= udi_pred <= 1:
            print("✅ UDI预测值在合理范围内 (0-1)")
        else:
            print(f"❌ UDI预测值异常: {udi_pred}")
            
        if udi_pred * 100 > 100:
            print("⚠️  UDI百分比超过100%，可能存在问题")
        elif udi_pred * 100 > 80:
            print("⚠️  UDI百分比过高，请检查输入参数")
        else:
            print("✅ UDI百分比在正常范围内")
            
        return udi_pred
        
    except Exception as e:
        print(f"❌ 预测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_multiple_cases():
    """测试多个案例"""
    print("\n🧪 测试多个案例...")
    
    test_cases = [
        [8.0, 8.0, 3.2, 0.9, 1.5, 0.3, 0.4, 1.8, 0.9, 1.0],  # 标准案例
        [6.0, 6.0, 2.8, 0.8, 1.2, 0.2, 0.3, 1.5, 0.8, 0.8],  # 小户型
        [10.0, 10.0, 3.6, 1.0, 2.0, 0.4, 0.5, 2.2, 1.0, 1.2], # 大户型
    ]
    
    case_names = ["标准案例", "小户型", "大户型"]
    
    try:
        # 加载模型和scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        udi_model = UDI_DGI_Model(input_features=10)
        device = torch.device('cpu')
        udi_model.load_state_dict(torch.load('UDI_trained_model.pth', map_location=device))
        udi_model.eval()
        
        for i, (case, name) in enumerate(zip(test_cases, case_names)):
            print(f"\n📋 {name}:")
            test_data = np.array([case])
            scaled_data = scaler.transform(test_data)
            
            with torch.no_grad():
                input_tensor = torch.FloatTensor(scaled_data)
                udi_pred = udi_model(input_tensor).item()
            
            print(f"  UDI: {udi_pred:.4f} ({udi_pred * 100:.2f}%)")
            
            if udi_pred > 8:  # 检查是否还有异常高的值
                print(f"  ⚠️  异常高值detected!")
            elif 0 <= udi_pred <= 1:
                print(f"  ✅ 正常范围")
            else:
                print(f"  ❌ 异常值")
                
    except Exception as e:
        print(f"❌ 多案例测试失败: {e}")

if __name__ == "__main__":
    print("🚀 开始预测测试...")
    result = test_prediction()
    
    if result is not None:
        test_multiple_cases()
        print("\n✅ 预测测试完成")
    else:
        print("\n❌ 预测测试失败")