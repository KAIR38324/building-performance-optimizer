import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np

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
        x = self.output_layer(x)
        return x

class CEUI_Model(nn.Module):
    def __init__(self, input_features=10):
        super().__init__()
        self.layer1 = nn.Linear(input_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)
        self.layer3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.output_layer = nn.Linear(256, 1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.dropout1(torch.relu(self.bn1(self.layer1(x))))
        x = self.dropout2(torch.relu(self.bn2(self.layer2(x))))
        x = torch.relu(self.bn3(self.layer3(x)))
        x = self.output_layer(x)
        return x

def test_models():
    print("测试替换后的模型文件...")
    
    # 测试模型文件是否存在且可加载
    loaded_models = {}
    
    try:
        # 加载UDI模型
        udi_model = UDI_DGI_Model(input_features=10)
        udi_model.load_state_dict(torch.load('UDI_trained_model.pth', map_location='cpu'))
        udi_model.eval()
        loaded_models['UDI'] = udi_model
        print("✓ UDI 模型加载成功")
        
        # 加载DGI模型
        dgi_model = UDI_DGI_Model(input_features=10)
        dgi_model.load_state_dict(torch.load('DGI_trained_model.pth', map_location='cpu'))
        dgi_model.eval()
        loaded_models['DGI'] = dgi_model
        print("✓ DGI 模型加载成功")
        
        # 加载CEUI模型
        ceui_model = CEUI_Model(input_features=10)
        ceui_model.load_state_dict(torch.load('CEUI_trained_model.pth', map_location='cpu'))
        ceui_model.eval()
        loaded_models['CEUI'] = ceui_model
        print("✓ CEUI 模型加载成功")
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return False
    
    # 测试scaler文件
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✓ scaler.pkl 加载成功")
    except Exception as e:
        print(f"✗ scaler.pkl 加载失败: {e}")
        return False
    
    # 测试帕累托解集文件
    try:
        pareto_data = pd.read_csv('帕累托解集.csv')
        print(f"✓ 帕累托解集.csv 加载成功，包含 {len(pareto_data)} 行数据")
    except Exception as e:
        print(f"✗ 帕累托解集.csv 加载失败: {e}")
        return False
    
    # 简单的预测测试
    try:
        # 创建测试输入数据 (10个特征)
        test_input = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
        test_input_scaled = scaler.transform(test_input)
        test_tensor = torch.FloatTensor(test_input_scaled)
        
        predictions = {}
        for name, model in loaded_models.items():
            model.eval()
            with torch.no_grad():
                pred = model(test_tensor)
                predictions[name] = pred.item()
        
        print("\n预测测试结果:")
        for name, pred in predictions.items():
            print(f"{name}: {pred:.4f}")
        
        # 检查预测值是否在合理范围内
        if 0 <= predictions['UDI'] <= 1:
            print("✓ UDI 预测值在合理范围内 (0-1)")
        else:
            print(f"⚠ UDI 预测值可能异常: {predictions['UDI']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 预测测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_models()
    if success:
        print("\n🎉 所有模型文件替换成功，预测功能正常！")
    else:
        print("\n❌ 模型替换或测试过程中出现问题")