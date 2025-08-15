#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试scaler.pkl文件
"""

import pickle
import numpy as np

def test_scaler():
    """测试scaler文件加载"""
    try:
        # 尝试加载scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        print(f"✅ scaler.pkl 加载成功")
        print(f"📊 特征数量: {scaler.n_features_in_}")
        print(f"📈 样本数量: {scaler.n_samples_seen_}")
        
        # 测试转换功能
        test_data = np.random.rand(1, 10)  # 10个特征的随机数据
        scaled_data = scaler.transform(test_data)
        print(f"🔄 数据转换测试成功")
        print(f"原始数据形状: {test_data.shape}")
        print(f"转换后数据形状: {scaled_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ scaler.pkl 测试失败: {e}")
        return False

if __name__ == "__main__":
    test_scaler()