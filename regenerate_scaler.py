#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新生成scaler.pkl文件
基于帕累托解集数据重新训练StandardScaler
"""

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

def regenerate_scaler():
    """重新生成scaler文件"""
    try:
        # 1. 加载帕累托解集数据
        print("📊 加载帕累托解集数据...")
        df = pd.read_csv('帕累托解集.csv', encoding='utf-8')
        print(f"✅ 数据加载成功，包含 {len(df)} 行数据")
        
        # 2. 定义特征列（根据实际CSV文件的列名）
        feature_names = [
            '开间', '进深', '层高', '北侧窗台高', 
            '北侧窗高', '北侧窗墙比', '南侧窗墙比',
            '南侧窗高', '南侧窗台高', '南侧窗间距'
        ]
        
        # 3. 检查特征列是否存在
        missing_cols = [col for col in feature_names if col not in df.columns]
        if missing_cols:
            print(f"❌ 缺少特征列: {missing_cols}")
            print(f"📋 可用列: {list(df.columns)}")
            return False
        
        # 4. 提取特征数据
        X = df[feature_names].values
        print(f"📈 特征数据形状: {X.shape}")
        
        # 5. 创建并训练StandardScaler
        print("🔧 训练StandardScaler...")
        scaler = StandardScaler()
        scaler.fit(X)
        
        # 6. 保存scaler
        with open('scaler_new.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("✅ 新的scaler已保存为 scaler_new.pkl")
        
        # 7. 测试新的scaler
        print("🧪 测试新的scaler...")
        with open('scaler_new.pkl', 'rb') as f:
            test_scaler = pickle.load(f)
        
        # 测试转换
        test_data = X[:5]  # 使用前5行数据测试
        scaled_data = test_scaler.transform(test_data)
        print(f"📊 测试转换成功")
        print(f"原始数据均值: {np.mean(test_data, axis=0)[:3]}...")
        print(f"标准化后均值: {np.mean(scaled_data, axis=0)[:3]}...")
        print(f"标准化后标准差: {np.std(scaled_data, axis=0)[:3]}...")
        
        # 8. 显示scaler信息
        print(f"\n📋 Scaler信息:")
        print(f"特征数量: {test_scaler.n_features_in_}")
        print(f"样本数量: {test_scaler.n_samples_seen_}")
        print(f"均值 (前3个): {test_scaler.mean_[:3]}")
        print(f"标准差 (前3个): {test_scaler.scale_[:3]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 重新生成scaler失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def replace_old_scaler():
    """替换旧的scaler文件"""
    try:
        import os
        import shutil
        
        # 备份旧文件
        if os.path.exists('scaler.pkl'):
            shutil.move('scaler.pkl', 'scaler_old_backup.pkl')
            print("📦 旧scaler已备份为 scaler_old_backup.pkl")
        
        # 替换为新文件
        shutil.move('scaler_new.pkl', 'scaler.pkl')
        print("✅ 新scaler已替换为 scaler.pkl")
        
        return True
    except Exception as e:
        print(f"❌ 替换scaler文件失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始重新生成scaler.pkl文件...")
    
    # 重新生成scaler
    if regenerate_scaler():
        print("\n🔄 是否要替换旧的scaler.pkl文件？")
        print("新文件已保存为 scaler_new.pkl")
        
        # 自动替换
        if replace_old_scaler():
            print("\n✅ scaler.pkl文件已成功更新！")
            print("💡 现在可以重新运行测试脚本验证")
        else:
            print("\n⚠️  替换失败，请手动将 scaler_new.pkl 重命名为 scaler.pkl")
    else:
        print("\n❌ 重新生成失败，请检查数据文件")

if __name__ == "__main__":
    main()