import pickle
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

def test_and_fix_scaler():
    print("测试和修复scaler文件...")
    
    # 尝试不同的方法加载scaler
    methods = [
        ('pickle.load', lambda: pickle.load(open('scaler.pkl', 'rb'))),
        ('joblib.load', lambda: joblib.load('scaler.pkl')),
        ('pickle with protocol 2', lambda: pickle.load(open('scaler.pkl', 'rb'), encoding='latin1')),
    ]
    
    scaler = None
    working_method = None
    
    for method_name, load_func in methods:
        try:
            print(f"尝试使用 {method_name}...")
            scaler = load_func()
            working_method = method_name
            print(f"✓ {method_name} 成功加载scaler")
            break
        except Exception as e:
            print(f"✗ {method_name} 失败: {e}")
    
    if scaler is None:
        print("所有方法都失败，尝试从源目录重新复制...")
        return False
    
    # 测试scaler功能
    try:
        # 创建测试数据 (10个特征)
        test_data = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
        scaled_data = scaler.transform(test_data)
        print(f"✓ scaler变换测试成功")
        print(f"原始数据: {test_data[0][:3]}...")
        print(f"缩放后数据: {scaled_data[0][:3]}...")
        
        # 如果加载成功但方法不是标准pickle，重新保存
        if working_method != 'pickle.load':
            print("重新保存scaler为标准pickle格式...")
            with open('scaler_fixed.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            print("✓ 已保存为 scaler_fixed.pkl")
        
        return True
        
    except Exception as e:
        print(f"✗ scaler功能测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_and_fix_scaler()
    if success:
        print("\n🎉 scaler文件测试/修复成功！")
    else:
        print("\n❌ scaler文件无法修复")