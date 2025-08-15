#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
部署版本测试脚本
测试Google Gemini API集成和模型加载
"""

import os
import sys
import torch
import pandas as pd
import pickle
import google.generativeai as genai
from pathlib import Path

def test_model_loading():
    """测试模型文件加载"""
    print("\n=== 测试模型文件加载 ===")
    
    model_files = [
        'UDI_trained_model.pth',
        'DGI_trained_model.pth', 
        'CEUI_trained_model.pth'
    ]
    
    for model_file in model_files:
        try:
            if os.path.exists(model_file):
                model = torch.load(model_file, map_location='cpu')
                print(f"✅ {model_file} 加载成功")
            else:
                print(f"❌ {model_file} 文件不存在")
        except Exception as e:
            print(f"❌ {model_file} 加载失败: {e}")

def test_data_loading():
    """测试数据文件加载"""
    print("\n=== 测试数据文件加载 ===")
    
    # 测试scaler.pkl
    try:
        if os.path.exists('scaler.pkl'):
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            print("✅ scaler.pkl 加载成功")
        else:
            print("❌ scaler.pkl 文件不存在")
    except Exception as e:
        print(f"❌ scaler.pkl 加载失败: {e}")
    
    # 测试帕累托解集.csv
    try:
        if os.path.exists('帕累托解集.csv'):
            df = pd.read_csv('帕累托解集.csv', encoding='utf-8')
            print(f"✅ 帕累托解集.csv 加载成功，包含 {len(df)} 行数据")
        else:
            print("❌ 帕累托解集.csv 文件不存在")
    except Exception as e:
        print(f"❌ 帕累托解集.csv 加载失败: {e}")

def test_gemini_api():
    """测试Google Gemini API连接"""
    print("\n=== 测试Google Gemini API ===")
    
    # 检查API密钥
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ 未找到GOOGLE_API_KEY环境变量")
        print("💡 请设置环境变量: set GOOGLE_API_KEY=your-api-key")
        return False
    
    try:
        # 配置API
        genai.configure(api_key=api_key)
        
        # 创建模型实例
        model = genai.GenerativeModel('gemini-pro')
        
        # 测试API调用
        test_prompt = "请简单回答：1+1等于几？"
        response = model.generate_content(test_prompt)
        
        if response and response.text:
            print("✅ Google Gemini API 连接成功")
            print(f"📝 测试响应: {response.text.strip()}")
            return True
        else:
            print("❌ API响应为空")
            return False
            
    except Exception as e:
        print(f"❌ Google Gemini API 测试失败: {e}")
        return False

def test_dependencies():
    """测试依赖包"""
    print("\n=== 测试依赖包 ===")
    
    dependencies = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'torch': 'torch',
        'sklearn': 'scikit-learn',
        'numpy': 'numpy',
        'google.generativeai': 'google-generativeai'
    }
    
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"❌ {package} 未安装")

def main():
    """主测试函数"""
    print("🚀 开始测试部署版本...")
    print(f"📁 当前目录: {os.getcwd()}")
    print(f"🐍 Python版本: {sys.version}")
    
    # 运行所有测试
    test_dependencies()
    test_model_loading()
    test_data_loading()
    api_success = test_gemini_api()
    
    print("\n=== 测试总结 ===")
    if api_success:
        print("✅ 所有核心功能测试通过，可以进行部署")
        print("📋 下一步: 上传到GitHub并配置Streamlit Cloud")
    else:
        print("⚠️  API测试失败，请检查Google Gemini API密钥配置")
        print("💡 获取API密钥: https://aistudio.google.com/app/apikey")
    
    print("\n🎯 部署清单:")
    print("   1. 确保所有文件已上传到GitHub")
    print("   2. 在Streamlit Cloud中配置GOOGLE_API_KEY")
    print("   3. 设置主文件为app_deploy.py")
    print("   4. 等待自动构建完成")

if __name__ == "__main__":
    main()