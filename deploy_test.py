#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
部署测试脚本
用于本地测试云部署版本的应用
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """检查依赖包是否安装"""
    print("🔍 检查依赖包...")
    
    required_packages = [
        'streamlit',
        'pandas', 
        'torch',
        'scikit-learn',
        'numpy',
        'openai'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - 已安装")
        except ImportError:
            print(f"❌ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements_deploy.txt")
        return False
    
    print("\n✅ 所有依赖包已安装")
    return True

def check_files():
    """检查必要文件是否存在"""
    print("\n🔍 检查必要文件...")
    
    required_files = [
        'app_deploy.py',
        'requirements_deploy.txt',
        'UDI_trained_model.pth',
        'DGI_trained_model.pth', 
        'CEUI_trained_model.pth',
        'scaler.pkl',
        '帕累托解集.csv',
        '.streamlit/config.toml'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} - 存在")
        else:
            print(f"❌ {file_path} - 缺失")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  缺少以下文件: {', '.join(missing_files)}")
        return False
    
    print("\n✅ 所有必要文件存在")
    return True

def check_secrets():
    """检查API密钥配置"""
    print("\n🔍 检查API密钥配置...")
    
    secrets_file = '.streamlit/secrets.toml'
    example_file = '.streamlit/secrets.toml.example'
    
    if not os.path.exists(secrets_file):
        print(f"⚠️  {secrets_file} 不存在")
        if os.path.exists(example_file):
            print(f"💡 请复制 {example_file} 为 {secrets_file} 并配置您的API密钥")
        return False
    
    # 检查secrets.toml内容
    try:
        with open(secrets_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'your-openai-api-key-here' in content:
                print("⚠️  请在 secrets.toml 中配置您的实际OpenAI API密钥")
                return False
            elif 'OPENAI_API_KEY' in content:
                print("✅ API密钥配置文件存在")
                return True
            else:
                print("⚠️  secrets.toml 格式不正确")
                return False
    except Exception as e:
        print(f"❌ 读取secrets.toml失败: {e}")
        return False

def run_app():
    """运行应用"""
    print("\n🚀 启动应用...")
    print("📝 注意: 按 Ctrl+C 停止应用")
    print("🌐 应用将在 http://localhost:8501 运行")
    print("-" * 50)
    
    try:
        subprocess.run(['streamlit', 'run', 'app_deploy.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败: {e}")
    except KeyboardInterrupt:
        print("\n👋 应用已停止")

def main():
    """主函数"""
    print("🏗️  建筑性能优化系统 - 部署测试工具")
    print("=" * 50)
    
    # 检查依赖
    if not check_requirements():
        sys.exit(1)
    
    # 检查文件
    if not check_files():
        sys.exit(1)
    
    # 检查API密钥
    if not check_secrets():
        print("\n💡 提示: 您可以先运行应用测试基本功能，智能决策功能需要配置API密钥")
        response = input("\n是否继续启动应用? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print("\n✅ 所有检查通过!")
    
    # 运行应用
    run_app()

if __name__ == "__main__":
    main()