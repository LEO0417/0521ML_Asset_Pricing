#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_api.py
FastAPI 服务测试脚本
"""
import requests
import json
import numpy as np

def test_fastapi_service():
    """测试FastAPI服务"""
    print("=== Lesson 07 - FastAPI 部署测试 ===")
    
    # 服务地址
    base_url = "http://localhost:8000"
    
    try:
        # 1. 测试根路径
        print("🔍 测试根路径...")
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ 根路径响应正常:", response.json())
        else:
            print("❌ 根路径测试失败")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ 无法连接到服务: {e}")
        print("💡 请先启动 FastAPI 服务:")
        print("   cd lessons/lesson07_deployment/app")
        print("   uvicorn main:app --reload")
        return False
    
    try:
        # 2. 测试预测接口
        print("\n🔍 测试预测接口...")
        
        # 生成模拟输入数据 (20天 x 6特征)
        np.random.seed(42)
        mock_sequence = np.random.randn(20, 6).tolist()
        
        test_data = {
            "sequence": mock_sequence
        }
        
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 预测接口响应正常")
            print(f"   预测结果: {result}")
            return True
        else:
            print(f"❌ 预测接口测试失败: {response.status_code}")
            print(f"   错误信息: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 预测接口请求失败: {e}")
        return False

def test_with_invalid_data():
    """测试无效数据处理"""
    print("\n🔍 测试错误处理...")
    
    base_url = "http://localhost:8000"
    
    # 测试无效的输入格式
    invalid_data = {
        "sequence": [[1, 2, 3]]  # 不足6个特征
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=invalid_data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        if response.status_code == 422:  # 验证错误
            print("✅ 错误处理正常，返回验证错误")
        else:
            print(f"⚠️  意外的响应码: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 错误处理测试失败: {e}")

def main():
    """主函数"""
    success = test_fastapi_service()
    
    if success:
        test_with_invalid_data()
        print("\n✅ FastAPI 服务测试完成!")
    else:
        print("\n❌ 服务未启动或有问题")
        print("\n📋 启动服务的命令:")
        print("1. cd lessons/lesson07_deployment/app")
        print("2. uvicorn main:app --reload --host 0.0.0.0 --port 8000")
        print("3. 在浏览器访问: http://localhost:8000/docs")

if __name__ == "__main__":
    main() 