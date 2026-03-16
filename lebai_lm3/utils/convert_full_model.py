# lebo_lm3/utils/convert_full_model.py
"""
从完整的乐巴功能包生成URDF文件
自动处理所有依赖和mesh路径
"""

import os
import re
import sys
from pathlib import Path

class LebaiModelConverter:
    def __init__(self, base_dir):
        """
        base_dir: 你的 lebo_lm3 目录路径
        """
        self.base_dir = Path(base_dir)
        self.config_dir = self.base_dir / "config"
        
    def find_ros_package_path(self, package_name):
        """
        在config目录中查找包的路径
        """
        # 首先检查config目录下是否有同名文件夹
        pkg_path = self.config_dir / package_name
        if pkg_path.exists():
            return str(pkg_path)
        
        # 如果没有，尝试从原始位置查找
        home = Path.home()
        possible_paths = [
            home / "ZJJ/differentiable-robot-model/lebai-ros-sdk-noetic-dev" / package_name
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def resolve_package_paths(self, xacro_content):
        """
        将xacro中的 $(find package_name) 替换为实际路径
        """
        def replace_package_path(match):
            package_name = match.group(1)
            pkg_path = self.find_ros_package_path(package_name)
            if pkg_path:
                return pkg_path
            else:
                print(f"⚠️ 警告: 找不到包 {package_name}，使用相对路径")
                return f"../{package_name}"
        
        # 替换 $(find package_name)
        pattern = r'\$\(find ([^)]+)\)'
        return re.sub(pattern, replace_package_path, xacro_content)
    
    def convert_to_urdf(self, xacro_file, output_file=None):
        """
        将xacro文件转换为URDF
        """
        from xacrodoc import XacroDoc
        
        xacro_path = self.config_dir / xacro_file
        if not xacro_path.exists():
            print(f"❌ 找不到文件: {xacro_path}")
            return None
        
        # 读取并预处理xacro内容
        with open(xacro_path, 'r') as f:
            content = f.read()
        
        # 解析 package:// 路径
        content = self.resolve_package_paths(content)
        
        # 写入临时文件
        temp_xacro = self.config_dir / "temp_preprocessed.xacro"
        with open(temp_xacro, 'w') as f:
            f.write(content)
        
        try:
            # 转换
            doc = XacroDoc.from_file(str(temp_xacro))
            
            if output_file is None:
                output_file = xacro_file.replace('.xacro', '.urdf')
            
            output_path = self.config_dir / output_file
            doc.to_urdf_file(str(output_path))
            
            print(f"✅ 转换成功: {output_path}")
            
            # 可选：修复mesh路径
            self.fix_mesh_paths(output_path)
            
            return str(output_path)
            
        except Exception as e:
            print(f"❌ 转换失败: {e}")
            return None
        finally:
            # 清理临时文件
            if temp_xacro.exists():
                temp_xacro.unlink()
    
    def fix_mesh_paths(self, urdf_path):
        """
        修复URDF中的mesh路径，使其指向正确的位置
        """
        with open(urdf_path, 'r') as f:
            content = f.read()
        
        # 将 package:// 路径改为相对路径
        def replace_mesh_path(match):
            package_path = match.group(1)
            # 提取相对路径部分
            if 'lebai_lm3_support' in package_path:
                mesh_rel_path = package_path.split('lebai_lm3_support/')[-1]
                return f"package://{mesh_rel_path}"
            return f"file://{package_path}"
        
        pattern = r'package://([^"]+)'
        content = re.sub(pattern, replace_mesh_path, content)
        
        # 写回文件
        with open(urdf_path, 'w') as f:
            f.write(content)
        
        print(f"✅ 已修复mesh路径")

def main():
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent.parent
    converter = LebaiModelConverter(current_dir)
    
    print("🔧 开始转换乐白LM3模型...")
    
    # 转换纯机械臂版本
    print("\n1. 转换纯机械臂模型...")
    urdf1 = converter.convert_to_urdf(
        xacro_file="lebai_lm3.xacro",
        output_file="lebai_lm3.urdf"
    )
    
    # 转换带夹具版本
    print("\n2. 转换带夹具模型...")
    urdf2 = converter.convert_to_urdf(
        xacro_file="lm3_with_gripper.xacro",
        output_file="lebai_lm3_with_gripper.urdf"
    )
    
    print("\n📋 转换结果:")
    if urdf1:
        print(f"  纯机械臂: {urdf1}")
    if urdf2:
        print(f"  带夹具: {urdf2}")

if __name__ == "__main__":
    main()