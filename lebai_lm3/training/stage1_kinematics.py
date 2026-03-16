#!/usr/bin/env python
# lebai_lm3/training/stage1_kinematics.py

import os
import sys
import torch
import numpy as np

# 添加父目录到路径，以便导入 differentiable_robot_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from differentiable_robot_model.robot_model import DifferentiableRobotModel
from differentiable_robot_model.rigid_body_params import UnconstrainedTensor

def main():
    # 你的代码...
    print("开始乐白LM3运动学参数学习")
    
    # 获取URDF路径
    current_dir = os.path.dirname(__file__)
    urdf_path = os.path.join(current_dir, "../config/lebai_lm3.urdf")
    
    # 创建模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DifferentiableRobotModel(urdf_path, "lebai_lm3", device=device)
    
    # 设置可学习参数
    link_names = ["lebai_link_1", "lebai_link_2", "lebai_link_3", "lebai_link_4", "lebai_link_5", "lebai_link_6"]
    for link_name in link_names:
        try:
            model.make_link_param_learnable(
                link_name, "trans", UnconstrainedTensor(dim1=1, dim2=3)
            )
            print(f"已设置 {link_name} 的平移参数为可学习")
        except Exception as e:
            print(f"跳过 {link_name}: {e}")
    
    print("模型创建成功！")
    print(f"可学习参数数量: {len(list(model.parameters()))}")

if __name__ == "__main__":
    main()