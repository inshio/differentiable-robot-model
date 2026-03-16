#!/usr/bin/env python
# lebai_lm3/training/stage1_kinematics_train.py

'''没用了，我有真实机器人数据了，不用模拟生成了'''

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from differentiable_robot_model.robot_model import DifferentiableRobotModel
from differentiable_robot_model.rigid_body_params import UnconstrainedTensor

def generate_training_data(model, n_samples=500):
    """
    生成训练数据：随机关节角 + 对应的末端位置
    这里用一个"假想"的真实模型来生成数据
    """
    device = next(model.parameters()).device
    
    # 生成随机关节角 [-pi, pi]
    q = torch.rand(n_samples, 6, device=device) * 2 * np.pi - np.pi
    
    # 用当前模型计算末端位置（假装这是"真实"数据）
    # 在实际应用中，这里应该用真实机器人采集的数据
    ee_pos = []
    for i in range(n_samples):
        pos, _ = model.compute_forward_kinematics(
            q=q[i:i+1], 
            link_name="lebai_link_6"  # 注意这里要用正确的名称
        )
        ee_pos.append(pos)
    
    ee_pos = torch.cat(ee_pos, dim=0)
    
    return q, ee_pos

def train_kinematics():
    print("开始乐白LM3运动学参数学习")
    
    # 配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    urdf_path = os.path.join(os.path.dirname(__file__), "../config/lebai_lm3.urdf")
    n_epochs = 2000
    n_samples = 200
    lr = 1e-3
    
    # 1. 创建两个模型：真实模型（固定）和可学习模型
    print("\n1. 创建模型...")
    
    # 真实模型（作为数据生成器，参数固定）
    gt_model = DifferentiableRobotModel(urdf_path, "lebai_lm3_gt", device=device)
    
    # 可学习模型（我们要训练的）
    learnable_model = DifferentiableRobotModel(urdf_path, "lebai_lm3_learn", device=device)
    
    # 设置可学习参数
    link_names = [f"lebai_link_{i}" for i in range(1, 7)]
    for link_name in link_names:
        learnable_model.make_link_param_learnable(
            link_name, "trans", UnconstrainedTensor(dim1=1, dim2=3)
        )
        print(f"  ✅ 设置 {link_name} 为可学习")
    
    # 给可学习模型添加一些初始噪声（模拟URDF不精确）
    with torch.no_grad():
        for param in learnable_model.parameters():
            param.add_(torch.randn_like(param) * 0.01)
    
    # 2. 生成训练数据
    print("\n2. 生成训练数据...")
    q, gt_ee_pos = generate_training_data(gt_model, n_samples)
    
    # 划分训练/验证集
    n_train = int(0.8 * n_samples)
    train_q, val_q = q[:n_train], q[n_train:]
    train_gt, val_gt = gt_ee_pos[:n_train], gt_ee_pos[n_train:]
    
    # 3. 设置优化器
    optimizer = torch.optim.Adam(learnable_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200
    )
    loss_fn = torch.nn.MSELoss()
    
    # 记录训练历史
    history = {"train_loss": [], "val_loss": []}
    
    # 4. 训练循环
    print("\n3. 开始训练...")
    for epoch in range(n_epochs):
        # 训练
        learnable_model.train()
        optimizer.zero_grad()
        
        ee_pos_pred, _ = learnable_model.compute_forward_kinematics(
            q=train_q, 
            link_name="lebai_link_6"
        )
        train_loss = loss_fn(ee_pos_pred, train_gt)
        
        train_loss.backward()
        optimizer.step()
        
        # 验证
        learnable_model.eval()
        with torch.no_grad():
            ee_pos_pred_val, _ = learnable_model.compute_forward_kinematics(
                q=val_q, 
                link_name="lebai_link_6"
            )
            val_loss = loss_fn(ee_pos_pred_val, val_gt)
        
        scheduler.step(val_loss)
        
        # 记录
        history["train_loss"].append(train_loss.item())
        history["val_loss"].append(val_loss.item())
        
        # 每200轮打印一次
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d}: train_loss = {train_loss:.2e}, val_loss = {val_loss:.2e}")
    
    # 5. 结果分析
    print("\n4. 训练完成！")
    print(f"最终训练损失: {history['train_loss'][-1]:.2e}")
    print(f"最终验证损失: {history['val_loss'][-1]:.2e}")
    
    # 6. 保存模型
    torch.save(learnable_model.state_dict(), 
               os.path.join(os.path.dirname(__file__), "../models/kinematics_model.pth"))
    print("✅ 模型已保存到 models/kinematics_model.pth")
    
    # 7. 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("运动学参数学习收敛曲线")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), "../training_curve.png"))
    plt.show()
    
    return learnable_model, history

if __name__ == "__main__":
    model, history = train_kinematics()