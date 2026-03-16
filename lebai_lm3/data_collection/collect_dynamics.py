# data_collection/collect_dynamics.py
"""
采集动力学数据：连续轨迹 + 关节力矩
用于阶段2的训练
"""

import lebai
import numpy as np
import time
import csv
from datetime import datetime

class DynamicsDataCollector:
    def __init__(self, robot_ip="192.168.1.100"):
        self.robot = lebai.LebaiRobot(robot_ip)
        
    def execute_sine_trajectory(self, joint_idx, freq, amp, duration):
        """
        执行单关节正弦轨迹
        joint_idx: 关节索引 (0-5)
        freq: 频率 Hz
        amp: 幅度 rad
        duration: 持续时间 s
        """
        dt = 0.01  # 采样间隔 10ms
        steps = int(duration / dt)
        
        # 记录数据
        data = []
        
        start_time = time.time()
        
        for i in range(steps):
            t = i * dt
            
            # 计算目标位置
            target_angle = amp * np.sin(2 * np.pi * freq * t)
            
            # 获取当前关节位置（作为初始位置）
            current_joints = self.robot.get_actual_joint_positions()
            
            # 只修改目标关节
            target_joints = current_joints.copy()
            target_joints[joint_idx] = target_angle
            
            # 发送运动命令（速度模式）
            self.robot.movej(
                target_joints,
                speed=0.5,
                wait=False  # 不等待，连续运动
            )
            
            # 读取当前状态
            actual_joints = self.robot.get_actual_joint_positions()
            actual_torques = self.robot.get_actual_joint_torques()  # 如果支持
            actual_tcp = self.robot.get_actual_tcp_pose()
            
            # 记录
            data.append({
                'timestamp': t,
                'cmd_j1': target_joints[0], 'cmd_j2': target_joints[1], 
                'cmd_j3': target_joints[2], 'cmd_j4': target_joints[3],
                'cmd_j5': target_joints[4], 'cmd_j6': target_joints[5],
                'act_j1': actual_joints[0], 'act_j2': actual_joints[1],
                'act_j3': actual_joints[2], 'act_j4': actual_joints[3],
                'act_j5': actual_joints[4], 'act_j6': actual_joints[5],
                'tau1': actual_torques[0] if actual_torques else 0,
                'tau2': actual_torques[1] if actual_torques else 0,
                'tau3': actual_torques[2] if actual_torques else 0,
                'tau4': actual_torques[3] if actual_torques else 0,
                'tau5': actual_torques[4] if actual_torques else 0,
                'tau6': actual_torques[5] if actual_torques else 0,
                'x': actual_tcp[0], 'y': actual_tcp[1], 'z': actual_tcp[2]
            })
            
            time.sleep(dt)
        
        return data
    
    def collect_dynamics_data(self):
        """
        采集完整的动力学数据集
        """
        all_data = []
        
        # 不同频率和幅度组合
        frequencies = [0.2, 0.5, 1.0, 2.0]
        amplitudes = [0.1, 0.2, 0.3, 0.5]
        
        for joint_idx in range(6):
            for freq in frequencies:
                for amp in amplitudes:
                    print(f"关节 {joint_idx+1}: freq={freq}Hz, amp={amp}rad")
                    
                    data = self.execute_sine_trajectory(
                        joint_idx, freq, amp, duration=5.0
                    )
                    all_data.extend(data)
        
        # 保存数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/real/dynamics_data_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_data[0].keys())
            writer.writeheader()
            writer.writerows(all_data)
        
        print(f"✅ 动力学数据已保存: {filename}")
        print(f"   共 {len(all_data)} 个样本点")