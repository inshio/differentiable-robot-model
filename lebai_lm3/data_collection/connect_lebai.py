# data_collection/connect_lebai.py
"""
连接乐白LM3机器人
机器臂默认IP: 192.168.0.50
机器人192.168.0.100
"""

import time
from lebai import LebaiRobot, CartesianPose, JointPose

class LebaiLM3:
    def __init__(self, robot_ip="192.168.0.50"):
        """
        连接乐白机器人
        """
        self.robot_ip = robot_ip
        self.robot = None
        
        
    def connect(self):
        """建立连接"""
        try:
            self.robot = LebaiRobot(self.robot_ip)

            # # 1. 先上电（关键步骤！）
            # print("⚡ 上电中...")
            # self.robot.start_sys()
            # print("✅ 上电成功！")


            print(f"✅ 成功连接到机器人 {self.robot_ip}")
            return True
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    def get_joint_positions(self):
        """读取当前关节角度"""
        # 根据乐白SDK文档，返回 [j1,j2,j3,j4,j5,j6] 弧度
        return self.robot.get_actual_joint_positions()
    
    def get_tcp_pose(self):
        """读取当前末端位姿"""
        # 返回 [x, y, z, rx, ry, rz] 其中rx,ry,rz是Z-Y-X欧拉角
        return self.robot.get_actual_tcp_pose()
    
    def get_joint_torques(self):
        """读取当前关节力矩（如果支持）"""
        # 部分SDK版本支持，请查阅具体API
        try:
            return self.robot.get_actual_joint_torques()
        except:
            print("⚠️ 关节力矩读取失败，可能当前SDK版本不支持")
            return None
        
    def get_robot_info(self):
        """获取机器人基本信息"""
        res = self.robot.get_robot_data()
        acc = res.actual_acc
        vel = res.actual_vel
        joint = res.actual_joint
        pose = res.actual_pose
        torques = res.actual_torque
        info = {
            "acc": acc,
            "vel": vel,
            "joint": joint,
            "pose": pose,
            "torques": torques
        }
        return info

# 测试连接
if __name__ == "__main__":
    robot = LebaiLM3("192.168.0.50")  # 改为你的机器人IP
    if robot.connect():
        # 读取一次数据测试
        joints = robot.get_joint_positions()
        tcp = robot.get_tcp_pose()
        torques = robot.get_joint_torques()
        info = robot.get_robot_info()
        print(f"机器人信息: {info}")
        print(f"关节角度: {joints}")
        print(f"末端位姿: {tcp}")
        print(f"关节力矩: {torques}")