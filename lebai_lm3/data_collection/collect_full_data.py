# data_collection/collect_full_data.py
"""
完整数据集采集脚本
采集：位置、速度、加速度、力矩
使用 move_pt 进行连续轨迹运动
"""
from lebai.pb2 import robot_controller_pb2
import lebai
from lebai import JointPose
import numpy as np
import time
import csv
import os
import signal
import sys
from datetime import datetime
from typing import List, Dict, Any

# ==================== 关节安全限位 ====================
class JointSafetyChecker:
    """关节安全限位检查器（根据你之前提供的碰撞信息）"""
    
    DEG2RAD = np.pi / 180
    RAD2DEG = 180 / np.pi
    
    # 碰撞范围（单位：度）
    COLLISION_ZONES = {
        'j2': {
            'name': '关节2',
            'collision_with_base': {'min': -180, 'max': 20},  # 与车体顶盖板碰撞
            'safe_range': {'min': -180, 'max': 20}              # 安全范围
        }

    }
    
    # 整体关节运动范围（单位：度）
    JOINT_RANGES = {
        'j1': {'min': -180, 'max': 180},
        'j2': {'min': -180, 'max': 180},
        'j3': {'min': -180, 'max': 180},
        'j4': {'min': -180, 'max': 180},
        'j5': {'min': -180, 'max': 180},
        'j6': {'min': -180, 'max': 180}
    }
    
    @classmethod
    def check_safety(cls, joints_deg):
        """检查关节角度是否安全"""
        j2, j3 = joints_deg[1], joints_deg[2]
        
        # 1. 关节2的基本范围 [-180, 20]
        if j2 < -180 or j2 > 20:
            return False, f"关节2角度 {j2:.1f}° 超出安全范围 [-180, 20]"
        
        # 2. 关节3与关节2的耦合限制
        if j2 > -20:  # 关节2大于-20度时
            if j3 > 40:
                return False, f"关节2={j2:.1f}° > -20° 时，关节3={j3:.1f}° 不能超过40°"
        elif j2 == -20:  # 关节2等于-20度时
            if j3 > 90:
                return False, f"关节2={j2:.1f}° = -20° 时，关节3={j3:.1f}° 不能超过90°"       
        elif j2 == -180:  # 关节2等于-180度时
            if j3 <= -65:
                return False, f"关节2={j2:.1f}° = -180° 时，关节3={j3:.1f}° 不能超过-65°"
        
        return True, "安全"
    
    @classmethod
    def generate_safe_trajectory(cls, n_points=100, duration=10, max_vel=2.0):
        """
        生成安全的连续轨迹
        n_points: 请求的点数
        duration: 运动时间（秒）
        max_velocity: 最大允许速度（rad/s）
        返回：位置列表 [ [j1,...,j6], ... ]
        """
        trajectory = []
        dt = duration / n_points

        max_angle_change = 2.0  # rad
        
        # 计算安全点数
        safe_n_points = int((max_angle_change * duration) / max_vel)
        safe_n_points = max(5, min(safe_n_points, n_points))  # 在5到请求点数之间
        
        if safe_n_points < n_points:
            print(f"⚠️ 根据速度限制，点数从 {n_points} 调整为 {safe_n_points}")
            n_points = safe_n_points

        
        for i in range(n_points):
            t = i * dt / duration  # 归一化时间 [0,1]
            
            # 生成平滑变化的关节角度
            # 根据安全范围调整幅度
            j2_amp = 100  # 最大幅度100度（在-180~20范围内）
            j3_amp = 30   # 基础幅度
            
            # 根据时间调整，确保不会同时进入危险区域
            j2 = -80 + j2_amp * np.sin(2*np.pi * 0.2 * t)  # 让j2在-180~20之间变化
            j3 = 30 + 20 * np.sin(2*np.pi * 0.25 * t)       # 让j3在10~50之间变化
            
            joints_deg = [
                30 * np.sin(2*np.pi * 0.2 * t),      # j1
                j2,                                    # j2
                j3,                                    # j3
                40 * np.sin(2*np.pi * 0.4 * t + 1.2), # j4
                30 * np.sin(2*np.pi * 0.35 * t - 0.8),# j5
                60 * np.sin(2*np.pi * 0.5 * t + 2.1)  # j6
            ]
            
            # 安全检查
            is_safe, _ = cls.check_safety(joints_deg)
            if is_safe:
                # 转换为弧度
                joints_rad = [j * cls.DEG2RAD for j in joints_deg]
                trajectory.append(joints_rad)

         # 如果生成的轨迹为空，使用一个绝对安全的默认轨迹
        if len(trajectory) == 0:
            print("⚠️ 警告：生成的安全轨迹为空，使用默认安全轨迹")
            for i in range(n_points):
                t = i / n_points
                # 非常保守的轨迹
                joints_deg = [
                    10 * np.sin(2*np.pi * 0.1 * t),   # j1
                    -90 + 10 * np.sin(2*np.pi * 0.1 * t),  # j2 在 -100 ~ -80 之间（安全）
                    20 * np.sin(2*np.pi * 0.1 * t),   # j3 在 -20 ~ 20 之间（安全）
                    10 * np.sin(2*np.pi * 0.1 * t),   # j4
                    10 * np.sin(2*np.pi * 0.1 * t),   # j5
                    20 * np.sin(2*np.pi * 0.1 * t)    # j6
                ]
                trajectory.append([j * cls.DEG2RAD for j in joints_deg])
        
        print(f"生成轨迹: {len(trajectory)}/{n_points} 个路径点")
        
        return trajectory

# ==================== 主采集类 ====================
class DataCollector:
    def __init__(self, robot_ip="192.168.0.50"):
        """连接机器人"""
        print(f"🔌 连接机器人 {robot_ip}...")
        self.robot = lebai.LebaiRobot(robot_ip)

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        self._emergency_stop = False
        
        # 上电
        print("⚡ 上电中...")
        self.robot.start_sys()
        print("✅ 上电成功！")
        
        # 等待系统就绪
        time.sleep(2)
        
        # 移动到安全起始位置
        self.move_to_home()
    
    def move_to_home(self):
        """移动到安全起始位置"""
        home_pose = JointPose(0, -np.pi/2, 0, -np.pi/2, 0, 0)
        self.robot.movej(home_pose, a=1.0, v=0.5, t=0, r=0)
        time.sleep(2)
        print("✅ 已移动到home位")
        time.sleep(1)
    
    def get_full_state(self) -> Dict[str, Any]:
        """
        获取完整的机器人状态
        包含：位置、速度、加速度、力矩、末端位姿
        """
        robot_data = self.robot.get_robot_data()
        
        # 解析数据（需要根据实际返回结构调整）
        return {
            'timestamp': time.time(),
            'q': robot_data.actual_joint,           # 位置 ✓
            'qd': robot_data.actual_joint,    # 速度 ✓
            'qdd': robot_data.actual_acc,  # 加速度 ✓
            'tau': robot_data.actual_torque,  # 力矩 ✓
            'tcp': robot_data.actual_pose       # 末端位姿
        }
    
    def execute_pt_trajectory(self, positions: List[list], duration: float):
        """
        执行PT轨迹（位置-时间）
        使用 move_pt 函数，速度和加速度自动计算
        
        参数:
            positions: 位置列表，每个元素是 [j1,...,j6]
            duration: 总运动时间（秒）
        """

        n_points = len(positions)
        if n_points == 0:
            print("⚠️ 警告：输入路径点为空")
            return []
        
        segment_duration = duration / n_points
        print(f"🔄 执行PT轨迹，时长 {duration}s，{n_points} 个路径点，每个路径点耗时 {segment_duration:.3f}s")

        # 检查当前位置
        initial_state = self.get_full_state()
        initial_q = initial_state['q']
        print(f"   起始位置: [{initial_q[0]:.3f}, {initial_q[1]:.3f}, {initial_q[2]:.3f}, {initial_q[3]:.3f}, {initial_q[4]:.3f}, {initial_q[5]:.3f}]")

        data = []
        for i, pos in enumerate(positions):     
            # 使用 move_pt 连续运动    
            try:
                self.robot.move_pt(pos, segment_duration)
                # print("✅ 轨迹开始执行")
                time.sleep(segment_duration+1)  # 等待当前段运动完成

                # 采集当前状态
                state = self.get_full_state()
                data.append(state)

                if (i+1) % 5 == 0:
                    current_q = state['q']
                    print(f"   进度: {i+1}/{n_points}, 当前位置: [{current_q[0]:.3f}, {current_q[1]:.3f}, {current_q[2]:.3f}]") 

            except Exception as e:
                print(f"❌ 点 {i} 运动失败: {e}")
                return []
        
        # # 等待运动开始
        # time.sleep(0.5)
        
        # # 运动过程中持续采集数据
        # data = []
        # dt = 0.02  # 50Hz采样
        # n_samples = int(duration / dt)
        
        # for i in range(n_samples):
        #     # 检查急停
        #     if self.emergency_stop():
        #         self.robot.stop()
        #         break
            
          
            
        #     time.sleep(dt)
        
        # # 等待运动完全结束
        # time.sleep(0.5)
        print(f"采集完成，采集到 {len(data)} 条数据")
        return data
    
    '''    
    def execute_pvt_trajectory(self, positions: List[list], velocities: List[list], duration: float):
        """
        执行PVT轨迹（位置-速度-时间）
        更精确的控制，但需要自己规划速度
        """
        self.robot.move_pvt(positions, velocities, duration)
        
        # 采集数据（同上）
        # ...
    '''
    def collect_multiple_trajectories(self):
        """采集多种类型的轨迹数据"""
        all_data = []
        
        # 轨迹1：低速运动（主要测重力、静摩擦）
        print("\n📊 轨迹1：低速运动")
        pos1 = JointSafetyChecker.generate_safe_trajectory(20, 10, 1.0)
        data1 = self.execute_pt_trajectory(pos1, 10.0)
        all_data.extend(data1)
        
        # 轨迹2：中速运动（测科里奥利力）
        print("\n📊 轨迹2：中速运动")
        pos2 = JointSafetyChecker.generate_safe_trajectory(30, 5, 1.5)
        data2 = self.execute_pt_trajectory(pos2, 5.0)
        all_data.extend(data2)
        
        # 轨迹3：高速运动（测离心力、动摩擦）
        print("\n📊 轨迹3：高速运动")
        pos3 = JointSafetyChecker.generate_safe_trajectory(50, 3, 2.0)   
        data3 = self.execute_pt_trajectory(pos3, 3.0)
        all_data.extend(data3)
        
        return all_data
# ==================== 急停处理 ====================
    def signal_handler(self, sig, frame):
        print("\n⚠️⚠️⚠️ 检测到急停信号！正在安全停止... ⚠️⚠️⚠️")
        self._emergency_stop = True
        self.robot.stop()
        sys.exit(0)

    def emergency_stop(self):
        """检查是否触发急停"""
        return getattr(self, '_emergency_stop', False)

# ==================== 数据保存 ====================
    def save_data(self, data, filename=None):
        """保存完整的数据集"""
        if not data:
            print("❌ 没有数据可保存")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lebai_lm3/data/real/dynamics_data_{timestamp}.csv"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 转换数据格式
        rows = []
        for d in data:
            row = {
                'timestamp': d['timestamp'],
                'j1': d['q'][0], 'j2': d['q'][1], 'j3': d['q'][2],
                'j4': d['q'][3], 'j5': d['q'][4], 'j6': d['q'][5],
                'v1': d['qd'][0], 'v2': d['qd'][1], 'v3': d['qd'][2],
                'v4': d['qd'][3], 'v5': d['qd'][4], 'v6': d['qd'][5],
                'a1': d['qdd'][0], 'a2': d['qdd'][1],
                'a3': d['qdd'][2], 'a4': d['qdd'][3],
                'a5': d['qdd'][4], 'a6': d['qdd'][5],
                'tau1': d['tau'][0], 'tau2': d['tau'][1], 'tau3': d['tau'][2],
                'tau4': d['tau'][3], 'tau5': d['tau'][4], 'tau6': d['tau'][5],
                'x': d['tcp'][0], 'y': d['tcp'][1], 'z': d['tcp'][2],
                'rx': d['tcp'][3], 'ry': d['tcp'][4], 'rz': d['tcp'][5]
            }
            rows.append(row)
        
        # 保存CSV
        with open(filename, 'w', newline='') as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        
        print(f"✅ 已保存 {len(rows)} 条数据到 {filename}")
        
        # 同时保存为numpy格式，方便训练
        npz_filename = filename.replace('.csv', '.npz')
        np.savez(npz_filename,
                 q=np.array([r['q'] for r in data]),
                 qd=np.array([r['qd'] for r in data]),
                 tau=np.array([r['tau'] for r in data]))
        
        print(f"✅ 已保存NPZ格式: {npz_filename}")
        
        return filename
    
    def close(self):
        """关闭连接"""
        self.robot.stop_sys()
        print("🔌 连接已关闭")

    '''    
    def test_single_trajectory(self):
        """测试单条轨迹是否真的会运动"""
        
        # 生成一个简单的轨迹（从0到30度）
        simple_traj = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],  # 30度
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        
        print("测试轨迹运动...")
        initial_q = self.get_full_state()['q']
        print(f"初始位置: {[f'{j:.3f}' for j in initial_q]}")
        
        try:
            self.robot.move_pt(simple_traj[0], 3.0)
            time.sleep(2)  # 等待运动
            self.robot.move_pt(simple_traj[1], 3.0)
            time.sleep(2)  # 等待运动   
            self.robot.move_pt(simple_traj[2], 3.0)
            time.sleep(2)  # 等待运动   
        except Exception as e:
            print(f"❌ 运动失败: {e}")
            return
        
        final_q = self.get_full_state()['q']
        print(f"最终位置: {[f'{j:.3f}' for j in final_q]}")
        
        diff = np.linalg.norm(np.array(final_q) - np.array(initial_q))
        if diff > 0.1:
            print(f"✅ 运动成功，位移: {diff:.3f}")
        else:
            print(f"❌ 没有明显运动，位移: {diff:.3f}")
        '''

def print_safety_info():
    """打印安全信息"""
    print("\n" + "="*60)
    print("🔒 关节安全限位信息")
    print("="*60)
    
    for joint, info in JointSafetyChecker.COLLISION_ZONES.items():
        print(f"\n{info['name']}:")
        if 'collision_with_base' in info:
            print(f"  ⚠️ 与车体碰撞范围: {info['collision_with_base']['min']}° ~ {info['collision_with_base']['max']}°")
        if 'collision_with_display' in info:
            print(f"  ⚠️ 与显示屏碰撞范围: {info['collision_with_display']['min']}° ~ {info['collision_with_display']['max']}°")
        print(f"  ✅ 安全范围: {info['safe_range']['min']}° ~ {info['safe_range']['max']}°")
    
    print("\n📋 安全使用提示:")
    print("  - 按 Ctrl+C 可以随时安全停止")
    print("  - 所有轨迹都会避开碰撞区域")
    print("  - 建议先小范围测试再全量运行")
    print("="*60)

if __name__ == "__main__":
    # 打印安全信息
    print_safety_info()
    
    # 确认继续
    response = input("\n是否开始采集数据？(y/n): ")
    if response.lower() != 'y':
        print("采集已取消")
        sys.exit(0)
    
    # 配置
    ROBOT_IP = "192.168.0.50"  # 你的机器人IP
    
    # 创建采集器
    collector = DataCollector(ROBOT_IP)
    # collector.test_single_trajectory()
    # res =JointSafetyChecker.generate_safe_trajectory(100, 10)
    # print(f"生成的安全轨迹: {res}")
    
    try:
        # 采集多种轨迹数据
        data = collector.collect_multiple_trajectories()
        
        # 保存数据
        filename = collector.save_data(data)
        
        # 显示统计信息
        print(f"\n📊 采集完成！共 {len(data)} 个数据点")
        
        
    except Exception as e:
        print(f"❌ 采集过程中出错: {e}")
        
    finally:
        collector.close()