# data_collection/collect_full_data.py
"""
数据集采集脚本（适配含额外列的CSV）
从CSV读取关节路径点（仅提取后6列），通过move_pt/movej执行运动并采集数据
采集：位置、速度、加速度、力矩
时间：10s
"""

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
import threading

# ==================== 主采集类 ====================
class DataCollector:
    def __init__(self, robot_ip="192.168.0.50"):
        """连接机器人"""
        print(f"🔌 连接机器人 {robot_ip}...")
        self.robot = lebai.LebaiRobot(robot_ip)

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        self._emergency_stop = False

        self.running = False
        self.data_list = []  # 用于存储采集到的原始数据
        self.lock = threading.Lock() # 线程锁，保证数据写入安全
        
        # 上电
        print("⚡ 上电中...")
        self.robot.start_sys()
        print("✅ 上电成功！")
        
        # 等待系统就绪
        time.sleep(1)
        
        # 移动到安全起始位置
        self.move_to_home()
    
    def move_to_home(self):
        """移动到安全起始位置"""
        home_pose = JointPose(0, -np.pi/2, 0, -np.pi/2, 0, 0)
        self.robot.movej(home_pose, a=1.0, v=0.5, t=0, r=0)
        time.sleep(5)
        print("✅ 已移动到home位")
        time.sleep(1)
    
    def _collect_loop(self):
        """后台高频采集线程 (约200Hz)"""
        print("📊 异步采集线程启动...")
        while self.running:
            try:
                # 获取全量状态反馈
                d = self.robot.get_robot_data()
                state = {
                    'timestamp': time.time(),
                    'q': d.actual_joint,      # 实际位置
                    'qd': d.actual_vel,      # 实际速度
                    'qdd': d.actual_acc,     # 实际加速度
                    'tau': d.actual_torque,  # 实际力矩
                    'tcp': d.actual_pose     # 末端位姿
                }
                with self.lock:
                    self.data_list.append(state)
                
                # 采样频率控制
                time.sleep(0.005) 
            except Exception as e:
                if self.running:
                    print(f"⚠️ 采集异常: {e}")
    
    def load_trajectory_from_csv(self, csv_path):
        """解析 C++ 生成的 CSV (格式: Index, Time, J1, J2, J3, J4, J5, J6)"""
        times = []
        positions = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader) # 跳过表header
            for row in reader:
                if not row: continue
                times.append(float(row[1])) # 第二列 Time_Sec
                positions.append([float(x) for x in row[2:8]]) # 第三到八列 J1-J6
        print(f"✅ 已加载 {len(positions)} 个路径点")
        return times, positions
    
    def execute_pt_trajectory(self, csv_path):
        """
        执行PT轨迹（位置-时间）
        使用 move_pt 函数，速度和加速度自动计算
        多线程采集数据，确保高频率和完整性
        """
        times, positions = self.load_trajectory_from_csv(csv_path)
        
        # 1. 先平滑移动到轨迹的第一个点
        print("移动到轨迹起点...")
        self.robot.movej(JointPose(*positions[0]), v=0.5, a=1.0)
        time.sleep(5)

        # 2. 启动后台采集
        self.data_list = []
        self.running = True
        collect_thread = threading.Thread(target=self._collect_loop)
        collect_thread.daemon = True
        collect_thread.start()

        # 3. 100Hz 轨迹推送
        print("开始执行 move_pt 连续轨迹...")
        start_time = time.time()    
        try:
            for i in range(1, len(positions)):
                target_time = times[i] - times[0]
                dt = times[i] - times[i-1]
                # 容错处理：确保 dt 合法
                # dt = max(dt, 0.01) 

                # 发送指令
                self.robot.move_pt(positions[i], dt)
                elapsed_time = time.time() - start_time
                wait_time = target_time - elapsed_time
                if wait_time > 0:
                    # 网络延迟补偿0.002
                    time.sleep(max(wait_time - 0.002, 0))

                if i % 10 == 0:
                    sys.stdout.write(f"\r已执行 {i}/{len(positions)} 个路径点")
                    sys.stdout.flush()
            
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ 执行失败: {e}")
        finally:
            self.running = False
            collect_thread.join()
            print(f"运动结束，采集到 {len(self.data_list)} 条数据")


    def execute_continuous_trajectory(self, positions: List[list], duration: float):
        """
        连续轨迹执行 - 使用 movej   
        """
        n_points = len(positions)
        if n_points < 2:
            return []
        
        # 计算每个路径点的速度
        v = 0.5  # 固定速度（rad/s）
        a = 1.0  # 加速度（rad/s²）
        
        data = []
        
        # 连续移动到后续点（使用轨迹过渡）
        for i, pos in enumerate(positions, 1):
            try:
                # 计算到下一个点的距离
                current_q = self.get_full_state()['q']
                delta = np.linalg.norm(np.array(pos) - np.array(current_q))
                
                # 动态调整速度
                segment_time = duration / n_points
                speed = min(v, delta / segment_time)
                
                # 执行运动，r>0 表示轨迹过渡
                pos = JointPose(*pos)
                self.robot.movej(pos, a=a, v=speed, t=0, r=0.05)  # r=5% 轨迹过渡
                
                # 采集数据
                if i % 1 == 0:  # 每个路径点都采集
                    # 等待当前段实际执行到该点（根据时间估算）
                    time.sleep(segment_time * 0.05)  # 等待让机器人更接近目标点
                    state = self.get_full_state()
                    data.append(state)

                if (i+1) % 10 == 0:
                    current_q = state['q']
                    print(f"   进度: {i+1}/{n_points}, 当前位置: [{current_q[0]:.3f}, {current_q[1]:.3f}, {current_q[2]:.3f}]") 

                state = self.get_full_state()
                data.append(state)
                print(f"采集完成，采集到 {len(data)} 条数据")

            except Exception as e:
                print(f"❌ 点 {i} 运动失败: {e}")
                return []
        
        return data
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
    def save_data(self,csv_name):
        """保存完整的数据集"""
        if not self.data_list:
            print("❌ 没有数据可保存")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lebai_lm3/data/real/moveit_ompl_chomp_movept/realfeedback_test5_{timestamp}.csv"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 保存CSV
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['sys_time','j1','j2','j3','j4','j5','j6',
                             'v1','v2','v3','v4','v5','v6',
                             'tau1','tau2','tau3','tau4','tau5','tau6'])
            
            with self.lock:
                for d in self.data_list:
                    row = [d['timestamp']] + list(d['q']) + list(d['qd']) + list(d['tau'])
                    writer.writerow(row)
        
        print(f"已保存 {len(self.data_list)} 条数据到 {filename}")
        
        # 同时保存为numpy格式，方便训练
        npz_filename = filename.replace('.csv', '.npz')
        np.savez(npz_filename,
                 q=np.array([d['q'] for d in self.data_list]),
                 qd=np.array([d['qd'] for d in self.data_list]),
                 qdd=np.array([d['qdd'] for d in self.data_list]),
                 tau=np.array([d['tau'] for d in self.data_list]))
        
        print(f"已保存NPZ格式: {npz_filename}")
        
        return filename
    
    def close(self):
        """关闭连接"""
        self.robot.stop()
        print("🔌 连接已关闭")

def print_usage_info():
    """打印使用信息"""
    print("\n" + "="*60)
    print("🔧 CSV轨迹执行器（适配含额外列）")
    print("="*60)
    print("📋 CSV文件格式要求:")
    print("  - 前6列必须是j1-j6关节角度（弧度）")
    print("  - 后续列可以是任意内容（时间、点位ID、备注等），会自动忽略")
    print("  - 支持带表头（如time,point_id,j1,j2,j3,j4,j5,j6,remark）")
    print("\n💡 操作提示:")
    print("  - 按 Ctrl+C 可以随时安全停止")
    print("  - 确保前6列数据为有效数字（弧度）")
    print("  - 建议先小范围测试再全量运行")
    print("="*60)

if __name__ == "__main__":
    # 打印使用信息
    print_usage_info()
    
    # 确认继续
    response = input("\n是否开始执行CSV轨迹？(y/n): ")
    if response.lower() != 'y':
        print("执行已取消")
        sys.exit(0)
    
    # 配置
    ROBOT_IP = "192.168.0.50"  # 你的机器人IP
    CSV_PATH = input("请输入CSV文件路径: ").strip()  # CSV文件路径
    TOTAL_DURATION = 10 # 总运动时间S
    
    # 创建采集器
    collector = DataCollector(ROBOT_IP)
    
    try:       
        # 执行轨迹并采集数据
        collector.execute_pt_trajectory(CSV_PATH)
        # data = collector.execute_continuous_trajectory(positions, TOTAL_DURATION)
        
        # 保存数据
        filename = collector.save_data(CSV_PATH)
        
    except Exception as e:
        print(f"❌ 执行过程中出错: {e}")
        
    finally:
        collector.close()
