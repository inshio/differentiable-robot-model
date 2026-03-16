# test_safety.py
from lebai_lm3.data_collection.collect_full_data import JointSafetyChecker

# 测试用例
test_cases = [
    ([-180, -90, 30], "正常位置1"),
    ([-180, 0, 50], "j2=0 > -20, j3=50 >40 应该不安全"),
    ([-180, -30, 70], "j2=-30 ≤ -20, j3=70 >65 应该安全"),
    ([-180, -90, 60], "j2=-90 ≤ -20, j3=60 ≤65 应该安全"),
    ([-180, 10, 35], "j2=10 > -20, j3=35 ≤40 应该安全"),
]

for joints, desc in test_cases:
    is_safe, reason = JointSafetyChecker.check_safety(joints)
    print(f"{desc}: {'✅' if is_safe else '❌'} {reason}")