"""
Hệ thống Luật Mờ (Fuzzy Logic) cho Dự đoán Trầm Cảm
Dựa trên phân tích SHAP của mô hình XGBoost

Đầu vào (Inputs):
- Áp lực học tập (Academic Pressure): 0-5
- Áp lực tài chính (Financial Stress): 1-5
- Thời gian ngủ (Sleep Duration): 4-9 giờ

Đầu ra (Output):
- Mức độ rủi ro trầm cảm (Depression Risk): 0-100
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = r"C:\Users\Thai Binh\Documents\HocHanh\Python\project\Depression Prediction"

# ============================================================
# 1. Định nghĩa các biến mờ (Fuzzy Variables)
# ============================================================
print("=" * 60)
print("BƯỚC 1: Định nghĩa các biến mờ")
print("=" * 60)

# Đầu vào 1: Áp lực học tập (Academic Pressure) - thang 0-5
academic_pressure = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'academic_pressure')
academic_pressure['thấp'] = fuzz.trimf(academic_pressure.universe, [0, 0, 2])
academic_pressure['trung_bình'] = fuzz.trimf(academic_pressure.universe, [1, 2.5, 4])
academic_pressure['cao'] = fuzz.trimf(academic_pressure.universe, [3, 5, 5])

# Đầu vào 2: Áp lực tài chính (Financial Stress) - thang 1-5
financial_stress = ctrl.Antecedent(np.arange(1, 5.1, 0.1), 'financial_stress')
financial_stress['thấp'] = fuzz.trimf(financial_stress.universe, [1, 1, 2.5])
financial_stress['trung_bình'] = fuzz.trimf(financial_stress.universe, [1.5, 3, 4.5])
financial_stress['cao'] = fuzz.trimf(financial_stress.universe, [3.5, 5, 5])

# Đầu vào 3: Thời gian ngủ (Sleep Duration) - 4-9 giờ
sleep_duration = ctrl.Antecedent(np.arange(4, 9.1, 0.1), 'sleep_duration')
sleep_duration['ít'] = fuzz.trimf(sleep_duration.universe, [4, 4, 5.5])
sleep_duration['trung_bình'] = fuzz.trimf(sleep_duration.universe, [5, 6.5, 8])
sleep_duration['nhiều'] = fuzz.trimf(sleep_duration.universe, [7, 9, 9])

# Đầu ra: Mức độ rủi ro trầm cảm (Depression Risk) - 0-100
depression_risk = ctrl.Consequent(np.arange(0, 101, 1), 'depression_risk')
depression_risk['rất_thấp'] = fuzz.trimf(depression_risk.universe, [0, 0, 20])
depression_risk['thấp'] = fuzz.trimf(depression_risk.universe, [10, 25, 40])
depression_risk['trung_bình'] = fuzz.trimf(depression_risk.universe, [30, 50, 70])
depression_risk['cao'] = fuzz.trimf(depression_risk.universe, [60, 75, 90])
depression_risk['rất_cao'] = fuzz.trimf(depression_risk.universe, [80, 100, 100])

# --- Trực quan hóa hàm thành viên mờ ---
fig, axes = plt.subplots(4, 1, figsize=(12, 16))

academic_pressure.view(ax=axes[0])
axes[0].set_title('Áp lực học tập (Academic Pressure)', fontsize=14, fontweight='bold')
axes[0].legend()

financial_stress.view(ax=axes[1])
axes[1].set_title('Áp lực tài chính (Financial Stress)', fontsize=14, fontweight='bold')
axes[1].legend()

sleep_duration.view(ax=axes[2])
axes[2].set_title('Thời gian ngủ (Sleep Duration)', fontsize=14, fontweight='bold')
axes[2].legend()

depression_risk.view(ax=axes[3])
axes[3].set_title('Mức độ rủi ro trầm cảm (Depression Risk)', fontsize=14, fontweight='bold')
axes[3].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}\\fuzzy_membership_functions.png", dpi=150, bbox_inches='tight')
plt.close()
print("Đã lưu biểu đồ hàm thành viên mờ → fuzzy_membership_functions.png")

# ============================================================
# 2. Định nghĩa các luật mờ (Fuzzy Rules)
#    Dựa trên phân tích SHAP: 3 đặc trưng quan trọng
#    Academic Pressure, Financial Stress, Sleep Duration
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 2: Định nghĩa các luật mờ")
print("=" * 60)

rules = []

# ===== NHÓM 1: Rủi ro RẤT CAO =====

# R1: Áp lực học cao VÀ tài chính cao VÀ ngủ ít → Rất cao
rules.append(ctrl.Rule(
    academic_pressure['cao'] & financial_stress['cao'] & sleep_duration['ít'],
    depression_risk['rất_cao']
))

# R2: Áp lực học cao VÀ tài chính cao VÀ ngủ TB → Rất cao
rules.append(ctrl.Rule(
    academic_pressure['cao'] & financial_stress['cao'] & sleep_duration['trung_bình'],
    depression_risk['rất_cao']
))

# ===== NHÓM 2: Rủi ro CAO =====

# R3: Áp lực học cao VÀ tài chính TB VÀ ngủ ít → Cao
rules.append(ctrl.Rule(
    academic_pressure['cao'] & financial_stress['trung_bình'] & sleep_duration['ít'],
    depression_risk['cao']
))

# R4: Áp lực học TB VÀ tài chính cao VÀ ngủ ít → Cao
rules.append(ctrl.Rule(
    academic_pressure['trung_bình'] & financial_stress['cao'] & sleep_duration['ít'],
    depression_risk['cao']
))

# R5: Áp lực học cao VÀ tài chính cao VÀ ngủ nhiều → Cao
rules.append(ctrl.Rule(
    academic_pressure['cao'] & financial_stress['cao'] & sleep_duration['nhiều'],
    depression_risk['cao']
))

# R6: Áp lực học cao VÀ tài chính TB VÀ ngủ TB → Cao
rules.append(ctrl.Rule(
    academic_pressure['cao'] & financial_stress['trung_bình'] & sleep_duration['trung_bình'],
    depression_risk['cao']
))

# R7: Áp lực học TB VÀ tài chính cao VÀ ngủ TB → Cao
rules.append(ctrl.Rule(
    academic_pressure['trung_bình'] & financial_stress['cao'] & sleep_duration['trung_bình'],
    depression_risk['cao']
))

# ===== NHÓM 3: Rủi ro TRUNG BÌNH =====

# R8: Áp lực học TB VÀ tài chính TB VÀ ngủ ít → TB
rules.append(ctrl.Rule(
    academic_pressure['trung_bình'] & financial_stress['trung_bình'] & sleep_duration['ít'],
    depression_risk['trung_bình']
))

# R9: Áp lực học TB VÀ tài chính TB VÀ ngủ TB → TB
rules.append(ctrl.Rule(
    academic_pressure['trung_bình'] & financial_stress['trung_bình'] & sleep_duration['trung_bình'],
    depression_risk['trung_bình']
))

# R10: Áp lực học cao VÀ tài chính thấp VÀ ngủ TB → TB
rules.append(ctrl.Rule(
    academic_pressure['cao'] & financial_stress['thấp'] & sleep_duration['trung_bình'],
    depression_risk['trung_bình']
))

# R11: Áp lực học thấp VÀ tài chính cao VÀ ngủ ít → TB
rules.append(ctrl.Rule(
    academic_pressure['thấp'] & financial_stress['cao'] & sleep_duration['ít'],
    depression_risk['trung_bình']
))

# R12: Áp lực học cao VÀ tài chính thấp VÀ ngủ ít → TB
rules.append(ctrl.Rule(
    academic_pressure['cao'] & financial_stress['thấp'] & sleep_duration['ít'],
    depression_risk['trung_bình']
))

# R13: Áp lực học TB VÀ tài chính TB VÀ ngủ nhiều → TB
rules.append(ctrl.Rule(
    academic_pressure['trung_bình'] & financial_stress['trung_bình'] & sleep_duration['nhiều'],
    depression_risk['trung_bình']
))

# R14: Áp lực học thấp VÀ tài chính cao VÀ ngủ TB → TB
rules.append(ctrl.Rule(
    academic_pressure['thấp'] & financial_stress['cao'] & sleep_duration['trung_bình'],
    depression_risk['trung_bình']
))

# ===== NHÓM 4: Rủi ro THẤP =====

# R15: Áp lực học thấp VÀ tài chính TB VÀ ngủ TB → Thấp
rules.append(ctrl.Rule(
    academic_pressure['thấp'] & financial_stress['trung_bình'] & sleep_duration['trung_bình'],
    depression_risk['thấp']
))

# R16: Áp lực học TB VÀ tài chính thấp VÀ ngủ TB → Thấp
rules.append(ctrl.Rule(
    academic_pressure['trung_bình'] & financial_stress['thấp'] & sleep_duration['trung_bình'],
    depression_risk['thấp']
))

# R17: Áp lực học thấp VÀ tài chính TB VÀ ngủ nhiều → Thấp
rules.append(ctrl.Rule(
    academic_pressure['thấp'] & financial_stress['trung_bình'] & sleep_duration['nhiều'],
    depression_risk['thấp']
))

# R18: Áp lực học TB VÀ tài chính thấp VÀ ngủ nhiều → Thấp
rules.append(ctrl.Rule(
    academic_pressure['trung_bình'] & financial_stress['thấp'] & sleep_duration['nhiều'],
    depression_risk['thấp']
))

# R19: Áp lực học cao VÀ tài chính thấp VÀ ngủ nhiều → Thấp
rules.append(ctrl.Rule(
    academic_pressure['cao'] & financial_stress['thấp'] & sleep_duration['nhiều'],
    depression_risk['thấp']
))

# R20: Áp lực học thấp VÀ tài chính cao VÀ ngủ nhiều → Thấp
rules.append(ctrl.Rule(
    academic_pressure['thấp'] & financial_stress['cao'] & sleep_duration['nhiều'],
    depression_risk['thấp']
))

# R21: Áp lực học thấp VÀ tài chính thấp VÀ ngủ ít → Thấp
rules.append(ctrl.Rule(
    academic_pressure['thấp'] & financial_stress['thấp'] & sleep_duration['ít'],
    depression_risk['thấp']
))

# ===== NHÓM 5: Rủi ro RẤT THẤP =====

# R22: Áp lực học thấp VÀ tài chính thấp VÀ ngủ nhiều → Rất thấp
rules.append(ctrl.Rule(
    academic_pressure['thấp'] & financial_stress['thấp'] & sleep_duration['nhiều'],
    depression_risk['rất_thấp']
))

# R23: Áp lực học thấp VÀ tài chính thấp VÀ ngủ TB → Rất thấp
rules.append(ctrl.Rule(
    academic_pressure['thấp'] & financial_stress['thấp'] & sleep_duration['trung_bình'],
    depression_risk['rất_thấp']
))

# R24: Áp lực học thấp VÀ tài chính TB VÀ ngủ ít → Thấp
rules.append(ctrl.Rule(
    academic_pressure['thấp'] & financial_stress['trung_bình'] & sleep_duration['ít'],
    depression_risk['thấp']
))

# R25: Áp lực học TB VÀ tài chính thấp VÀ ngủ ít → Thấp
rules.append(ctrl.Rule(
    academic_pressure['trung_bình'] & financial_stress['thấp'] & sleep_duration['ít'],
    depression_risk['thấp']
))

print(f"Tổng số luật mờ đã định nghĩa: {len(rules)}")

# --- Tạo hệ thống điều khiển mờ ---
depression_ctrl = ctrl.ControlSystem(rules)
depression_sim = ctrl.ControlSystemSimulation(depression_ctrl)

print("Hệ thống luật mờ đã được xây dựng thành công!")

# ============================================================
# 3. Hàm tiện ích: Dự đoán rủi ro trầm cảm
# ============================================================

def predict_depression_risk(academic, financial, sleep):
    """
    Dự đoán mức độ rủi ro trầm cảm bằng hệ thống luật mờ.

    Tham số:
        academic (float): Áp lực học tập (0-5)
        financial (float): Áp lực tài chính (1-5)
        sleep (float): Thời gian ngủ (4-9 giờ)

    Trả về:
        float: Mức độ rủi ro trầm cảm (0-100)
    """
    sim = ctrl.ControlSystemSimulation(depression_ctrl)
    sim.input['academic_pressure'] = np.clip(academic, 0, 5)
    sim.input['financial_stress'] = np.clip(financial, 1, 5)
    sim.input['sleep_duration'] = np.clip(sleep, 4, 9)
    sim.compute()
    return sim.output['depression_risk']


print("\n" + "=" * 60)
print("HOÀN THÀNH!")
print("=" * 60)
print(f"\nCác thành phần đã tạo:")
print(f"  - Hệ thống luật mờ với {len(rules)} luật")
print(f"  - Hàm predict_depression_risk() để dự đoán")
print(f"  - Biểu đồ: fuzzy_membership_functions.png")
print(f"\nCách sử dụng:")
print(f"  risk = predict_depression_risk(")
print(f"      academic=4.0,    # Áp lực học tập (0-5)")
print(f"      financial=4.0,   # Áp lực tài chính (1-5)")
print(f"      sleep=5.0        # Thời gian ngủ (4-9h)")
print(f"  )")
