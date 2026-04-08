"""
Ứng dụng Demo Dự đoán Trầm cảm
Quy trình: Người dùng nhập → XGBoost predict_proba → Đẩy 5 số vào Fuzzy Logic → Kết luận cuối cùng

Pipeline:
  1) Người dùng nhập thông tin (Học tập, Tài chính, Tự tử, ...)
  2) Gọi XGBoost: model.predict_proba(data) → xác suất ML (vd: 0.55)
  3) Đẩy 5 giá trị (academic_pressure, financial_stress, suicidal, sleep_duration, ml_prob)
     vào hệ thống Luật Mờ (skfuzzy)
  4) Luật mờ kích hoạt → Kết luận cuối cùng cho người dùng
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import warnings
warnings.filterwarnings('ignore')

# (Đã bỏ SHAP — không cần ánh xạ feature tiếng Việt)

# ============================================================
# CẤU HÌNH
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

XGB_MODEL_PATH = os.path.join(BASE_DIR, "xgb_model.pkl")
XGB_MODEL_NS_PATH = os.path.join(BASE_DIR, "xgb_model_ns.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "threshold.pkl")

SLEEP_MAP = {
    "Ít hơn 5 giờ": 4,
    "5-6 giờ": 5.5,
    "7-8 giờ": 7.5,
    "Hơn 8 giờ": 9,
}

DIET_MAP = {"Không lành mạnh": 0, "Bình thường": 1, "Lành mạnh": 2}

PROFESSIONS = [
    "Student", "Chef", "Civil Engineer", "Content Writer",
    "Digital Marketer", "Doctor", "Educational Consultant",
    "Entrepreneur", "Lawyer", "Manager", "Pharmacist",
    "Teacher", "UX/UI Designer"
]

# ============================================================
# TẢI MÔ HÌNH
# ============================================================
@st.cache_resource
def load_models():
    models = {}
    models['xgb'] = joblib.load(XGB_MODEL_PATH)
    models['xgb_ns'] = joblib.load(XGB_MODEL_NS_PATH)
    try:
        models['threshold'] = joblib.load(THRESHOLD_PATH)
    except:
        models['threshold'] = 0.5
    return models

# ============================================================
# HỆ THỐNG LUẬT MỜ — 5 ĐẦU VÀO  (MF MỞ RỘNG OVERLAP)
# Đầu vào: academic_pressure, financial_stress, suicidal_thoughts, sleep_duration, ml_probability
# Đầu ra:  depression_risk (0-100)
#
# So với bản gốc: giữ nguyên vị trí đỉnh (peak) và tỉ lệ tổng thể,
# chỉ mở rộng vùng chồng lấp (overlap) giữa các MF liền kề để
# defuzzification cho ra phổ giá trị liên tục thay vì dính ở vài giá trị cố định.
# ============================================================
@st.cache_resource
def build_fuzzy_system():
    # --- Đầu vào 1: Áp lực học tập (1-5) ---
    #   Gốc: thấp [1,1,2.5] | TB [2,3,4] | cao [3.5,5,5]
    #   Mới:  mở rộng tails để overlap nhiều hơn
    ap = ctrl.Antecedent(np.arange(1, 5.1, 0.1), 'academic_pressure')
    ap['thấp']       = fuzz.trimf(ap.universe, [1, 1, 2.5])
    ap['trung_bình'] = fuzz.trimf(ap.universe, [2.5, 3, 4])
    ap['cao']        = fuzz.trimf(ap.universe, [3.5, 5, 5])

    # --- Đầu vào 2: Áp lực tài chính (1-5) ---
    #   Tương tự academic_pressure
    fs = ctrl.Antecedent(np.arange(1, 5.1, 0.1), 'financial_stress')
    fs['thấp']       = fuzz.trimf(fs.universe, [1, 1, 2.5])
    fs['trung_bình'] = fuzz.trimf(fs.universe, [2.5, 3, 4])
    fs['cao']        = fuzz.trimf(fs.universe, [3.5, 5, 5])

    # --- Đầu vào 3: Suy nghĩ tự tử (0 hoặc 1) — giữ nguyên ---
    st_in = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'suicidal_thoughts')
    st_in['không'] = fuzz.trimf(st_in.universe, [0, 0, 0.4])
    st_in['có']    = fuzz.trimf(st_in.universe, [0.6, 1, 1])

    # --- Đầu vào 4: Thời lượng giấc ngủ (giờ) ---
    #   Gốc: thiếu [3,3,4.5,6] | BT [5,6,7,7.5] | đủ [7,7.5,10,10]
    #   Mới:  mở rộng overlap
    slp = ctrl.Antecedent(np.arange(3, 10.1, 0.1), 'sleep_duration')
    slp['thiếu_ngủ']   = fuzz.trapmf(slp.universe, [3, 3, 4.5, 6])
    slp['bình_thường']  = fuzz.trapmf(slp.universe, [5, 5.5, 7, 7.5])
    slp['đủ_giấc']      = fuzz.trapmf(slp.universe, [7, 7.5, 10, 10])

    # --- Đầu vào 5: XÁC SUẤT TỪ XGBOOST (0.0 - 1.0) ---
    #   Gốc: thấp [0,0,0.64] | lấp_lửng [0.55,0.75,0.84] | cao [0.85,1,1]
    #   Mới:  mở rộng overlap, giữ tỉ lệ tương đối
    ml = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'ml_probability')
    ml['thấp']      = fuzz.trimf(ml.universe, [0.0, 0.0, 0.70])
    ml['lấp_lửng']  = fuzz.trimf(ml.universe, [0.50, 0.72, 0.90])
    ml['cao']        = fuzz.trimf(ml.universe, [0.78, 1.0, 1.0])

    # --- Đầu ra: Rủi ro trầm cảm (0-100) ---
    #   Gốc: rất_thấp [0,0,20] | thấp [10,25,40] | TB [30,50,70]
    #         cao [60,75,90] | rất_cao [80,100,100]
    #   Mới:  giữ peaks (0,25,50,75,100), mở rộng overlap đáng kể
    dr = ctrl.Consequent(np.arange(0, 101, 1), 'depression_risk')
    dr['rất_thấp']   = fuzz.trimf(dr.universe, [0, 0, 25])
    dr['thấp']       = fuzz.trimf(dr.universe, [5, 25, 48])
    dr['trung_bình'] = fuzz.trimf(dr.universe, [25, 50, 75])
    dr['cao']        = fuzz.trimf(dr.universe, [52, 75, 95])
    dr['rất_cao']    = fuzz.trimf(dr.universe, [72, 100, 100])

    rules = []

    # ================================================================
    #  NHÓM A: CÓ SUY NGHĨ TỰ TỬ  →  luôn nghiêm trọng
    #  Phân biệt rõ hơn theo từng tổ hợp áp lực
    # ================================================================
    # ML cao + tự tử → rất cao (bất kể áp lực)
    rules.append(ctrl.Rule(st_in['có'] & ml['cao'], dr['rất_cao']))

    # ML lấp lửng + tự tử + ít nhất 1 áp lực cao → rất cao
    rules.append(ctrl.Rule(st_in['có'] & ml['lấp_lửng'] & ap['cao'], dr['rất_cao']))
    rules.append(ctrl.Rule(st_in['có'] & ml['lấp_lửng'] & fs['cao'], dr['rất_cao']))
    # ML lấp lửng + tự tử + cả hai TB → cao
    rules.append(ctrl.Rule(st_in['có'] & ml['lấp_lửng'] & ap['trung_bình'] & fs['trung_bình'], dr['cao']))
    # ML lấp lửng + tự tử + 1 TB + 1 thấp → cao
    rules.append(ctrl.Rule(st_in['có'] & ml['lấp_lửng'] & ap['trung_bình'] & fs['thấp'], dr['cao']))
    rules.append(ctrl.Rule(st_in['có'] & ml['lấp_lửng'] & ap['thấp'] & fs['trung_bình'], dr['cao']))
    # ML lấp lửng + tự tử + cả hai thấp → trung bình
    rules.append(ctrl.Rule(st_in['có'] & ml['lấp_lửng'] & ap['thấp'] & fs['thấp'], dr['cao']))

    # ML thấp + tự tử + ít nhất 1 áp lực cao → cao
    rules.append(ctrl.Rule(st_in['có'] & ml['thấp'] & ap['cao'], dr['cao']))
    rules.append(ctrl.Rule(st_in['có'] & ml['thấp'] & fs['cao'], dr['cao']))
    # ML thấp + tự tử + cả hai TB → cao (vẫn nguy hiểm vì có ý tự tử)
    rules.append(ctrl.Rule(st_in['có'] & ml['thấp'] & ap['trung_bình'] & fs['trung_bình'], dr['cao']))
    # ML thấp + tự tử + 1 TB + 1 thấp → cao
    rules.append(ctrl.Rule(st_in['có'] & ml['thấp'] & ap['trung_bình'] & fs['thấp'], dr['cao']))
    rules.append(ctrl.Rule(st_in['có'] & ml['thấp'] & ap['thấp'] & fs['trung_bình'], dr['cao']))
    # ML thấp + tự tử + cả hai thấp → cao (ý tự tử luôn cần cảnh báo)
    rules.append(ctrl.Rule(st_in['có'] & ml['thấp'] & ap['thấp'] & fs['thấp'], dr['cao']))

    # ================================================================
    #  NHÓM B: KHÔNG TỰ TỬ — ML CAO
    #  Ngủ thiếu/BT → giữ mức cao | Ngủ đủ giấc → cap ở trung_bình
    # ================================================================
    # --- Ngủ KHÔNG đủ: cap ở cao (rất_cao chỉ khi có ý tự tử) ---
    rules.append(ctrl.Rule(st_in['không'] & ml['cao'] & ap['cao'] & fs['cao'] & (slp['thiếu_ngủ'] | slp['bình_thường']), dr['cao']))
    rules.append(ctrl.Rule(st_in['không'] & ml['cao'] & ap['cao'] & fs['trung_bình'] & (slp['thiếu_ngủ'] | slp['bình_thường']), dr['cao']))
    rules.append(ctrl.Rule(st_in['không'] & ml['cao'] & ap['trung_bình'] & fs['cao'] & (slp['thiếu_ngủ'] | slp['bình_thường']), dr['cao']))
    rules.append(ctrl.Rule(st_in['không'] & ml['cao'] & ap['trung_bình'] & fs['trung_bình'] & (slp['thiếu_ngủ'] | slp['bình_thường']), dr['cao']))
    rules.append(ctrl.Rule(st_in['không'] & ml['cao'] & ap['thấp'] & fs['cao'] & (slp['thiếu_ngủ'] | slp['bình_thường']), dr['cao']))
    rules.append(ctrl.Rule(st_in['không'] & ml['cao'] & ap['cao'] & fs['thấp'] & (slp['thiếu_ngủ'] | slp['bình_thường']), dr['cao']))
    # --- Ngủ ĐỦ GIẤC: giấc ngủ bảo vệ, nhưng không đủ khi cả hai áp lực MAX ---
    rules.append(ctrl.Rule(st_in['không'] & ml['cao'] & ap['cao'] & fs['cao'] & slp['đủ_giấc'], dr['cao']))
    rules.append(ctrl.Rule(st_in['không'] & ml['cao'] & ap['cao'] & fs['trung_bình'] & slp['đủ_giấc'], dr['trung_bình']))
    rules.append(ctrl.Rule(st_in['không'] & ml['cao'] & ap['trung_bình'] & fs['cao'] & slp['đủ_giấc'], dr['trung_bình']))
    rules.append(ctrl.Rule(st_in['không'] & ml['cao'] & ap['trung_bình'] & fs['trung_bình'] & slp['đủ_giấc'], dr['trung_bình']))
    rules.append(ctrl.Rule(st_in['không'] & ml['cao'] & ap['thấp'] & fs['cao'] & slp['đủ_giấc'], dr['trung_bình']))
    rules.append(ctrl.Rule(st_in['không'] & ml['cao'] & ap['cao'] & fs['thấp'] & slp['đủ_giấc'], dr['trung_bình']))
    # --- Đã ở mức trung bình/thấp — không cần tách theo giấc ngủ ---
    rules.append(ctrl.Rule(st_in['không'] & ml['cao'] & ap['thấp'] & fs['thấp'], dr['trung_bình']))
    rules.append(ctrl.Rule(st_in['không'] & ml['cao'] & ap['thấp'] & fs['trung_bình'], dr['trung_bình']))
    rules.append(ctrl.Rule(st_in['không'] & ml['cao'] & ap['trung_bình'] & fs['thấp'], dr['trung_bình']))

    # ================================================================
    #  NHÓM C: KHÔNG TỰ TỬ — ML LẤP LỬNG
    #  Ngủ thiếu/BT → giữ mức cao | Ngủ đủ giấc → cap ở trung_bình
    # ================================================================
    # --- Ngủ KHÔNG đủ: giữ nguyên ---
    rules.append(ctrl.Rule(st_in['không'] & ml['lấp_lửng'] & ap['cao'] & fs['cao'] & (slp['thiếu_ngủ'] | slp['bình_thường']), dr['cao']))
    rules.append(ctrl.Rule(st_in['không'] & ml['lấp_lửng'] & ap['cao'] & fs['trung_bình'] & (slp['thiếu_ngủ'] | slp['bình_thường']), dr['cao']))
    rules.append(ctrl.Rule(st_in['không'] & ml['lấp_lửng'] & ap['trung_bình'] & fs['cao'] & (slp['thiếu_ngủ'] | slp['bình_thường']), dr['cao']))
    # --- Ngủ ĐỦ GIẤC: giấc ngủ bảo vệ, nhưng không đủ khi cả hai áp lực MAX ---
    rules.append(ctrl.Rule(st_in['không'] & ml['lấp_lửng'] & ap['cao'] & fs['cao'] & slp['đủ_giấc'], dr['cao']))
    rules.append(ctrl.Rule(st_in['không'] & ml['lấp_lửng'] & ap['cao'] & fs['trung_bình'] & slp['đủ_giấc'], dr['trung_bình']))
    rules.append(ctrl.Rule(st_in['không'] & ml['lấp_lửng'] & ap['trung_bình'] & fs['cao'] & slp['đủ_giấc'], dr['trung_bình']))
    # --- Đã ở mức trung bình/thấp — tách theo giấc ngủ ---
    rules.append(ctrl.Rule(st_in['không'] & ml['lấp_lửng'] & ap['trung_bình'] & fs['trung_bình'], dr['trung_bình']))
    rules.append(ctrl.Rule(st_in['không'] & ml['lấp_lửng'] & ap['cao'] & fs['thấp'], dr['trung_bình']))
    rules.append(ctrl.Rule(st_in['không'] & ml['lấp_lửng'] & ap['thấp'] & fs['cao'], dr['trung_bình']))
    # thấp+thấp: thiếu ngủ → trung_bình, ngủ BT/đủ → thấp
    rules.append(ctrl.Rule(st_in['không'] & ml['lấp_lửng'] & ap['thấp'] & fs['thấp'] & slp['thiếu_ngủ'], dr['trung_bình']))
    rules.append(ctrl.Rule(st_in['không'] & ml['lấp_lửng'] & ap['thấp'] & fs['thấp'] & (slp['bình_thường'] | slp['đủ_giấc']), dr['thấp']))
    # thấp+TB: thiếu ngủ → trung_bình, ngủ BT/đủ → thấp
    rules.append(ctrl.Rule(st_in['không'] & ml['lấp_lửng'] & ap['thấp'] & fs['trung_bình'] & slp['thiếu_ngủ'], dr['trung_bình']))
    rules.append(ctrl.Rule(st_in['không'] & ml['lấp_lửng'] & ap['thấp'] & fs['trung_bình'] & (slp['bình_thường'] | slp['đủ_giấc']), dr['thấp']))
    # TB+thấp: thiếu ngủ → trung_bình, ngủ BT/đủ → thấp
    rules.append(ctrl.Rule(st_in['không'] & ml['lấp_lửng'] & ap['trung_bình'] & fs['thấp'] & slp['thiếu_ngủ'], dr['trung_bình']))
    rules.append(ctrl.Rule(st_in['không'] & ml['lấp_lửng'] & ap['trung_bình'] & fs['thấp'] & (slp['bình_thường'] | slp['đủ_giấc']), dr['thấp']))

    # ================================================================
    #  NHÓM D: KHÔNG TỰ TỬ — ML THẤP  →  nghiêng về an toàn
    #  Thiếu ngủ vẫn đẩy lên 1 bậc
    # ================================================================
    # ML thấp + cả hai thấp: thiếu ngủ → thấp, BT/đủ → rất thấp
    rules.append(ctrl.Rule(st_in['không'] & ml['thấp'] & ap['thấp'] & fs['thấp'] & slp['thiếu_ngủ'], dr['thấp']))
    rules.append(ctrl.Rule(st_in['không'] & ml['thấp'] & ap['thấp'] & fs['thấp'] & (slp['bình_thường'] | slp['đủ_giấc']), dr['rất_thấp']))
    # ML thấp + cả hai TB: thiếu ngủ → trung_bình, BT/đủ → thấp
    rules.append(ctrl.Rule(st_in['không'] & ml['thấp'] & ap['trung_bình'] & fs['trung_bình'] & slp['thiếu_ngủ'], dr['trung_bình']))
    rules.append(ctrl.Rule(st_in['không'] & ml['thấp'] & ap['trung_bình'] & fs['trung_bình'] & (slp['bình_thường'] | slp['đủ_giấc']), dr['thấp']))
    # ML thấp + học tập cao + tài chính cao → trung bình (yếu tố stress vẫn cao)
    rules.append(ctrl.Rule(st_in['không'] & ml['thấp'] & ap['cao'] & fs['cao'], dr['trung_bình']))
    # ML thấp + thấp+TB: thiếu ngủ → thấp, BT/đủ → rất thấp
    rules.append(ctrl.Rule(st_in['không'] & ml['thấp'] & ap['thấp'] & fs['trung_bình'] & slp['thiếu_ngủ'], dr['thấp']))
    rules.append(ctrl.Rule(st_in['không'] & ml['thấp'] & ap['thấp'] & fs['trung_bình'] & (slp['bình_thường'] | slp['đủ_giấc']), dr['rất_thấp']))
    # ML thấp + TB+thấp: thiếu ngủ → thấp, BT/đủ → rất thấp
    rules.append(ctrl.Rule(st_in['không'] & ml['thấp'] & ap['trung_bình'] & fs['thấp'] & slp['thiếu_ngủ'], dr['thấp']))
    rules.append(ctrl.Rule(st_in['không'] & ml['thấp'] & ap['trung_bình'] & fs['thấp'] & (slp['bình_thường'] | slp['đủ_giấc']), dr['rất_thấp']))
    # ML thấp + cao+thấp: thiếu ngủ → trung_bình, BT/đủ → thấp
    rules.append(ctrl.Rule(st_in['không'] & ml['thấp'] & ap['cao'] & fs['thấp'] & slp['thiếu_ngủ'], dr['trung_bình']))
    rules.append(ctrl.Rule(st_in['không'] & ml['thấp'] & ap['cao'] & fs['thấp'] & (slp['bình_thường'] | slp['đủ_giấc']), dr['thấp']))
    # ML thấp + thấp+cao: thiếu ngủ → trung_bình, BT/đủ → thấp
    rules.append(ctrl.Rule(st_in['không'] & ml['thấp'] & ap['thấp'] & fs['cao'] & slp['thiếu_ngủ'], dr['trung_bình']))
    rules.append(ctrl.Rule(st_in['không'] & ml['thấp'] & ap['thấp'] & fs['cao'] & (slp['bình_thường'] | slp['đủ_giấc']), dr['thấp']))
    # ML thấp + cao+TB: thiếu ngủ → trung_bình, BT/đủ → thấp
    rules.append(ctrl.Rule(st_in['không'] & ml['thấp'] & ap['cao'] & fs['trung_bình'] & slp['thiếu_ngủ'], dr['trung_bình']))
    rules.append(ctrl.Rule(st_in['không'] & ml['thấp'] & ap['cao'] & fs['trung_bình'] & (slp['bình_thường'] | slp['đủ_giấc']), dr['thấp']))
    # ML thấp + TB+cao: thiếu ngủ → trung_bình, BT/đủ → thấp
    rules.append(ctrl.Rule(st_in['không'] & ml['thấp'] & ap['trung_bình'] & fs['cao'] & slp['thiếu_ngủ'], dr['trung_bình']))
    rules.append(ctrl.Rule(st_in['không'] & ml['thấp'] & ap['trung_bình'] & fs['cao'] & (slp['bình_thường'] | slp['đủ_giấc']), dr['thấp']))

    return ctrl.ControlSystem(rules)

# ============================================================
# GIẢI THÍCH NGÔN NGỮ TỰ NHIÊN
# ============================================================
def generate_natural_explanation(risk_score, academic, financial, sleep_hours, ml_prob, has_suicidal):
    """Tạo giải thích bằng ngôn ngữ đời thường, thân thiện."""
    parts = []

    # Mở đầu theo mức rủi ro
    if risk_score >= 80:
        parts.append("Dựa trên những thông tin bạn cung cấp, hệ thống nhận thấy có **nhiều yếu tố** đáng lo ngại liên quan đến sức khỏe tinh thần của bạn.")
    elif risk_score >= 60:
        parts.append("Dựa trên những thông tin bạn cung cấp, hệ thống nhận thấy **một số yếu tố** cần được quan tâm về mặt sức khỏe tinh thần.")
    elif risk_score >= 30:
        parts.append("Dựa trên những thông tin bạn cung cấp, hệ thống nhận thấy bạn đang ở mức rủi ro **trung bình**. Một số yếu tố nên được theo dõi thêm.")
    else:
        parts.append("Dựa trên những thông tin bạn cung cấp, hệ thống nhận thấy bạn đang ở mức rủi ro **thấp**. Các yếu tố sức khỏe tinh thần của bạn nhìn chung ổn định.")

    # Yếu tố nguy hiểm: suy nghĩ tự tử
    if has_suicidal:
        parts.append("\nViệc bạn chia sẻ rằng mình **từng có suy nghĩ tự tử** là thông tin rất quan trọng. Đây là dấu hiệu cho thấy bạn cần được lắng nghe và hỗ trợ từ người có chuyên môn. Hãy nhớ rằng việc tìm kiếm sự giúp đỡ là điều hoàn toàn bình thường và dũng cảm.")

    # Phân tích XGBoost
    parts.append(f"\n**Phân tích từ mô hình Machine Learning (XGBoost):**")
    if ml_prob >= 0.85:
        parts.append(f"Mô hình ML đánh giá xác suất trầm cảm ở mức **cao ({ml_prob:.0%})**. Nhiều yếu tố tổng hợp từ câu trả lời của bạn đang cho thấy nguy cơ rõ ràng.")
    elif ml_prob >= 0.65:
        parts.append(f"Mô hình ML đánh giá xác suất trầm cảm ở mức **lấp lửng ({ml_prob:.0%})** — chưa rõ ràng theo một hướng cụ thể. Hệ thống luật mờ sẽ kết hợp với các yếu tố khác để đưa ra đánh giá chính xác hơn.")
    else:
        parts.append(f"Mô hình ML đánh giá xác suất trầm cảm ở mức **thấp ({ml_prob:.0%})**. Các yếu tố nhìn chung không cho thấy nguy cơ cao.")

    # Phân tích các yếu tố
    reasons = []
    if academic >= 3.5:
        reasons.append("mức áp lực học tập bạn đang gánh chịu khá nặng nề")
    elif academic >= 2:
        reasons.append("áp lực học tập của bạn ở mức vừa phải")

    if financial >= 3.5:
        reasons.append("áp lực tài chính đang là một gánh nặng đáng kể")
    elif financial >= 2:
        reasons.append("tình hình tài chính có phần căng thẳng nhưng vẫn kiểm soát được")

    if sleep_hours >= 7.5:
        reasons.append("bạn đang ngủ đủ giấc — đây là một yếu tố bảo vệ rất tốt cho sức khỏe tinh thần")
    elif sleep_hours >= 5.5:
        reasons.append("giấc ngủ của bạn ở mức trung bình, nên cố gắng ngủ đủ 7-8 tiếng mỗi đêm")
    else:
        reasons.append("bạn đang thiếu ngủ nghiêm trọng — đây là yếu tố làm tăng nguy cơ các vấn đề về sức khỏe tinh thần")

    if reasons:
        parts.append("\n**Yếu tố ảnh hưởng:**")
        combined = "Câu trả lời của bạn cho thấy " + ", ".join(reasons) + "."
        parts.append(combined)

        if risk_score >= 30:
            parts.append("Sự kết hợp của các yếu tố này cùng với kết quả từ mô hình ML đã khiến hệ thống luật mờ đưa ra mức đánh giá hiện tại.")

    return "\n\n".join(parts)

# ============================================================
# XÂY DỰNG INPUT DATAFRAME
# ============================================================
def build_input_df(model, input_dict, profession):
    try:
        feature_names = model.get_booster().feature_names
    except:
        try:
            feature_names = model.feature_names_in_.tolist()
        except:
            feature_names = None

    if feature_names:
        row = {}
        for col in feature_names:
            if col in input_dict:
                row[col] = input_dict[col]
            elif col.startswith("Profession_"):
                p_name = col.replace("Profession_", "")
                row[col] = 1 if (profession == p_name) else 0
            else:
                row[col] = 0
        df = pd.DataFrame([row])
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        return df
    else:
        df = pd.DataFrame([input_dict])
        for p in PROFESSIONS:
            col_name = f"Profession_{p}"
            if col_name not in df.columns:
                df[col_name] = 1 if (profession == p) else 0
        return df


# ============================================================
# GIAO DIỆN CHÍNH — THIẾT KẾ MỚI (không sidebar, không checkbox)
# ============================================================
def main():
    st.set_page_config(
        page_title="Sàng lọc sức khỏe tinh thần sinh viên",
        layout="centered"
    )

    # ========== CUSTOM CSS ==========
    st.markdown("""
    <style>
    /* Hero section */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
        line-height: 1.3;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #4a5568;
        max-width: 640px;
        margin: 0 auto;
        line-height: 1.7;
    }
    .privacy-badge {
        display: inline-block;
        background: #f0fff4;
        color: #276749;
        padding: 0.45rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        border: 1px solid #c6f6d5;
        margin-top: 1rem;
    }

    /* Step cards */
    .step-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1.1rem 1rem;
        border: 1px solid #e2e8f0;
        text-align: center;
        height: 100%;
        margin-bottom: 0.75rem;
    }

    /* Mobile: stack cards vertically with clear spacing */
    @media (max-width: 768px) {
        .step-card {
            margin-bottom: 1rem;
        }
    }
    .step-card h4 {
        margin: 0 0 0.3rem 0;
        color: #2d3748;
        font-size: 0.95rem;
    }
    .step-card p {
        margin: 0;
        color: #718096;
        font-size: 0.85rem;
        line-height: 1.5;
    }

    /* Disclaimer */
    .disclaimer-text {
        font-size: 0.8rem;
        color: #718096;
        text-align: center;
        margin-top: 0.5rem;
        line-height: 1.5;
    }
    </style>
    """, unsafe_allow_html=True)

    # ========== HERO / GIỚI THIỆU ==========
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">Công cụ Sàng lọc<br>Sức khỏe Tinh thần Sinh viên</h1>
        <p class="hero-subtitle">
            Ứng dụng hỗ trợ sinh viên tự đánh giá sớm các yếu tố rủi ro liên quan đến trầm cảm.
            Kết hợp mô hình <strong>Machine Learning (XGBoost)</strong> và
            hệ thống <strong>Luật Mờ (Fuzzy Logic)</strong> để đưa ra kết quả
            đa chiều và đáng tin cậy hơn so với một mô hình đơn lẻ.
        </p>
        <div class="privacy-badge">Dữ liệu chỉ xử lý tạm thời trong trình duyệt — không lưu trữ</div>
    </div>
    """, unsafe_allow_html=True)

    # === Quy trình 3 bước ===
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="step-card">
            <h4>Bước 1 — Nhập liệu</h4>
            <p>Bạn điền thông tin về học tập, tài chính, giấc ngủ và sức khỏe tinh thần</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="step-card">
            <h4>Bước 2 — ML phân tích</h4>
            <p>Mô hình XGBoost phân tích tổng hợp và dự đoán xác suất trầm cảm ban đầu</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="step-card">
            <h4>Bước 3 — Luật Mờ</h4>
            <p>Hệ thống Fuzzy Logic tổng hợp nhiều chiều và đưa ra đánh giá cuối cùng</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <p class="disclaimer-text">
        Công cụ này <strong>không phải chẩn đoán y tế</strong>. Kết quả chỉ mang tính <strong>tham khảo</strong>
        và không thay thế ý kiến của chuyên gia tâm lý.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ========== FORM NHẬP LIỆU (trên main content, không sidebar) ==========
    st.subheader("Nhập thông tin của bạn")
    st.caption("Điền các thông tin bên dưới để hệ thống đánh giá. Dữ liệu không được lưu trữ hay gửi đi bất kỳ đâu.")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Thông tin cá nhân**")
        gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
        age = st.number_input("Tuổi", min_value=15, max_value=60, value=20)
        profession = st.selectbox("Nghề nghiệp / Ngành", PROFESSIONS)
        cgpa = st.slider("Điểm GPA", 0.0, 10.0, 7.0, 0.1)

    with col_right:
        st.markdown("**Yếu tố rủi ro chính**")
        academic_pressure = st.slider("Mức áp lực học tập", 1, 5, 3, 1)
        financial_stress  = st.slider("Mức áp lực tài chính", 1, 5, 3, 1)
        sleep_choice = st.selectbox("Thời gian ngủ trung bình", list(SLEEP_MAP.keys()))
        sleep_val = SLEEP_MAP[sleep_choice]
        suicidal = st.selectbox("Bạn có từng có suy nghĩ tự tử?", ["Không", "Có"])

    # --- Thông tin bổ sung (tuỳ chọn) ---
    use_extra = st.checkbox("Bổ sung thêm thông tin (tuỳ chọn)", value=False)
    if use_extra:
        with st.expander("Thông tin bổ sung", expanded=True):
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                work_pressure = st.slider("Áp lực công việc", 0.0, 5.0, 0.0, 1.0)
                study_satisfaction = st.slider("Mức hài lòng học tập", 0.0, 5.0, 3.0, 1.0)
                job_satisfaction = st.slider("Mức hài lòng công việc", 0.0, 5.0, 0.0, 1.0)
            with col_e2:
                work_study_hours = st.slider("Giờ học/làm việc mỗi ngày", 0.0, 12.0, 6.0, 1.0)
                dietary = st.selectbox("Chế độ ăn", list(DIET_MAP.keys()))
    else:
        # Giá trị trung lập — không ảnh hưởng đáng kể đến kết quả model
        work_pressure = 0.0
        study_satisfaction = 3.0
        job_satisfaction = 0.0
        work_study_hours = 6.0
        dietary = "Bình thường"

    st.markdown("")

    # ========== NÚT SÀNG LỌC ==========
    if st.button("Bắt đầu sàng lọc", type="primary", use_container_width=True):
        try:
            all_models = load_models()
            fuzzy_ctrl = build_fuzzy_system()
            threshold = all_models['threshold']

            gender_val = 1 if gender == "Nam" else 0
            suicidal_val = 1 if suicidal == "Có" else 0
            dietary_val = DIET_MAP[dietary]

            # ── BƯỚC 1: Chọn mô hình XGBoost phù hợp ──
            if suicidal_val == 1:
                xgb_model = all_models['xgb']
            else:
                xgb_model = all_models['xgb_ns']

            input_dict = {
                'Gender': gender_val, 'Age': age,
                'Academic Pressure': academic_pressure,
                'Work Pressure': work_pressure, 'CGPA': cgpa,
                'Study Satisfaction': study_satisfaction,
                'Job Satisfaction': job_satisfaction,
                'Sleep Duration': sleep_val,
                'Dietary Habits': dietary_val,
                'Have you ever had suicidal thoughts ?': suicidal_val,
                'Work/Study Hours': work_study_hours,
                'Financial Stress': financial_stress,
                'Family History of Mental Illness': 0,
            }

            # Model _ns không có cột suicidal → loại bỏ
            if suicidal_val == 0:
                input_dict_ns = {k: v for k, v in input_dict.items() if k != 'Have you ever had suicidal thoughts ?'}
                xgb_input = build_input_df(xgb_model, input_dict_ns, profession)
            else:
                xgb_input = build_input_df(xgb_model, input_dict, profession)

            # ── BƯỚC 2: XGBoost predict_proba → xác suất ML ──
            xgb_prob = xgb_model.predict_proba(xgb_input)[0][1]

            # ── BƯỚC 3: Đẩy 5 giá trị vào Hệ thống Luật Mờ ──
            try:
                sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)
                sim.input['academic_pressure'] = np.clip(academic_pressure, 1, 5)
                sim.input['financial_stress']  = np.clip(financial_stress, 1, 5)
                sim.input['suicidal_thoughts'] = float(suicidal_val)
                sim.input['sleep_duration']    = np.clip(sleep_val, 3, 10)
                sim.input['ml_probability']    = np.clip(xgb_prob, 0, 1)
                sim.compute()
                fuzzy_risk = sim.output['depression_risk']
            except Exception as fuzzy_err:
                st.warning(f"Luật mờ không kích hoạt được (fallback sang ML): {fuzzy_err}")
                fuzzy_risk = xgb_prob * 100

            # ── BƯỚC 4: Xác định KẾT LUẬN CUỐI CÙNG ──
            if fuzzy_risk >= 85:
                level = "Rất cao"
                conclusion = "Có nguy cơ trầm cảm CAO"
            elif fuzzy_risk >= 65:
                level = "Cao"
                conclusion = "Có nguy cơ trầm cảm"
            elif fuzzy_risk >= 30:
                level = "Trung bình"
                conclusion = "Có một số yếu tố cần theo dõi"
            else:
                level = "Thấp"
                conclusion = "Nguy cơ trầm cảm thấp"

            # ============================================================
            # HIỂN THỊ KẾT QUẢ
            # ============================================================
            st.markdown("---")

            # === KẾT LUẬN CUỐI CÙNG (nổi bật) ===
            st.subheader("Kết luận cuối cùng")

            # Chọn màu nền theo mức rủi ro — đồng bộ với level
            if fuzzy_risk >= 85:
                bg_color, border_color, text_color = '#fee2e2', '#dc2626', '#991b1b'
            elif fuzzy_risk >= 65:
                bg_color, border_color, text_color = '#ffedd5', '#ea580c', '#9a3412'
            elif fuzzy_risk >= 30:
                bg_color, border_color, text_color = '#fef9c3', '#eab308', '#854d0e'
            else:
                bg_color, border_color, text_color = '#dcfce7', '#16a34a', '#166534'

            st.markdown(
                f"""
                <div style="
                    padding: 20px;
                    border-radius: 12px;
                    background: {bg_color};
                    border-left: 6px solid {border_color};
                    margin-bottom: 20px;
                ">
                    <h2 style="margin:0; color: {text_color};">
                        {conclusion}
                    </h2>
                    <p style="margin-top:8px; font-size:16px; color: #374151;">
                        Mức rủi ro: <strong>{level}</strong> — Điểm đánh giá: <strong>{fuzzy_risk:.1f}/100</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # === QUY TRÌNH — CHI TIẾT 4 BƯỚC ===
            st.subheader("Chi tiết quy trình đánh giá")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown("**Bước 1: Đầu vào**")
                st.write(f"Học tập: **{academic_pressure}** {'(Cao)' if academic_pressure >= 3.5 else '(TB)' if academic_pressure >= 2.5 else '(Thấp)'}")
                st.write(f"Tài chính: **{financial_stress}** {'(Cao)' if financial_stress >= 3.5 else '(TB)' if financial_stress >= 2.5 else '(Thấp)'}")
                st.write(f"Tự tử: **{'Có' if suicidal_val else 'Không'}**")

            with col2:
                st.markdown("**Bước 2: XGBoost**")
                st.metric("Xác suất ML", f"{xgb_prob:.2%}")
                if xgb_prob >= 0.85:
                    st.caption("→ ML đánh giá: **Cao**")
                elif xgb_prob >= 0.65:
                    st.caption("→ ML đánh giá: **Lấp lửng**")
                else:
                    st.caption("→ ML đánh giá: **Thấp**")

            with col3:
                st.markdown("**Bước 3: Luật Mờ**")
                st.write("5 đầu vào Fuzzy:")
                st.code(
                    f"Học tập    = {academic_pressure}\n"
                    f"Tài chính  = {financial_stress}\n"
                    f"Tự tử     = {'Có' if suicidal_val else 'Không'}\n"
                    f"Giấc ngủ  = {sleep_val}h\n"
                    f"ML prob   = {xgb_prob:.2f}",
                    language=None
                )

            with col4:
                st.markdown("**Bước 4: Kết luận**")
                st.metric("Điểm Fuzzy", f"{fuzzy_risk:.1f}/100")
                st.write(f"**{conclusion}**")

            # === CẢNH BÁO NHẠY CẢM ===
            if suicidal_val == 1 or fuzzy_risk >= 60:
                st.error(
                    "Kết quả cho thấy bạn có thể đang gặp khó khăn về mặt tinh thần. "
                    "Xin hãy nhớ rằng bạn không đơn độc, và việc tìm kiếm sự giúp đỡ là điều hoàn toàn tốt đẹp."
                )
                st.markdown(
                    "**Nếu bạn cần ai đó lắng nghe, hãy liên hệ:**\n"
                    "- Tổng đài tư vấn tâm lý: **1900 1267** (miễn phí, 24/7)\n"
                    "- Hoặc nói chuyện với một người bạn tin tưởng, thầy cô, hoặc người thân"
                )
            elif fuzzy_risk >= 30:
                st.warning(
                    "Một số yếu tố trong câu trả lời của bạn cho thấy bạn nên chú ý hơn đến sức khỏe tinh thần. "
                    "Đây không phải là chẩn đoán, nhưng là dấu hiệu bạn nên theo dõi thêm."
                )
            else:
                st.success(
                    "Dựa trên thông tin bạn cung cấp, các chỉ số sức khỏe tinh thần của bạn nhìn chung tích cực. "
                    "Hãy tiếp tục duy trì lối sống lành mạnh."
                )

            # === GIẢI THÍCH NGÔN NGỮ TỰ NHIÊN ===
            st.markdown("---")
            st.subheader("Lý do hệ thống đưa ra đánh giá này")
            explanation = generate_natural_explanation(
                fuzzy_risk, academic_pressure, financial_stress, sleep_val, xgb_prob,
                has_suicidal=(suicidal_val == 1)
            )
            st.markdown(explanation)

            # === KHUYẾN CÁO ===
            st.markdown("---")
            st.caption(
                "Lưu ý quan trọng: Công cụ này chỉ là phương tiện sàng lọc ban đầu, "
                "đo lường các yếu tố rủi ro dựa trên thông tin bạn tự báo cáo. "
                "Nó KHÔNG THỂ thay thế đánh giá lâm sàng của chuyên gia tâm lý hoặc bác sĩ. "
                "Nếu bạn hoặc người thân đang gặp khó khăn, hãy tìm đến sự hỗ trợ chuyên môn."
            )
            st.caption(
                "Cam kết bảo mật: Toàn bộ dữ liệu chỉ xử lý tạm thời trong bộ nhớ, "
                "không lưu trữ, không gửi đến máy chủ bên ngoài."
            )

        except Exception as e:
            st.error(f"Lỗi: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
