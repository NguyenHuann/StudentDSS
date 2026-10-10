import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from recommendation.cbf import recommend_one

st.set_page_config(page_title="Hệ thống Cảnh báo Sớm (DSS)", page_icon="🎓", layout="wide")

# 1. KHỞI TẠO MÔ HÌNH VÀ CẤU HÌNH
@st.cache_resource
def load_models():
    rf_model = joblib.load('models/rf_model1.pkl')
    xgb_model = joblib.load('models/xgb_model2.pkl')
    features = joblib.load('models/feature_columns.pkl')
    return rf_model, xgb_model, features


rf_model1, xgb_model2, ML_FEATURES = load_models()
CAUTION_THRESHOLD = 0.45

# TỪ ĐIỂN VIỆT HÓA ĐẶC TRƯNG & YẾU TỐ KÍCH HOẠT
FEATURE_MAP = {
    '1st_sem_enrolled': 'Số môn đăng ký',
    '1st_sem_evaluations': 'Số môn có đánh giá',
    '1st_sem_without_evaluations': 'Số môn bỏ thi',
    '1st_sem_approved': 'Số môn đạt (Qua môn)',
    '1st_sem_failed': 'Số môn rớt',
    '1st_sem_grade': 'Điểm trung bình kỳ 1',
    '1st_sem_pass_rate': 'Tỷ lệ qua môn',
    'failed_burden': 'Gánh nặng rớt môn',
    'ghosting_rate': 'Tỷ lệ bỏ thi',
    'Target': 'Trạng thái thực tế',
    'Dự đoán DSS': 'Dự đoán của hệ thống',
    # Các yếu tố kích hoạt CBF (Semantic dimensions)
    'low_grade': 'Điểm số thấp',
    'low_pass_rate': 'Tỷ lệ qua môn thấp',
    'evaluation_issue': 'Vấn đề tham gia thi',
    'workload_gap': 'Khoảng cách khối lượng học tập',
    'high_performance': 'Thành tích xuất sắc'
}

# TỪ ĐIỂN VIỆT HÓA DANH MỤC GIẢI PHÁP
CATEGORY_MAP = {
    'Academic Support': 'Hỗ trợ Học thuật',
    'Course Recovery': 'Khắc phục Học phần',
    'Workload Management': 'Quản lý Khối lượng',
    'Academic Advising': 'Tư vấn Học tập',
    'Study Planning': 'Lập Kế hoạch Học tập',
    'Peer Learning': 'Học tập Nhóm',
    'Progress Monitoring': 'Theo dõi Tiến độ',
    'Study Maintenance': 'Duy trì Lộ trình',
    'Academic Enrichment': 'Nâng cao Học thuật',
    'Remedial Study': 'Ôn tập Củng cố'
}


# 2. HÀM XỬ LÝ & DỰ ĐOÁN
def preprocess_for_ml(df_input):
    df = df_input.copy()
    if 'Target' in df.columns:
        df = df.drop(columns=['Target'])

    df['1st_sem_pass_rate'] = np.where(df['1st_sem_evaluations'] == 0, 0,
                                       df['1st_sem_approved'] / df['1st_sem_evaluations'])
    df['failed_burden'] = np.where(df['1st_sem_enrolled'] == 0, 0, df['1st_sem_failed'] / df['1st_sem_enrolled'])
    df['ghosting_rate'] = np.where(df['1st_sem_enrolled'] == 0, 0,
                                   df['1st_sem_without_evaluations'] / df['1st_sem_enrolled'])

    return df[ML_FEATURES]


def predict_single(student_dict):
    X_ml = preprocess_for_ml(pd.DataFrame([student_dict]))
    stage1_pred = rf_model1.predict(X_ml)[0]

    if stage1_pred == 'Safe':
        return 'Safe', 1.0, 0.0, 0.0, X_ml

    probs = xgb_model2.predict_proba(X_ml)[0]
    final_pred = 'Caution' if probs[0] >= CAUTION_THRESHOLD else 'Dropout'
    return final_pred, 0.0, probs[0], probs[1], X_ml


def predict_batch(df_raw):
    X_ml = preprocess_for_ml(df_raw)
    preds1 = rf_model1.predict(X_ml)

    final_preds = np.array(preds1, dtype=object)
    at_risk_mask = (preds1 == 'At-Risk')

    if np.sum(at_risk_mask) > 0:
        X_at_risk = X_ml[at_risk_mask]
        probs2 = xgb_model2.predict_proba(X_at_risk)
        stage2_preds = np.where(probs2[:, 0] >= CAUTION_THRESHOLD, 'Caution', 'Dropout')
        final_preds[at_risk_mask] = stage2_preds

    df_result = df_raw.copy()
    df_result['Dự đoán DSS'] = final_preds
    return df_result

# 3. QUẢN LÝ DỮ LIỆU (SESSION STATE)
if 'student_df' not in st.session_state:
    try:
        st.session_state['student_df'] = pd.read_csv('dataset/dataset_preprocessed.csv')
    except:
        st.session_state['student_df'] = pd.DataFrame()
# 4. GIAO DIỆN THANH BÊN (SIDEBAR)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3281/3281329.png", width=100)
    st.title("Menu Chức Năng")

    menu = st.radio(
        "Lựa chọn thao tác:",
        ["Báo cáo Tổng quan", "Phân tích Cá nhân", "Nhập liệu Thủ công"]
    )

    st.markdown("---")
    st.subheader("📂 Tải lên Danh sách mới")
    uploaded_file = st.file_uploader("Định dạng CSV", type=["csv"])
    if uploaded_file is not None:
        if st.button("Cập nhật Dữ liệu"):
            st.session_state['student_df'] = pd.read_csv(uploaded_file)
            st.success("Tải dữ liệu thành công! Hãy vào 'Báo cáo Tổng quan' để xem.")

# 5. MÀN HÌNH CHÍNH (MAIN CONTENT)
df_current = st.session_state['student_df']

if menu == "Báo cáo Tổng quan":
    st.header("Báo cáo Tổng quan Danh sách Sinh viên")
    if df_current.empty:
        st.warning("Hệ thống chưa có dữ liệu. Vui lòng tải file CSV lên từ menu bên trái.")
    else:
        with st.spinner("Đang chạy mô hình dự đoán cho toàn bộ sinh viên..."):
            df_scored = predict_batch(df_current)

        total = len(df_scored)
        safe_cnt = len(df_scored[df_scored['Dự đoán DSS'] == 'Safe'])
        caut_cnt = len(df_scored[df_scored['Dự đoán DSS'] == 'Caution'])
        drop_cnt = len(df_scored[df_scored['Dự đoán DSS'] == 'Dropout'])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tổng số Sinh viên", total)
        col2.metric("🟢 An toàn (Safe)", safe_cnt, f"{(safe_cnt / total) * 100:.1f}%")
        col3.metric("🟡 Cảnh báo (Caution)", caut_cnt, f"{(caut_cnt / total) * 100:.1f}%", delta_color="off")
        col4.metric("🔴 Bỏ học (Dropout)", drop_cnt, f"{(drop_cnt / total) * 100:.1f}%", delta_color="inverse")

        st.markdown("---")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Tỷ lệ Phân bổ Rủi ro")
            fig, ax = plt.subplots(figsize=(5, 5))
            colors = {'Safe': '#2ecc71', 'Caution': '#f1c40f', 'Dropout': '#e74c3c'}
            counts = df_scored['Dự đoán DSS'].value_counts()
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=[colors.get(x, '#333') for x in counts.index])
            st.pyplot(fig)

        with c2:
            st.subheader("Danh sách Sinh viên cần chú ý")
            df_risk = df_scored[df_scored['Dự đoán DSS'].isin(['Caution', 'Dropout'])].copy()
            df_risk_display = df_risk[
                ['Dự đoán DSS', '1st_sem_enrolled', '1st_sem_approved', '1st_sem_failed',
                 '1st_sem_grade']].rename(columns=FEATURE_MAP)
            # Format float về 2 chữ số thập phân trên dataframe
            st.dataframe(df_risk_display.style.format(precision=2), height=300)

elif menu == "Phân tích Cá nhân":
    st.header("Phân tích & Đề xuất Kế hoạch")

    if df_current.empty:
        st.warning("Hệ thống chưa có dữ liệu.")
    else:
        df_scored = predict_batch(df_current)

        st.markdown("##### Lọc Danh sách Sinh viên")
        col_filter1, col_filter2 = st.columns([1, 2])
        with col_filter1:
            group_filter = st.selectbox(
                "Lọc theo dự đoán hệ thống:",
                ["Tất cả", "Safe (An toàn)", "Caution (Cảnh báo)", "Dropout (Nguy cơ bỏ học)"]
            )

        if group_filter != "Tất cả":
            filter_value = group_filter.split(" ")[0]
            filtered_df = df_scored[df_scored['Dự đoán DSS'] == filter_value]
        else:
            filtered_df = df_scored

        student_indices = filtered_df.index.tolist()

        if not student_indices:
            st.info(f"Không có sinh viên nào thuộc nhóm {group_filter}.")
        else:
            with col_filter2:
                selected_idx = st.selectbox(
                    "Chọn Mã Sinh viên (Index) để xem chi tiết:",
                    student_indices,
                    format_func=lambda x: f"Sinh viên #{x}"
                )

            student_row = df_current.loc[selected_idx].to_dict()

            st.markdown("---")
            st.subheader(f"Thông tin hồ sơ: Sinh viên #{selected_idx}")
            info_cols = st.columns(4)

            idx_col = 0
            for col_name, val in student_row.items():
                vi_name = FEATURE_MAP.get(col_name, col_name)
                # Làm tròn 2 chữ số nếu giá trị là kiểu float
                if isinstance(val, float):
                    info_cols[idx_col % 4].write(f"**{vi_name}:** {val:.2f}")
                else:
                    info_cols[idx_col % 4].write(f"**{vi_name}:** {val}")
                idx_col += 1

            st.markdown("---")

            pred_state, prob_safe, prob_caut, prob_drop, X_ml = predict_single(student_row)

            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("Nhận định từ AI (RF + XGBoost)")
                if pred_state == 'Safe':
                    st.success("AN TOÀN (SAFE)")
                elif pred_state == 'Caution':
                    st.warning("CẦN THEO DÕI (CAUTION)")
                else:
                    st.error("RỦI RO CAO (DROPOUT)")
                st.write(f"- Tỷ lệ Safe: **{prob_safe * 100:.1f}%**")
                st.write(f"- Tỷ lệ Caution: **{prob_caut * 100:.1f}%**")
                st.write(f"- Tỷ lệ Dropout: **{prob_drop * 100:.1f}%**")

            with c2:
                st.subheader("Nguyên nhân cốt lõi (SHAP Analysis)")
                model_to_explain = rf_model1 if pred_state == 'Safe' else xgb_model2
                explainer = shap.TreeExplainer(model_to_explain)
                shap_values = explainer.shap_values(X_ml)

                X_ml_display = X_ml.rename(columns=FEATURE_MAP)

                fig, ax = plt.subplots(figsize=(7, 3))
                shap.summary_plot(shap_values, X_ml_display, plot_type="bar", show=False, max_display=5)
                st.pyplot(fig)

            st.subheader("Đề xuất Kế hoạch Hỗ trợ (CBF Engine)")
            recs_df, profile = recommend_one(student_row, top_n=3)

            if recs_df.iloc[0]['low_confidence']:
                st.warning("Độ tin cậy của đề xuất thấp. Khuyến nghị gặp trực tiếp Cố vấn học tập để đánh giá thêm.")

            for idx, row in recs_df.iterrows():
                with st.expander(f"#{row['rank']} - {row['item_name']} (Độ phù hợp: {row['similarity_score']:.2f})",
                                 expanded=True):
                    # Việt hóa category và các top matching dimensions
                    cat_vi = CATEGORY_MAP.get(row['category'], row['category'])

                    raw_dims = row['top_matching_dimensions'].split(', ')
                    dims_vi = [FEATURE_MAP.get(d, d) for d in raw_dims]
                    dims_str_vi = ", ".join(dims_vi)

                    st.write(f"**Danh mục:** {cat_vi} | **Yếu tố kích hoạt:** {dims_str_vi}")
                    st.write(f"**Hành động cụ thể:** {row['action_plan']}")

elif menu == "Nhập liệu Thủ công":
    st.header("Nhập liệu & Phân tích Sinh viên mới")

    with st.form("manual_input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            enrolled = st.number_input("Số môn đăng ký", min_value=0, value=6)
            approved = st.number_input("Số môn đạt", min_value=0, value=2)
        with col2:
            evaluations = st.number_input("Số môn có đánh giá", min_value=0, value=6)
            failed = st.number_input("Số môn rớt", min_value=0, value=4)
        with col3:
            without_evals = st.number_input("Số môn bỏ thi", min_value=0, value=0)
            grade = st.number_input("Điểm trung bình (Hệ 20)", min_value=0.0, max_value=20.0, value=9.5)

        submitted = st.form_submit_button("Phân tích Dữ liệu", use_container_width=True)

    if submitted:
        student_input = {
            "1st_sem_enrolled": enrolled, "1st_sem_evaluations": evaluations,
            "1st_sem_without_evaluations": without_evals, "1st_sem_approved": approved,
            "1st_sem_failed": failed, "1st_sem_grade": grade
        }

        pred_state, prob_safe, prob_caut, prob_drop, X_ml = predict_single(student_input)

        st.markdown("---")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Kết quả Phân loại")
            if pred_state == 'Safe':
                st.success("AN TOÀN (SAFE)")
            elif pred_state == 'Caution':
                st.warning("CẦN THEO DÕI (CAUTION)")
            else:
                st.error("RỦI RO CAO (DROPOUT)")

        with c2:
            st.subheader("Giải thích SHAP")
            model_to_explain = rf_model1 if pred_state == 'Safe' else xgb_model2
            explainer = shap.TreeExplainer(model_to_explain)
            shap_values = explainer.shap_values(X_ml)

            X_ml_display = X_ml.rename(columns=FEATURE_MAP)

            fig, ax = plt.subplots(figsize=(7, 3))
            shap.summary_plot(shap_values, X_ml_display, plot_type="bar", show=False, max_display=4)
            st.pyplot(fig)

        st.subheader("Khuyến nghị Học tập")
        recs_df, profile = recommend_one(student_input, top_n=3)
        for idx, row in recs_df.iterrows():
            cat_vi = CATEGORY_MAP.get(row['category'], row['category'])
            st.info(
                f"**{row['rank']}. {row['item_name']}** ({cat_vi}) - Phù hợp: {row['similarity_score']:.2f}\n\n{row['action_plan']}")