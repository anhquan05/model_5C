# -*- coding: utf-8 -*-
"""
Ứng dụng Streamlit: Dự báo Rủi ro Tín dụng Khách hàng theo Mô hình 5C
Được chuyển đổi từ notebook Logistic Regression (4_7_qtrr.ipynb).
"""

import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# ============================================================
# 0) CẤU HÌNH TRANG (phải là lệnh Streamlit đầu tiên)
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="Dự báo Rủi ro Tín dụng - Mô hình 5C",
    page_icon="🏦",
)

# ============================================================
# 1) HẰNG SỐ & HÀM DÙNG CHUNG
# ============================================================
FEATURE_COLS = [
    "TC1", "TC2", "TC3", "TC4", "TC5",
    "NL1", "NL2", "NL3", "NL4",
    "DK1", "DK2", "DK3", "DK4", "DK5",
    "V1", "V2", "V3", "V4", "V5", "V6",
    "TS1", "TS2", "TS3", "TS4",
]
TARGET_COL = "PD"

GROUP_NAMES = {
    "TC": "Tư cách (Character)",
    "NL": "Năng lực (Capacity)",
    "DK": "Điều kiện (Condition)",
    "V": "Vốn (Capital)",
    "TS": "Tài sản đảm bảo (Collateral)",
}


def col_group(col: str) -> str:
    """Trả về tên nhóm 5C của một cột biến (TC1 -> TC, V6 -> V,...)."""
    for prefix in ["TC", "NL", "DK", "TS", "V"]:
        if col.startswith(prefix):
            return prefix
    return col


@st.cache_data
def load_data(file_bytes: bytes) -> pd.DataFrame:
    """Nạp dữ liệu CSV từ bytes (để cache_data có thể hash được)."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df


def validate_columns(df: pd.DataFrame):
    """Kiểm tra df có đủ các cột biến đầu vào & biến mục tiêu không."""
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    return missing


# ============================================================
# 2) SIDEBAR — VÙNG CẤU HÌNH
# ============================================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")

    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu khảo sát (CSV)",
        type=["csv"],
        help="Tệp CSV cần chứa 24 biến đầu vào (TC1-5, NL1-4, DK1-5, V1-6, TS1-4) "
        "và cột nhãn 'PD' (0 = không rủi ro, 1 = có rủi ro), giống cấu trúc tệp 5c.csv gốc.",
    )

    # Chỉ có 1 mô hình duy nhất trong notebook (Logistic Regression) -> không cần selectbox chọn mô hình

    st.subheader("Tham số mô hình AI")

    test_size = st.slider(
        "Tỷ lệ tập kiểm tra (test size)",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.05,
        help="Tỷ lệ dữ liệu dùng để kiểm định mô hình. Notebook gốc dùng 0.2 (20%).",
    )

    random_state = st.number_input(
        "Random state",
        min_value=0,
        max_value=9999,
        value=32,
        step=1,
        help="Hạt giống ngẫu nhiên khi chia tập train/test. Notebook gốc dùng 32.",
    )

    with st.expander("Tham số nâng cao (Logistic Regression)"):
        C_param = st.slider(
            "C (độ mạnh điều chuẩn, càng nhỏ càng regularize mạnh)",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.01,
            help="Notebook gốc không chỉnh tham số này (dùng mặc định scikit-learn = 1.0).",
        )
        max_iter = st.number_input(
            "Số vòng lặp tối đa (max_iter)",
            min_value=100,
            max_value=5000,
            value=100,
            step=100,
            help="Notebook gốc dùng giá trị mặc định của scikit-learn (100).",
        )
        solver = st.selectbox(
            "Solver",
            options=["lbfgs", "liblinear", "saga", "newton-cg"],
            index=0,
            help="Notebook gốc dùng solver mặc định của scikit-learn (lbfgs).",
        )

    st.divider()
    train_clicked = st.button(
        "🚀 Huấn luyện mô hình",
        type="primary",
        use_container_width=True,
    )

# ============================================================
# 3) HEADER — VÙNG ĐỊNH HƯỚNG
# ============================================================
st.title("🏦 Dự báo Rủi ro Tín dụng Khách hàng - Mô hình 5C")
st.caption(
    "Ứng dụng dùng thuật toán **Logistic Regression** để dự báo xác suất rủi ro tín dụng (PD) "
    "dựa trên 24 biến khảo sát thuộc 5 nhóm tiêu chí tín dụng cổ điển (5C): "
    "**T**ư cách, **N**ăng **l**ực, **Đ**iều **k**iện, **V**ốn, **T**ài **s**ản đảm bảo."
)

if uploaded_file is None:
    st.info("👈 Vui lòng tải lên tệp dữ liệu CSV ở thanh bên để bắt đầu.")
    st.stop()

file_bytes = uploaded_file.getvalue()
try:
    df = load_data(file_bytes)
except Exception as e:
    st.error(f"Không thể đọc tệp dữ liệu. Lỗi: {e}")
    st.stop()

missing_cols = validate_columns(df)
if missing_cols:
    st.error(
        "Tệp dữ liệu thiếu các cột bắt buộc sau: "
        + ", ".join(missing_cols)
        + ". Vui lòng kiểm tra lại cấu trúc tệp (cần đủ 24 biến 5C + cột 'PD')."
    )
    st.stop()

if df.empty:
    st.error("Tệp dữ liệu rỗng. Vui lòng tải lên tệp khác.")
    st.stop()

st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}**")
st.caption(f"Tệp có **{df.shape[0]}** dòng và **{df.shape[1]}** cột.")
st.divider()

# ============================================================
# 4) KHỐI HUẤN LUYỆN (chạy khi bấm nút ở sidebar)
# ============================================================
if train_clicked:
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state)
    )

    model = LogisticRegression(
        C=C_param, max_iter=int(max_iter), solver=solver
    )
    model.fit(X_train, Y_train)

    yhat_test = model.predict(X_test)
    proba_test = model.predict_proba(X_test)

    st.session_state["model"] = model
    st.session_state["feature_cols"] = FEATURE_COLS
    st.session_state["results"] = {
        "X_test": X_test,
        "Y_test": Y_test,
        "yhat_test": yhat_test,
        "proba_test": proba_test,
    }
    st.session_state["data_ranges"] = {
        col: {
            "min": int(df[col].min()),
            "max": int(df[col].max()),
            "median": int(df[col].median()),
        }
        for col in FEATURE_COLS
    }
    st.success("✅ Huấn luyện mô hình thành công! Xem kết quả ở các tab bên dưới.")

# ============================================================
# 5) CÁC TAB NỘI DUNG CHÍNH
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📋 Tổng quan dữ liệu",
        "📊 Trực quan hóa dữ liệu",
        "🧪 Kết quả huấn luyện & kiểm định",
        "🔮 Sử dụng mô hình",
    ]
)

# ------------------------------------------------------------
# TAB 1: TỔNG QUAN DỮ LIỆU
# ------------------------------------------------------------
with tab1:
    st.subheader("Kích thước dữ liệu")
    c1, c2, c3 = st.columns(3)
    c1.metric("Số dòng", df.shape[0])
    c2.metric("Số cột", df.shape[1])
    c3.metric("Dung lượng tệp (KB)", f"{len(file_bytes) / 1024:.1f}")

    st.subheader("Xem dữ liệu thô")
    with st.container(height=300):
        st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Thống kê mô tả các biến của mô hình")
    st.caption("Chỉ hiển thị 24 biến đầu vào (5C) và biến mục tiêu (PD).")
    st.dataframe(df[FEATURE_COLS + [TARGET_COL]].describe(), use_container_width=True)

# ------------------------------------------------------------
# TAB 2: TRỰC QUAN HÓA DỮ LIỆU
# ------------------------------------------------------------
with tab2:
    st.subheader("Phân phối biến mục tiêu (PD)")
    pd_counts = df[TARGET_COL].value_counts().sort_index().reset_index()
    pd_counts.columns = ["PD", "Số lượng"]
    pd_counts["PD"] = pd_counts["PD"].map({0: "0 - Không rủi ro", 1: "1 - Có rủi ro"})
    fig_target = px.bar(
        pd_counts, x="PD", y="Số lượng", color="PD",
        title="Phân phối lớp rủi ro tín dụng (PD)",
    )
    fig_target.update_layout(height=350)
    st.plotly_chart(fig_target, use_container_width=True)

    st.subheader("Trực quan hóa biến đầu vào (thang Likert 1-5)")
    default_vars = ["TC1", "NL1", "DK1", "V1"]
    selected_vars = st.multiselect(
        "Chọn tối đa 4 biến để trực quan hóa (mặc định: 1 biến đại diện mỗi nhóm)",
        options=FEATURE_COLS,
        default=default_vars,
        max_selections=4,
        help="Có 24 biến đầu vào nên chỉ hiển thị tối đa 4 biến cùng lúc, chọn biến bạn muốn xem.",
    )

    if selected_vars:
        cols_top = st.columns(2)
        cols_bottom = st.columns(2)
        grid_slots = cols_top + cols_bottom
        for i, var in enumerate(selected_vars[:4]):
            counts = df[var].value_counts().sort_index().reset_index()
            counts.columns = [var, "Số lượng"]
            fig = px.bar(
                counts, x=var, y="Số lượng",
                title=f"{var} - Nhóm {GROUP_NAMES[col_group(var)]}",
            )
            fig.update_layout(height=320)
            grid_slots[i].plotly_chart(fig, use_container_width=True)
    else:
        st.info("Vui lòng chọn ít nhất một biến để trực quan hóa.")

# ------------------------------------------------------------
# TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH
# ------------------------------------------------------------
with tab3:
    if "results" not in st.session_state:
        st.info("👈 Vui lòng bấm nút **'Huấn luyện mô hình'** ở thanh bên để xem kết quả.")
        st.stop()

    results = st.session_state["results"]
    Y_test = results["Y_test"]
    yhat_test = results["yhat_test"]
    proba_test = results["proba_test"]

    acc = accuracy_score(Y_test, yhat_test)
    prec = precision_score(Y_test, yhat_test, zero_division=0)
    rec = recall_score(Y_test, yhat_test, zero_division=0)
    f1 = f1_score(Y_test, yhat_test, zero_division=0)
    try:
        auc = roc_auc_score(Y_test, proba_test[:, 1])
    except ValueError:
        auc = float("nan")

    st.subheader("Chỉ số đánh giá mô hình (phân loại)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("Precision", f"{prec:.3f}")
    m3.metric("Recall", f"{rec:.3f}")
    m4.metric("F1-score", f"{f1:.3f}")
    m5.metric("ROC-AUC", f"{auc:.3f}" if not np.isnan(auc) else "N/A")

    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown("**Ma trận nhầm lẫn (Confusion Matrix)**")
        cm = confusion_matrix(Y_test, yhat_test)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Dự báo: 0", "Dự báo: 1"],
            y=["Thực tế: 0", "Thực tế: 1"],
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_roc:
        st.markdown("**Đường cong ROC**")
        try:
            fpr, tpr, _ = roc_curve(Y_test, proba_test[:, 1])
            fig_roc = px.area(
                x=fpr, y=tpr,
                labels={"x": "False Positive Rate", "y": "True Positive Rate"},
                title=f"ROC Curve (AUC = {auc:.3f})",
            )
            fig_roc.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
            fig_roc.update_layout(height=400)
            st.plotly_chart(fig_roc, use_container_width=True)
        except ValueError:
            st.warning("Không đủ dữ liệu để vẽ đường cong ROC.")

    st.markdown("**Báo cáo phân loại chi tiết (Classification Report)**")
    report_dict = classification_report(Y_test, yhat_test, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df, use_container_width=True)

# ------------------------------------------------------------
# TAB 4: SỬ DỤNG MÔ HÌNH
# ------------------------------------------------------------
with tab4:
    if "model" not in st.session_state:
        st.info("👈 Vui lòng bấm nút **'Huấn luyện mô hình'** ở thanh bên trước khi dự báo.")
        st.stop()

    model = st.session_state["model"]
    feature_cols = st.session_state["feature_cols"]
    data_ranges = st.session_state["data_ranges"]

    mode = st.radio(
        "Chọn chế độ sử dụng",
        options=["Nhập trực tiếp", "Tải tệp hàng loạt"],
        horizontal=True,
    )

    if mode == "Nhập trực tiếp":
        st.caption(
            "Nhập điểm khảo sát (thang 1-5) cho từng tiêu chí thuộc 5 nhóm 5C, "
            "sau đó bấm 'Dự báo' để xem xác suất rủi ro tín dụng."
        )
        with st.form("predict_form"):
            input_values = {}
            for prefix, group_label in GROUP_NAMES.items():
                st.markdown(f"**{group_label}**")
                cols_in_group = [c for c in feature_cols if col_group(c) == prefix]
                widget_cols = st.columns(len(cols_in_group))
                for wc, col in zip(widget_cols, cols_in_group):
                    r = data_ranges[col]
                    input_values[col] = wc.number_input(
                        col,
                        min_value=r["min"],
                        max_value=r["max"],
                        value=r["median"],
                        step=1,
                        help=f"Điểm khảo sát biến {col} (thang {r['min']}-{r['max']}).",
                    )
            submitted = st.form_submit_button("🔮 Dự báo", type="primary", use_container_width=True)

        if submitted:
            X_new = pd.DataFrame([[input_values[c] for c in feature_cols]], columns=feature_cols)
            pred = model.predict(X_new)[0]
            proba = model.predict_proba(X_new)[0]

            st.divider()
            if pred == 1:
                st.error(f"⚠️ Kết quả: **CÓ RỦI RO** tín dụng")
            else:
                st.success(f"✅ Kết quả: **KHÔNG CÓ RỦI RO** tín dụng")

            p1, p2 = st.columns(2)
            p1.metric("Xác suất không rủi ro (PD=0)", f"{proba[0] * 100:.2f}%")
            p2.metric("Xác suất có rủi ro (PD=1)", f"{proba[1] * 100:.2f}%")

    else:
        st.caption(
            "Tải lên tệp CSV chứa đúng 24 cột biến đầu vào (TC1-5, NL1-4, DK1-5, V1-6, TS1-4) "
            "để dự báo hàng loạt."
        )
        batch_file = st.file_uploader(
            "Tải tệp dữ liệu cần dự báo (CSV)", type=["csv"], key="batch_predict_uploader"
        )
        if batch_file is not None:
            try:
                new_df = pd.read_csv(io.BytesIO(batch_file.getvalue()))
            except Exception as e:
                st.error(f"Không thể đọc tệp. Lỗi: {e}")
                st.stop()

            missing_batch = [c for c in feature_cols if c not in new_df.columns]
            if missing_batch:
                st.error(
                    "Tệp thiếu các cột bắt buộc sau: " + ", ".join(missing_batch)
                )
            else:
                X_batch = new_df[feature_cols]
                preds = model.predict(X_batch)
                probas = model.predict_proba(X_batch)

                result_df = new_df.copy()
                result_df["Dự báo (PD)"] = preds
                result_df["Xác suất không rủi ro (%)"] = (probas[:, 0] * 100).round(2)
                result_df["Xác suất có rủi ro (%)"] = (probas[:, 1] * 100).round(2)

                st.subheader("Kết quả dự báo")
                with st.container(height=400):
                    st.dataframe(result_df, use_container_width=True)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "⬇️ Tải kết quả dự báo (CSV)",
                    data=csv_bytes,
                    file_name="ket_qua_du_bao_rui_ro_tin_dung.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
