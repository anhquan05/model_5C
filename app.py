import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# 1. SET_PAGE_CONFIG (Lệnh đầu tiên bắt buộc của Streamlit)
st.set_page_config(layout="wide", page_title="Hệ thống Dự báo Rủi ro", page_icon="🎯")

# Tập biến đầu vào (X) và biến mục tiêu (y) trích xuất từ notebook
FEATURE_COLS = ['TC1', 'TC2', 'TC3', 'TC4', 'TC5', 'NL1', 'NL2', 'NL3', 
                'NL4', 'DK1', 'DK2', 'DK3', 'DK4', 'DK5', 'V1', 'V2', 'V3', 
                'V4', 'V5', 'V6', 'TS1', 'TS2', 'TS3', 'TS4']
TARGET_COL = 'PD'

# 2. HÀM LOAD DỮ LIỆU CÓ CACHE
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file)
    else:
        st.error("❌ Định dạng file không được hỗ trợ. Vui lòng tải lên CSV hoặc Excel.")
        return None

# Khởi tạo session_state để lưu mô hình và kết quả, tránh train lại khi chuyển tab
if 'model' not in st.session_state:
    st.session_state.model = None
if 'eval_results' not in st.session_state:
    st.session_state.eval_results = {}

# 3. THÀNH PHẦN 1: SIDEBAR - VÙNG CẤU HÌNH
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    uploaded_file = st.file_uploader("1. Tải lên dữ liệu (.csv, .xlsx)", type=["csv", "xlsx"], help="Dữ liệu cần có đủ 24 biến X và biến mục tiêu PD.")
    
    st.subheader("Tham số mô hình AI")
    st.caption("Thuật toán: Logistic Regression")
    
    random_state = st.number_input("Random State", min_value=0, max_value=1000, value=32, step=1, help="Seed ngẫu nhiên (notebook đang dùng 32).")
    test_size = st.slider("Tỷ lệ chia Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05, help="Tỷ lệ tập kiểm định (notebook đang dùng 0.2).")
    
    st.divider()
    train_button = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

# 4. THÀNH PHẦN 2: HEADER - VÙNG ĐỊNH HƯỚNG
st.title("🎯 Hệ thống Đánh giá & Dự báo Rủi ro")
st.caption("Ứng dụng phân loại rủi ro (PD) dựa trên 24 chỉ số bằng thuật toán Logistic Regression.")

if not uploaded_file:
    st.info("👈 Vui lòng tải lên file dữ liệu ở thanh bên trái (Sidebar) để bắt đầu sử dụng.")
    st.stop()

# Đọc dữ liệu
df = load_data(uploaded_file)
if df is not None:
    st.caption(f"📁 Đang sử dụng tệp: **{uploaded_file.name}**")
else:
    st.stop()

# Kiểm tra tính hợp lệ của schema dữ liệu
missing_cols = [col for col in FEATURE_COLS + [TARGET_COL] if col not in df.columns]
if missing_cols:
    st.error(f"❌ Dữ liệu không hợp lệ. File tải lên đang thiếu các cột sau: {', '.join(missing_cols)}")
    st.stop()

st.divider()

# 5. KHỐI XỬ LÝ TRAIN MÔ HÌNH
if train_button:
    with st.spinner("Đang xử lý dữ liệu và huấn luyện mô hình..."):
        # Tách X, y
        X = df[FEATURE_COLS]
        y = df[TARGET_COL]
        
        # Tiền xử lý nhanh: lấp khoảng trống để tránh lỗi nếu dữ liệu thật có NaNs
        X = X.fillna(X.median())
        y = y.fillna(y.mode()[0])
        
        # Chia tập train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Huấn luyện Logistic Regression
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        # Kiểm định trên tập test
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Lưu vào session
        st.session_state.model = model
        st.session_state.eval_results = {
            'accuracy': acc,
            'confusion_matrix': cm,
            'classes': model.classes_
        }
    st.success("✅ Huấn luyện thành công! Xem chi tiết ở các tab bên dưới.")

# 6. KHỐI TABS (NỘI DUNG CHÍNH)
tab1, tab2, tab3, tab4 = st.tabs(["📊 Tổng quan dữ liệu", "📈 Trực quan hóa", "🧪 Kết quả kiểm định", "🔮 Sử dụng mô hình"])

# --- TAB 1: TỔNG QUAN ---
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng số dòng", f"{df.shape[0]:,}")
    col2.metric("Tổng số cột", df.shape[1])
    col3.metric("Dung lượng file", f"{uploaded_file.size / 1024:.2f} KB")
    
    st.subheader("Dữ liệu thô (5 dòng đầu)")
    with st.container(height=300):
        st.dataframe(df.head(), use_container_width=True)
        
    st.subheader("Thống kê mô tả các biến đặc trưng")
    st.dataframe(df[FEATURE_COLS + [TARGET_COL]].describe(), use_container_width=True)

# --- TAB 2: TRỰC QUAN HÓA ---
with tab2:
    st.subheader("Phân phối dữ liệu")
    selected_cols = st.multiselect(
        "Chọn biến để trực quan hóa (tối đa 4 biến để hiển thị đẹp nhất):", 
        options=[TARGET_COL] + FEATURE_COLS, 
        default=[TARGET_COL, FEATURE_COLS[0], FEATURE_COLS[1], FEATURE_COLS[2]]
    )
    
    if selected_cols:
        cols = st.columns(2)
        for i, col_name in enumerate(selected_cols):
            with cols[i % 2]:
                # Vẽ dạng Bar cho biến rời rạc/phân loại, dạng Hist cho biến liên tục
                if df[col_name].nunique() <= 10 or col_name == TARGET_COL:
                    counts = df[col_name].value_counts().reset_index()
                    counts.columns = [col_name, 'Số lượng']
                    fig = px.bar(counts, x=col_name, y='Số lượng', title=f"Phân phối: {col_name}")
                else:
                    fig = px.histogram(df, x=col_name, title=f"Phân phối: {col_name}")
                st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: KẾT QUẢ KIỂM ĐỊNH ---
with tab3:
    if st.session_state.model is None:
        st.warning("⚠️ Vui lòng cấu hình và bấm nút 'Huấn luyện mô hình' ở thanh bên trái trước.")
    else:
        st.subheader("Đánh giá độ chính xác (Trên tập Test)")
        acc_value = st.session_state.eval_results['accuracy']
        st.metric(label="Accuracy Score", value=f"{acc_value * 100:.2f}%")
        
        st.divider()
        st.subheader("Ma trận nhầm lẫn (Confusion Matrix)")
        cm = st.session_state.eval_results['confusion_matrix']
        classes = st.session_state.eval_results['classes']
        
        fig_cm = ff.create_annotated_heatmap(
            z=cm, 
            x=[f"Dự báo {c}" for c in classes],
            y=[f"Thực tế {c}" for c in classes],
            colorscale='Blues',
            showscale=True
        )
        fig_cm.update_layout(margin=dict(t=50, l=200))
        st.plotly_chart(fig_cm, use_container_width=True)

# --- TAB 4: SỬ DỤNG MÔ HÌNH ---
with tab4:
    if st.session_state.model is None:
        st.warning("⚠️ Bạn cần huấn luyện mô hình trước khi có thể thực hiện dự báo.")
    else:
        mode = st.radio("Lựa chọn phương thức dự báo:", ["✍️ Nhập dữ liệu thủ công", "📂 Tải file dự báo hàng loạt"], horizontal=True)
        
        if mode == "✍️ Nhập dữ liệu thủ công":
            st.markdown("### Nhập tham số đầu vào")
            with st.form("predict_form"):
                input_data = {}
                cols_form = st.columns(4)
                
                # Tạo form nhập liệu cho 24 biến
                for idx, col_name in enumerate(FEATURE_COLS):
                    with cols_form[idx % 4]:
                        default_val = float(df[col_name].median()) if pd.api.types.is_numeric_dtype(df[col_name]) else float(df[col_name].mode()[0])
                        input_data[col_name] = st.number_input(f"{col_name}", value=default_val)
                
                submitted = st.form_submit_button("🔮 Thực hiện dự báo", type="primary")
                
                if submitted:
                    input_df = pd.DataFrame([input_data])
                    model = st.session_state.model
                    
                    pred = model.predict(input_df)[0]
                    probas = model.predict_proba(input_df)[0]
                    
                    st.divider()
                    st.subheader("Kết quả dự báo")
                    st.success(f"**Phân loại rủi ro (PD): {pred}**")
                    st.write(f"- Xác suất **Không có rủi ro** (Lớp 0): {probas[0]*100:.2f}%")
                    st.write(f"- Xác suất **Có rủi ro** (Lớp 1): {probas[1]*100:.2f}%")
                    
        else:
            st.markdown("### Tải tập dữ liệu mới")
            st.info("File tải lên cần đảm bảo có đủ 24 cột biến đầu vào (Từ TC1 đến TS4).")
            pred_file = st.file_uploader("Tải file (.csv, .xlsx)", type=['csv', 'xlsx'], key="pred_file")
            
            if pred_file:
                pred_df = load_data(pred_file)
                missing_pred_cols = [c for c in FEATURE_COLS if c not in pred_df.columns]
                
                if missing_pred_cols:
                    st.error(f"❌ File thiếu các cột cần thiết: {', '.join(missing_pred_cols)}")
                else:
                    X_pred = pred_df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
                    
                    predictions = st.session_state.model.predict(X_pred)
                    pred_df['Du_Bao_PD'] = predictions
                    
                    st.subheader("Kết quả dự báo hàng loạt")
                    with st.container(height=300):
                        st.dataframe(pred_df, use_container_width=True)
                        
                    csv = pred_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                    st.download_button(
                        label="⬇️ Tải xuống kết quả dự báo (.csv)",
                        data=csv,
                        file_name="ket_qua_du_bao_hang_loat.csv",
                        mime="text/csv",
                    )
