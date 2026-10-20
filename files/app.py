import streamlit as st
from inferencing import ModelInferencer

# [opsional] atur konfigurasi halaman web
st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="",
    layout="centered"
)

# gunakan st.cache_resource agar model (file .pkl) tidak perlu
# di-load ulang dari nol setiap kali user memencet tombol prediksi
@st.cache_resource
def load_model():
    return ModelInferencer()

inferencer = load_model()

# judul aplikasi
st.title("Prediksi Skor Kredit Nasabah")
st.markdown("Aplikasi berbasis *Machine Learning* untuk memprediksi klasifikasi skor kredit nasabah berdasarkan riwayat keuangan.")
st.markdown("---")

# membuat form agar halaman tidak re-load setiap kali user mengetik angka
with st.form("form_prediksi"):
    st.subheader("Informasi Pribadi & Pekerjaan")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Umur (Age)", min_value=18, max_value=100, value=30)
        occupation = st.selectbox("Pekerjaan (Occupation)", [
            "Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer", 
            "Lawyer", "Media_Manager", "Doctor", "Journalist", "Manager", 
            "Accountant", "Musician", "Mechanic", "Writer", "Architect"
        ])
    
    with col2:
        annual_income = st.number_input("Pendapatan Tahunan (Annual Income)", min_value=0.0, value=50000.0)
    
    st.markdown("---")
    st.subheader("Riwayat Kredit & Pinjaman")
    col3, col4 = st.columns(2)
    
    with col3:
        num_of_loan = st.number_input("Jumlah Pinjaman (Num of Loan)", min_value=0, value=2)
        outstanding_debt = st.number_input("Hutang Belum Dibayar (Outstanding Debt)", min_value=0.0, value=1000.0)
        changed_credit_limit = st.number_input("Perubahan Limit Kredit (%)", value=5.0)
        
    with col4:
        # langsung meminta input dalam bentuk total bulan agar sesuai dengan format preprocessor
        credit_history_age = st.number_input("Lama Riwayat Kredit (Total Bulan)", min_value=0, value=120)
        amount_invested_monthly = st.number_input("Investasi Bulanan (Amount Invested)", min_value=0.0, value=150.0)
        num_of_delayed_payment = st.number_input("Jumlah Telat Bayar (Delayed Payment)", min_value=0, value=1)

    st.markdown("---")
    st.subheader("Perilaku Pembayaran")
    credit_mix = st.selectbox("Kombinasi Kredit (Credit Mix)", ["Good", "Standard", "Bad"])
    payment_behaviour = st.selectbox("Perilaku Pembayaran (Payment Behaviour)", [
        "High_spent_Small_value_payments",
        "High_spent_Medium_value_payments",
        "High_spent_Large_value_payments",
        "Low_spent_Small_value_payments",
        "Low_spent_Medium_value_payments",
        "Low_spent_Large_value_payments"
    ])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # tombol submit
    submit_button = st.form_submit_button(label="🔍 Analisis Credit Score")

# logika ketika tombol dipencet
if submit_button:
    # 1. kumpulkan semua input ke dalam dictionary yang format/penamaannya 
    # sama persis dengan nama kolom di dataframe pandas
    input_data = {
        "Age": age,
        "Occupation": occupation,
        "Annual_Income": annual_income,
        "Num_of_Loan": num_of_loan,
        "Outstanding_Debt": outstanding_debt,
        "Changed_Credit_Limit": changed_credit_limit,
        "Credit_History_Age": credit_history_age,
        "Amount_invested_monthly": amount_invested_monthly,
        "Num_of_Delayed_Payment": num_of_delayed_payment,
        "Credit_Mix": credit_mix,
        "Payment_Behaviour": payment_behaviour
    }
    
    # tampilkan spinner saat model sedang memproses
    with st.spinner("Memproses data..."):
        try:
            # lempar ke inferencing.py
            hasil_prediksi = inferencer.predict_from_json(input_data)
            
            # tampilkan hasil dengan warna yang sesuai
            st.markdown("---")
            st.subheader("Hasil Prediksi")
            
            if hasil_prediksi == 'Good':
                st.success(f"Skor Kredit Nasabah: **{hasil_prediksi}** 🌟")
            elif hasil_prediksi == 'Standard':
                st.warning(f"Skor Kredit Nasabah: **{hasil_prediksi}** ⚖️")
            else:
                st.error(f"Skor Kredit Nasabah: **{hasil_prediksi}** ⚠️")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")