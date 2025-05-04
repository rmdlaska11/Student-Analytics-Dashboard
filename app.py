import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Konfigurasi halaman ---
st.set_page_config(page_title="🎓 Student Status Classifier", layout="centered")
st.title("🎓 Prediksi Status Mahasiswa")

# --- CSS untuk tampilan ---
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stDataFrame th, .stDataFrame td {
        text-align: center !important;
        color: black !important;
        background-color: white !important;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5em 1em;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.write("Masukkan data untuk memprediksi apakah mahasiswa akan **Dropout**, **Enrolled**, atau **Graduate**.")

# --- Load model dan encoder ---
label_encoder = joblib.load("label_encoder.joblib")
scaler = joblib.load("minmax_scaler.joblib")
model = joblib.load("xgb_model.joblib")

# List of feature names (these should match the features used in the model)
feature_names = [
    "Marital_status", "Application_mode", "Application_order", "Course", "Daytime_evening_attendance",
    "Previous_qualification", "Previous_qualification_grade", "Nacionality", "Mothers_qualification",
    "Fathers_qualification", "Mothers_occupation", "Fathers_occupation", "Admission_grade", "Displaced",
    "Educational_special_needs", "Debtor", "Tuition_fees_up_to_date", "Gender", "Scholarship_holder",
    "Age_at_enrollment", "International", "Curricular_units_1st_sem_credited", "Curricular_units_1st_sem_enrolled",
    "Curricular_units_1st_sem_evaluations", "Curricular_units_1st_sem_approved", "Curricular_units_1st_sem_grade",
    "Curricular_units_1st_sem_without_evaluations", "Curricular_units_2nd_sem_credited",
    "Curricular_units_2nd_sem_enrolled", "Curricular_units_2nd_sem_evaluations",
    "Curricular_units_2nd_sem_approved", "Curricular_units_2nd_sem_grade",
    "Curricular_units_2nd_sem_without_evaluations", "Unemployment_rate", "Inflation_rate", "GDP"
]

# Dropdown options for categorical variables (include Mother's and Father's occupation)
dropdown_options = {
    "Gender": {0: "Female", 1: "Male"},
    "Marital_status": {1: "Single", 2: "Married", 3: "Widower", 4: "Divorced", 5: "Facto Union", 6: "Legally Separated"},
    "Daytime_evening_attendance": {0: "Evening", 1: "Daytime"},
    "Displaced": {0: "No", 1: "Yes"},
    "Educational_special_needs": {0: "No", 1: "Yes"},
    "Debtor": {0: "No", 1: "Yes"},
    "Tuition_fees_up_to_date": {0: "No", 1: "Yes"},
    "Scholarship_holder": {0: "No", 1: "Yes"},
    "International": {0: "No", 1: "Yes"},
    "Application_mode": {
        1: "1st phase - general contingent", 2: "Ordinance No. 612/93", 5: "1st phase - special contingent (Azores Island)",
        7: "Holders of other higher courses", 10: "Ordinance No. 854-B/99", 15: "1st phase - special contingent (Madeira Island)",
        16: "International student (bachelor)", 17: "2nd phase - general contingent", 18: "3rd phase - general contingent",
        26: "Ordinance No. 533-A/99, item b2) (Different Plan)", 27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
        39: "Over 23 years old", 42: "Transfer", 43: "Change of course", 44: "Technological specialization diploma holders",
        51: "Change of institution/course", 53: "Short cycle diploma holders", 57: "Change of institution/course (International)"
    },
    "Course": {
        33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design", 8014: "Social Service (evening attendance)",
        9003: "Agronomy", 9070: "Communication Design", 9085: "Veterinary Nursing", 9119: "Informatics Engineering",
        9130: "Equinculture", 9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
        9556: "Oral Hygiene", 9670: "Advertising and Marketing Management", 9773: "Journalism and Communication",
        9853: "Basic Education", 9991: "Management (evening attendance)"
    },
    "Previous_qualification": {1: "Secondary education", 2: "Higher education - bachelor's degree", 3: "Higher education - degree", 
                               4: "Higher education - master's", 5: "Higher education - doctorate", 6: "Frequency of higher education", 
                               9: "12th year of schooling - not completed", 10: "11th year of schooling - not completed"},
    "Nacionality": {1: "Portuguese", 2: "German", 6: "Spanish", 11: "Italian", 13: "Dutch", 14: "English", 17: "Lithuanian", 
                    21: "Angolan", 22: "Cape Verdean", 24: "Guinean", 25: "Mozambican", 26: "Santomean", 32: "Turkish", 
                    41: "Brazilian", 62: "Romanian", 100: "Moldova (Republic of)", 101: "Mexican", 103: "Ukrainian", 
                    105: "Russian", 108: "Cuban", 109: "Colombian"},
    "Mothers_qualification": {1: "Secondary Education", 2: "Higher Education - Bachelor's Degree", 3: "Higher Education - Degree", 
                             4: "Higher Education - Master's", 5: "Higher Education - Doctorate", 6: "Frequency of Higher Education"},
    "Fathers_qualification": {1: "Secondary Education", 2: "Higher Education - Bachelor's Degree", 3: "Higher Education - Degree", 
                             4: "Higher Education - Master's", 5: "Higher Education - Doctorate", 6: "Frequency of Higher Education"},
    "Mothers_occupation": {
        0: "Student", 1: "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers", 
        2: "Specialists in Intellectual and Scientific Activities", 3: "Intermediate Level Technicians and Professions", 
        4: "Administrative staff", 5: "Personal Services, Security and Safety Workers and Sellers", 6: "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry",
        7: "Skilled Workers in Industry, Construction and Craftsmen", 8: "Installation and Machine Operators and Assembly Workers", 
        9: "Unskilled Workers", 10: "Armed Forces Professions", 90: "Other Situation", 122: "Health professionals", 
        123: "Teachers", 125: "Specialists in information and communication technologies (ICT)", 
        131: "Intermediate level science and engineering technicians and professions", 
        132: "Technicians and professionals, of intermediate level of health", 
        134: "Intermediate level technicians from legal, social, sports, cultural and similar services", 
        141: "Office workers, secretaries in general and data processing operators", 
        143: "Data, accounting, statistical, financial services and registry-related operators", 
        144: "Other administrative support staff", 151: "Personal service workers", 152: "Sellers", 
        153: "Personal care workers and the like", 171: "Skilled construction workers and the like, except electricians", 
        173: "Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like", 
        175: "Workers in food processing, woodworking, clothing and other industries and crafts", 
        191: "Cleaning workers", 192: "Unskilled workers in agriculture, animal production, fisheries and forestry", 
        193: "Unskilled workers in extractive industry, construction, manufacturing and transport", 
        194: "Meal preparation assistants"
    },
    "Fathers_occupation": {
        0: "Student", 1: "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers", 
        2: "Specialists in Intellectual and Scientific Activities", 3: "Intermediate Level Technicians and Professions", 
        4: "Administrative staff", 5: "Personal Services, Security and Safety Workers and Sellers", 6: "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry",
        7: "Skilled Workers in Industry, Construction and Craftsmen", 8: "Installation and Machine Operators and Assembly Workers", 
        9: "Unskilled Workers", 10: "Armed Forces Professions", 90: "Other Situation", 101: "Armed Forces Officers", 
        102: "Armed Forces Sergeants", 103: "Other Armed Forces personnel", 112: "Directors of administrative and commercial services", 
        114: "Hotel, catering, trade and other services directors", 121: "Specialists in the physical sciences, mathematics, engineering and related techniques", 
        122: "Health professionals", 123: "Teachers", 124: "Specialists in finance, accounting, administrative organization, public and commercial relations", 
        131: "Intermediate level science and engineering technicians and professions", 
        132: "Technicians and professionals, of intermediate level of health", 
        134: "Intermediate level technicians from legal, social, sports, cultural and similar services", 
        135: "Information and communication technology technicians", 141: "Office workers, secretaries in general and data processing operators", 
        143: "Data, accounting, statistical, financial services and registry-related operators", 
        144: "Other administrative support staff", 151: "Personal service workers", 152: "Sellers", 
        153: "Personal care workers and the like", 154: "Protection and security services personnel", 
        161: "Market-oriented farmers and skilled agricultural and animal production workers", 
        163: "Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence", 171: "Skilled construction workers and the like, except electricians", 
        172: "Skilled workers in metallurgy, metalworking and similar", 174: "Skilled workers in electricity and electronics", 
        175: "Workers in food processing, woodworking, clothing and other industries and crafts", 181: "Fixed plant and machine operators", 
        182: "Assembly workers", 183: "Vehicle drivers and mobile equipment operators", 192: "Unskilled workers in agriculture, animal production, fisheries and forestry", 
        193: "Unskilled workers in extractive industry, construction, manufacturing and transport", 194: "Meal preparation assistants", 
        195: "Street vendors (except food) and street service providers"
    }
}

# --- Tab Input Manual & CSV ---
tab1, tab2 = st.tabs(["📝 Input Manual", "📂 Upload CSV"])

input_df = None

with tab1:
    st.subheader("Masukkan Data Mahasiswa Secara Manual")
    user_input = {}
    for feature in feature_names:
        if feature in dropdown_options:
            label_to_val = {v: k for k, v in dropdown_options[feature].items()}
            selection = st.selectbox(f"{feature}", list(dropdown_options[feature].values()), key=feature)
            user_input[feature] = label_to_val[selection]
        else:
            user_input[feature] = st.number_input(f"{feature}", value=0.0, key=feature)
    input_df = pd.DataFrame([user_input])
    with st.expander("📄 Tabel Data Input Manual"):
        st.dataframe(input_df)

    if st.button("🔍 Prediksi Status Mahasiswa"):
        input_scaled = scaler.transform(input_df)
        predictions = model.predict(input_scaled)
        labels = label_encoder.inverse_transform(predictions)
        result_df = input_df.copy()
        result_df["Prediksi"] = labels

        for label in labels:
            if label == "Dropout":
                st.markdown("<div style='background-color: red; padding: 1em; color: white; border-radius: 10px;'><strong>Hasil Prediksi: Dropout</strong></div>", unsafe_allow_html=True)
            elif label == "Graduate":
                st.markdown("<div style='background-color: green; padding: 1em; color: white; border-radius: 10px;'><strong>Hasil Prediksi: Graduate</strong></div>", unsafe_allow_html=True)
            elif label == "Enrolled":
                st.markdown("<div style='background-color: yellow; padding: 1em; color: black; border-radius: 10px;'><strong>Hasil Prediksi: Enrolled</strong></div>", unsafe_allow_html=True)

        with st.expander("📊 Tabel Data Hasil dan Prediksi"):
            st.dataframe(result_df)

        # Simpan ke riwayat
        history_file = "history.csv"
        if os.path.exists(history_file):
            existing = pd.read_csv(history_file)
            updated = pd.concat([existing, result_df], ignore_index=True)
        else:
            updated = result_df
        updated.to_csv(history_file, index=False)

    # --- Riwayat Prediksi hanya di Tab 1 ---
    with st.expander("🕘 Riwayat Prediksi Sebelumnya"):
        if os.path.exists("history.csv"):
            history_df = pd.read_csv("history.csv")
            st.dataframe(history_df)
            st.info(f"Total prediksi yang telah dilakukan: **{len(history_df)}**")
            csv = history_df.to_csv(index=False).encode('utf-8')
            st.download_button("📅 Unduh sebagai CSV", data=csv, file_name="riwayat_prediksi.csv", mime="text/csv")
        else:
            st.warning("Belum ada riwayat prediksi.")

with tab2:
    st.subheader("Upload File CSV")
    uploaded_file = st.file_uploader("Unggah file CSV dengan format yang sesuai", type=["csv"])
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        with st.expander("📄 Preview Data dari File Upload"):
            st.dataframe(input_df)

        if st.button("🔍 Prediksi dari File Upload"):
            input_scaled = scaler.transform(input_df)
            predictions = model.predict(input_scaled)
            labels = label_encoder.inverse_transform(predictions)
            result_df = input_df.copy()
            result_df["Prediksi"] = labels

            st.success("✅ Prediksi selesai!")

            with st.expander("📊 Tabel Hasil Prediksi"):
                st.dataframe(result_df)

            # Tombol unduh hasil
            csv_result = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📅 Unduh Hasil Prediksi", data=csv_result, file_name="hasil_prediksi.csv", mime="text/csv")
    

with st.sidebar:
    # Judul dan deskripsi singkat
    st.markdown(
    "<h2 style='color:#4B8BBE;'>🎓 Student Status Classifier</h2>",
    unsafe_allow_html=True
    )
    st.markdown("Prediksi status mahasiswa berdasarkan data akademik dan demografis. Aplikasi ini memanfaatkan model machine learning untuk klasifikasi status akhir mahasiswa.")

    # Garis pemisah
    st.markdown("---")

    # Navigasi penggunaan
    st.markdown("### 🔍 Petunjuk Penggunaan")
    st.markdown("- **Tab 1:** Input manual data mahasiswa satu per satu.")
    st.markdown("- **Tab 2:** Upload file CSV berisi banyak data mahasiswa.")
    st.markdown("- Lihat hasil prediksi dan unduh sebagai file CSV.")

    # Garis pemisah
    st.markdown("---")

    # Link tambahan atau resources
    st.markdown("### 📂 Contoh Data")

    # Link ke file contoh dari GitHub
    import requests
    sample_url = "https://raw.githubusercontent.com/rmdlaska11/Student-Analytics-Dashboard/refs/heads/main/sampled_data.csv"
    response = requests.get(sample_url)
    if response.status_code == 200:
        st.download_button(
            label="Unduh Sample CSV",
            data=response.content,
            file_name="sample_student_data.csv",
            mime="text/csv",
        )
    else:
        st.error("Gagal mengunduh file contoh dari GitHub.")

    # Garis pemisah
    st.markdown("---")

    # Info pengembang
    st.markdown("### 👨‍💻 Tentang")
    st.info("Dikembangkan oleh Rahmad Ramadhan Laska\n\nMenggunakan Python, Streamlit, dan Model XGBoost.")

    # Footer kecil
    st.markdown("<p style='font-size: small;'>© 2025 Student Classifier App</p>", unsafe_allow_html=True)
