import streamlit as st
import requests

st.set_page_config(page_title="Sepsis Predictive UI", layout="wide")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Patient Vital Signs Ingestion")
    hr_val = st.slider("Heart Rate (HR) [bpm]", 30, 200, 78)
    o2_val = st.slider("Oxygen Saturation (O2Sat) [%]", 70, 100, 98)
    temp_val = st.slider("Body Temperature (Temp) [°C]", 35.0, 41.0, 36.80)
    sbp_val = st.slider("Systolic Blood Pressure (SBP) [mmHg]", 50, 200, 120)
    map_val = st.slider("Mean Arterial Pressure (MAP) [mmHg]", 40, 150, 82)
    dbp_val = st.slider("Diastolic Blood Pressure (DBP) [mmHg]", 30, 150, 70)
    resp_val = st.slider("Respiration Rate (Resp) [breaths/min]", 10, 50, 16)
    
    st.markdown("---")
    st.subheader("📝 Clinical Notes (NLP Integration)")
    clinical_notes = st.text_area("Enter Observations:", placeholder="e.g., Patient is shivering and pale.")

with col2:
    st.header("📊 Real-Time Model Analysis")
    st.write("Adjust the vital signs panel on the left and trigger an AI diagnostic check below.")
    
    if st.button("Execute Diagnostic Assessment", type="primary", use_container_width=True):
        payload = {
            "HR": hr_val, "O2Sat": o2_val, "Temp": temp_val,
            "SBP": sbp_val, "MAP": map_val, "DBP": dbp_val, "Resp": resp_val,
            "notes": clinical_notes
        }
        
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            result = response.json()
            
            # Prediction Alert
            if result.get("prediction") == "Sepsis Confirmed":
                st.error("⚠️ CRITICAL WARNING: HIGHEST SEPSIS RISK ALERT")
            else:
                st.success("✅ PATIENT STABLE: Standard Monitoring Benchmarks Maintained")
            
            # Metrics
            m1, m2 = st.columns(2)
            m1.metric("Calculated Risk Score", f"{result.get('final_risk_score', 10.17)}%")
            status = "Critical" if result.get("prediction") == "Sepsis Confirmed" else "Clear / Normal"
            m2.metric("Inference Evaluation Status", status)
            
            # NLP Alert
            nlp_risk = result.get("nlp_added_risk", 0)
            if nlp_risk > 0:
                st.warning(f"🧠 NLP Note Match: +{nlp_risk}% Risk added due to critical keywords in notes.")
            
            # Smart Actions
            st.info("**Directed Medical Action Protocol:**")
            for act in result.get("actions", ["Patient stable."]):
                st.write("➡️ " + act)
            
            # XAI Explanation
            st.markdown("---")
            st.subheader("🔍 Risk Explanation (XAI)")
            if result.get("prediction") == "Sepsis Confirmed":
                if sbp_val < 90: st.error(f"👉 Critical Blood Pressure ({sbp_val} mmHg).")
                if hr_val > 100: st.error(f"👉 Elevated Heart Rate ({hr_val} bpm).")
                if temp_val > 38.5: st.error(f"👉 Fever detected ({temp_val}°C).")
                if o2_val < 92: st.error(f"👉 Low Oxygen ({o2_val}%).")
            else:
                st.write("Vitals are within stable ranges.")
                
        except Exception as e:
            st.error("⚠️ Error connecting to API. Please ensure your FastAPI backend is running.")