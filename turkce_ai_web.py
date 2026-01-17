import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import math

# 1. SAYFA AYARLARI
st.set_page_config(page_title="Mühendislik Asistanı (BETA)", page_icon="🧪", layout="wide")

# --- YAN PANEL: 20 MADDE MÜHENDİSLİK ARAÇLARI ---
st.sidebar.header("🔌 Mühendislik Çantası")

# 1. Ohm Kanunu (V=IxR)
with st.sidebar.expander("1. Ohm Kanunu (V, I, R)", expanded=False):
    v_1 = st.number_input("Gerilim (V)", value=12.0, key="v1")
    r_1 = st.number_input("Direnç (Ω)", value=100.0, key="r1")
    st.write(f"Sonuç (I): `{v_1/r_1:.3f} A`")

# 2. Güç Hesabı (P=VxI)
with st.sidebar.expander("2. Güç Hesabı (P, V, I)", expanded=False):
    v_2 = st.number_input("Gerilim (V)", value=220.0, key="v2")
    i_2 = st.number_input("Akım (A)", value=1.0, key="i2")
    st.write(f"Sonuç (P): `{v_2*i_2:.2f} W`")

# 3. Seri Direnç
with st.sidebar.expander("3. Seri Direnç (Reş)", expanded=False):
    r_s = st.text_input("Dirençler (Ω) (örn: 10, 20):", "10, 10", key="rs")
    vals_s = [float(x.strip()) for x in r_s.split(",") if x.strip()]
    st.write(f"Reş: `{sum(vals_s):.2f} Ω`")

# 4. Paralel Direnç
with st.sidebar.expander("4. Paralel Direnç (Reş)", expanded=False):
    r_p = st.text_input("Dirençler (Ω) (örn: 100, 100):", "100, 100", key="rp")
    vals_p = [float(x.strip()) for x in r_p.split(",") if x.strip()]
    st.write(f"Reş: `{1/sum(1/x for x in vals_p):.2f} Ω`" if all(x>0 for x in vals_p) else "Hata")

# 5. Seri Kondansatör
with st.sidebar.expander("5. Seri Kondansatör (Ceş)", expanded=False):
    c_s = st.text_input("Kapasiteler (µF):", "10, 10", key="cs")
    vals_cs = [float(x.strip()) for x in c_s.split(",") if x.strip()]
    st.write(f"Ceş: `{1/sum(1/x for x in vals_cs):.2f} µF`" if all(x>0 for x in vals_cs) else "Hata")

# 6. Paralel Kondansatör
with st.sidebar.expander("6. Paralel Kondansatör (Ceş)", expanded=False):
    c_p = st.text_input("Kapasiteler (µF) (örn: 22, 47):", "22, 47", key="cp")
    vals_cp = [float(x.strip()) for x in c_p.split(",") if x.strip()]
    st.write(f"Ceş: `{sum(vals_cp):.2f} µF`")

# 7. Gerilim Bölücü (Vout)
with st.sidebar.expander("7. Gerilim Bölücü (Vout)", expanded=False):
    vin_7 = st.number_input("Vin (V)", value=5.0, key="vin7")
    r1_7 = st.number_input("R1 (Ω)", value=1000.0, key="r17")
    r2_7 = st.number_input("R2 (Ω)", value=1000.0, key="r27")
    st.write(f"Vout: `{vin_7*(r2_7/(r1_7+r2_7)):.2f} V`")

# 8. Akım Bölücü
with st.sidebar.expander("8. Akım Bölücü (Ix)", expanded=False):
    it_8 = st.number_input("Toplam Akım (A)", value=1.0, key="it8")
    ra_8 = st.number_input("Kol 1 (Ω)", value=100.0, key="ra8")
    rb_8 = st.number_input("Kol 2 (Ω)", value=100.0, key="rb8")
    st.write(f"Kol 2 Akımı: `{it_8*(ra_8/(ra_8+rb_8)):.2f} A`")

# 9. Tersleyen Op-Amp Kazancı
with st.sidebar.expander("9. Tersleyen Op-Amp Kazancı", expanded=False):
    rf_9 = st.number_input("Rf (Ω)", value=10000.0, key="rf9")
    ri_9 = st.number_input("Ri (Ω)", value=1000.0, key="ri9")
    st.write(f"Kazanç (Av): `{-rf_9/ri_9:.2f}`")

# 10. Terslemeyen Op-Amp Kazancı
with st.sidebar.expander("10. Terslemeyen Op-Amp Kazancı", expanded=False):
    rf_10 = st.number_input("Rf (Ω)", value=10000.0, key="rf10")
    ri_10 = st.number_input("Ri (Ω)", value=1000.0, key="ri10")
    st.write(f"Kazanç (Av): `{1+(rf_10/ri_10):.2f}`")

# 11. LED Ön Direnç Hesabı
with st.sidebar.expander("11. LED Ön Direnç Hesabı", expanded=False):
    vs_11 = st.number_input("Kaynak (V)", value=5.0, key="vs11")
    vf_11 = st.number_input("LED Vf (V)", value=2.0, key="vf11")
    if_11 = st.number_input("LED Akımı (mA)", value=20.0, key="if11") / 1000
    st.write(f"Direnç: `{(vs_11-vf_11)/if_11:.1f} Ω`")

# 12. Endüktif Reaktans (XL)
with st.sidebar.expander("12. Endüktif Reaktans (XL)", expanded=False):
    f_12 = st.number_input("Frekans (Hz)", value=50.0, key="f12")
    l_12 = st.number_input("L (mH)", value=10.0, key="l12") / 1000
    st.write(f"XL: `{2*math.pi*f_12*l_12:.2f} Ω`")

# 13. Kapasitif Reaktans (XC)
with st.sidebar.expander("13. Kapasitif Reaktans (XC)", expanded=False):
    f_13 = st.number_input("Frekans (Hz)", value=50.0, key="f13")
    c_13 = st.number_input("C (µF)", value=100.0, key="c13") / 1000000
    st.write(f"XC: `{1/(2*math.pi*f_13*c_13):.2f} Ω`" if c_13>0 else "Hata")

# 14. Rezonans Frekansı (f0)
with st.sidebar.expander("14. Rezonans Frekansı (f0)", expanded=False):
    l_14 = st.number_input("L (mH)", value=10.0, key="l14") / 1000
    c_14 = st.number_input("C (µF)", value=100.0, key="c14") / 1000000
    st.write(f"f0: `{1/(2*math.pi*math.sqrt(l_14*c_14)):.2f} Hz`" if l_14*c_14>0 else "Hata")

# 15. RC Zaman Sabiti (Tau)
with st.sidebar.expander("15. RC Zaman Sabiti (τ)", expanded=False):
    r_15 = st.number_input("R (Ω)", value=1000.0, key="r15")
    c_15 = st.number_input("C (µF)", value=100.0, key="c15") / 1000000
    st.write(f"Tau: `{r_15*c_15*1000:.2f} ms`")

# 16. RL Zaman Sabiti (Tau)
with st.sidebar.expander("16. RL Zaman Sabiti (τ)", expanded=False):
    r_16 = st.number_input("R (Ω)", value=100.0, key="r16")
    l_16 = st.number_input("L (mH)", value=10.0, key="l16") / 1000
    st.write(f"Tau: `{(l_16/r_16)*1000:.2f} ms`" if r_16>0 else "Hata")

# 17. RMS -> Peak Dönüşümü
with st.sidebar.expander("17. RMS -> Peak Dönüşümü", expanded=False):
    vrms_17 = st.number_input("VRMS (V)", value=220.0, key="vrms17")
    st.write(f"Vpeak: `{vrms_17*1.414:.2f} V`")

# 18. Peak -> RMS Dönüşümü
with st.sidebar.expander("18. Peak -> RMS Dönüşümü", expanded=False):
    vp_18 = st.number_input("Vpeak (V)", value=311.0, key="vp18")
    st.write(f"VRMS: `{vp_18*0.707:.2f} V`")

# 19. Desibel (dB) -> Kazanç
with st.sidebar.expander("19. Desibel (dB) -> Kazanç", expanded=False):
    db_19 = st.number_input("dB Değeri", value=20.0, key="db19")
    st.write(f"Voltaj Oranı: `{10**(db_19/20):.2f}`")

# 20. Sayı Sistemi (Dec -> Bin/Hex)
with st.sidebar.expander("20. Sayı Sistemi (Bin/Hex)", expanded=False):
    dec_20 = st.number_input("Decimal", value=255, step=1, key="dec20")
    st.write(f"Bin: `{bin(dec_20)}` | Hex: `{hex(dec_20).upper()}`")

st.sidebar.markdown("---")
st.sidebar.caption("Elektrik Elektronik Mühendisliği Bölümü")
st.sidebar.caption("Yapımcı: Ömer Faruk Bulut")

# --- ANA PANEL: YAPAY ZEKA ---
st.title("🤖 Mühendislik Asistanı (BETA)")
st.caption("Aksaray Üniversitesi | Yapımcı: Ömer Faruk Bulut")

@st.cache_resource
def model_yukle():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype="auto")
    return tokenizer, model

with st.spinner("Sistem Hazırlanıyor..."):
    tokenizer, model = model_yukle()

def latexi_temizle(metin):
    metin = metin.replace("\\[", "$$").replace("\\]", "$$")
    metin = metin.replace("\\(", "$").replace("\\)", "$")
    metin = re.sub(r'(?<!\\)\[\s*(.*?)\s*\]', r'$$\1$$', metin)
    return metin

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(latexi_temizle(message["content"]))

if prompt := st.chat_input("Bir şeyler yazın veya teknik bir soru sorun..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        messages = [{"role": "system", "content": "Sen bir EE mühendisisin. Formülleri LaTeX ($$...$$) ile yaz."},
                    {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, temperature=0.6, do_sample=True)
        raw_response = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]
        cevap = latexi_temizle(raw_response)
        st.markdown(cevap)
        st.session_state.messages.append({"role": "assistant", "content": cevap})