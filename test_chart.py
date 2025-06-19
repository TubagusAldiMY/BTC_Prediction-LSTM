import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("Tes Render Grafik Plotly")

# 1. Buat DataFrame sederhana secara manual
data = {
    'Tanggal': pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04']),
    'Harga': [100, 110, 105, 120]
}
df_test = pd.DataFrame(data)
df_test = df_test.set_index('Tanggal')

st.write("Data yang akan digambar:")
st.dataframe(df_test)

# 2. Buat grafik garis sederhana dari DataFrame di atas
st.write("Tes Grafik:")
try:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test['Harga'], mode='lines+markers'))

    # 3. Tampilkan grafik
    st.plotly_chart(fig)
    st.success("Jika Anda melihat grafik di atas, berarti Streamlit & Plotly bekerja dengan baik!")
except Exception as e:
    st.error(f"Gagal membuat grafik: {e}")