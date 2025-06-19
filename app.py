import streamlit as st
import pandas as pd
from predictor import get_prediction
from datetime import datetime, timedelta
import plotly.graph_objects as go


# Kembali ke fungsi grafik Candlestick asli kita yang sudah bagus
def create_price_chart(df):
    """Membuat grafik harga candlestick interaktif menggunakan Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Harga Historis'))

    fig.update_layout(
        title='Grafik Harga Bitcoin (1 Tahun Terakhir)',
        yaxis_title='Harga (USD)',
        xaxis_title='Tanggal',
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    return fig


# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediktor Harga Bitcoin",
    page_icon="₿",
    layout="wide"
)

# --- Tampilan Aplikasi ---
st.title("₿ Prediktor Harga Bitcoin")
st.write(
    "Aplikasi ini menggunakan model Deep Learning (LSTM) untuk memprediksi "
    "harga penutupan Bitcoin untuk hari berikutnya. Klik tombol di bawah untuk mendapatkan prediksi terbaru."
)
st.markdown("---")

# Tombol untuk memulai prediksi
if st.button('🚀 Dapatkan Prediksi & Grafik Terbaru', type="primary"):

    with st.spinner('Model sedang bekerja... Mengambil data terbaru dan membuat prediksi...'):
        result = get_prediction()

    st.markdown("---")

    if result['prediction'] is not None and not result['historical_data'].empty:
        predicted_price = result['prediction']
        historical_df = result['historical_data']
        tomorrow_date = (datetime.now() + timedelta(days=1)).strftime('%d %B %Y')

        # --- LANGKAH PEMBERSIHAN DATA (SANITIZATION) - VERSI PERBAIKAN ---
        # Tambahkan .flatten() untuk memastikan semua data 1-dimensi
        clean_df_for_plotting = pd.DataFrame({
            'Open': historical_df['Open'].values.flatten(),
            'High': historical_df['High'].values.flatten(),
            'Low': historical_df['Low'].values.flatten(),
            'Close': historical_df['Close'].values.flatten(),
        }, index=historical_df.index)
        # -------------------------------------------------------------

        col1, col2 = st.columns([1, 3])

        with col1:
            st.success(f"Prediksi Berhasil!")
            st.metric(
                label=f"Prediksi untuk {tomorrow_date}",
                value=f"${predicted_price:,.2f}"
            )
            st.info("Gunakan mouse Anda pada grafik untuk zoom, pan, dan melihat detail harga harian.")

        with col2:
            # Buat dan tampilkan grafik menggunakan DataFrame yang sudah bersih
            price_chart = create_price_chart(clean_df_for_plotting)
            st.plotly_chart(price_chart, use_container_width=True)

        with st.expander("Lihat Log Proses"):
            st.text(result['log'])
    else:
        st.error("Gagal membuat prediksi. Silakan lihat log di bawah.")
        with st.expander("Lihat Log Error"):
            st.text(result['log'])

st.markdown(
    "<br><br><div style='text-align: center;'>Dibuat dengan ❤️ menggunakan Streamlit & TensorFlow</div>",
    unsafe_allow_html=True
)