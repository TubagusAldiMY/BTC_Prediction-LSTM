import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime, timedelta
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_PATH = 'model/bitcoin_prediction_lstm_FINAL.h5'
SCALER_PATH = 'scaler/scaler.gz'
TIME_STEP = 60


def get_prediction():
    """
    Fungsi utama untuk melakukan seluruh alur prediksi.
    Mengembalikan dictionary berisi prediksi, data historis untuk plot, dan log.
    """
    log_messages = []

    # 1. Muat artifak
    log_messages.append("1. Memuat model dan scaler...")
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        log_messages.append("   -> Model dan scaler berhasil dimuat.")
    except Exception as e:
        log_messages.append(f"   -> GAGAL: {e}.")
        return {'prediction': None, 'historical_data': pd.DataFrame(), 'log': "\n".join(log_messages)}

    # 2. Ambil data historis
    log_messages.append(f"\n2. Mengambil data harga 1 tahun terakhir...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    try:
        # Ambil data historis untuk diplot
        historical_data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)

        if historical_data.empty:
            log_messages.append("   -> GAGAL: Tidak ada data yang diunduh dari yfinance.")
            return {'prediction': None, 'historical_data': pd.DataFrame(), 'log': "\n".join(log_messages)}

        # Ambil 60 hari terakhir dari data tersebut untuk input prediksi
        latest_data_for_prediction = historical_data['Close'][-TIME_STEP:].values.reshape(-1, 1)

        if len(latest_data_for_prediction) < TIME_STEP:
            log_messages.append("   -> GAGAL: Tidak cukup data historis untuk prediksi.")
            return {'prediction': None, 'historical_data': historical_data, 'log': "\n".join(log_messages)}
        log_messages.append(f"   -> Data berhasil diambil.")
    except Exception as e:
        log_messages.append(f"   -> GAGAL mengambil data: {e}")
        return {'prediction': None, 'historical_data': pd.DataFrame(), 'log': "\n".join(log_messages)}

    # 3. Pra-pemrosesan data untuk prediksi
    log_messages.append("\n3. Melakukan pra-pemrosesan data...")
    scaled_data = scaler.transform(latest_data_for_prediction)
    reshaped_data = np.reshape(scaled_data, (1, TIME_STEP, 1))
    log_messages.append("   -> Data siap untuk diprediksi.")

    # 4. Buat prediksi
    log_messages.append("\n4. Membuat prediksi...")
    prediction_scaled = model.predict(reshaped_data)
    prediction_usd = scaler.inverse_transform(prediction_scaled)
    final_prediction = prediction_usd[0][0]
    log_messages.append("   -> Prediksi berhasil dibuat.")

    # Kembalikan semua hasil yang dibutuhkan oleh frontend
    return {
        'prediction': final_prediction,
        'historical_data': historical_data,
        'log': "\n".join(log_messages)
    }