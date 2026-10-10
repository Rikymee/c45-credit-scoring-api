from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="API Credit Scoring C4.5 Kopdit Sangosay")

# Izinkan CORS agar Web di Hostinger bisa mengakses API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model C4.5 dan Daftar Kolom Training
model = joblib.load('model_c45_sangosay.pkl')
feature_columns = joblib.load('model_columns.pkl')

class DebiturSchema(BaseModel):
    nama: str
    pekerjaan: str
    umur: int
    penghasilan: float
    jumlah_pinjaman: float
    jangka_waktu: int
    kode_pinjaman: str = "PRO3"
    kode_tujuan: str = "JSB"
    jenis_jaminan: str

@app.get("/")
def home():
    return {"status": "API Credit Scoring C4.5 Active"}

@app.post("/predict")
def predict_credit(data: DebiturSchema):
    angsuran = data.jumlah_pinjaman / data.jangka_waktu
    rasio = data.jumlah_pinjaman / data.penghasilan

    # Preprocessing Jaminan
    jaminan_upper = data.jenis_jaminan.upper()
    if 'TANAH' in jaminan_upper or 'SERTIFIKAT' in jaminan_upper:
        j_group = 'SERTIFIKAT_TANAH'
    elif 'BPKB' in jaminan_upper:
        j_group = 'BPKB'
    elif 'SAHAM' in jaminan_upper or 'SIMPANAN' in jaminan_upper:
        j_group = 'SIMPANAN_SAHAM'
    else:
        j_group = 'LAINNYA'

    # Buat Dataframe Input
    df_in = pd.DataFrame([{
        'Jlh Pinjaman': data.jumlah_pinjaman,
        'Jangka Waktu': data.jangka_waktu,
        'umur_tahun': data.umur,
        'penghasilan_numeric': data.penghasilan,
        'angsuran_pokok_per_bulan': angsuran,
        'rasio_pinjaman_penghasilan': rasio,
        'Kode Pinjaman': data.kode_pinjaman,
        'Kode Tujuan': data.kode_tujuan,
        'Pekerjaan_clean': data.pekerjaan.title().strip(),
        'Jaminan_Group': j_group
    }])

    # One-Hot Encoding & Alignment
    df_encoded = pd.get_dummies(df_in)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # Prediksi
    pred = model.predict(df_encoded)[0]

    return {
        "nama": data.nama,
        "prediksi": pred,
        "rekomendasi": "TERIMA (Kredit Layak Disetujui)" if pred == "Lancar" else "TOLAK (Risiko Kredit Macet)",
        "angsuran_per_bulan": round(angsuran, 2),
        "rasio_pinjaman": round(rasio, 2)
    }
