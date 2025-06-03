# backend/app/api.py

import pandas as pd
import io
import numpy as np
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import os
import sys
from .algorithm.chromosome_setup import FEATURE_ORDER, FEATURE_DETAILS, user_input_to_chromosome
from .algorithm.ga_core         import GeneticAlgorithmFeatureSelection

# --- Menambahkan Path untuk Impor Modul Lokal ---
# Asumsi api.py ada di backend/app/
# dan modul GA ada di backend/app/algorithm/
# dan chromosome_setup.py ada di backend/app/algorithm/
# serta Evolution_DataSets.csv ada di backend/data/ (sesuai gambar terakhir) atau ../../data/ dari api.py
# atau di TUBES_KDS/data/

# Menentukan base_dir proyek (TUBES_KDS)
# Ini mungkin perlu disesuaikan tergantung bagaimana Anda menjalankan aplikasi FastAPI
# Jika dijalankan dari direktori TUBES_KDS:
# current_dir = os.path.dirname(os.path.abspath(__file__)) # backend/app
# app_dir = os.path.dirname(current_dir) # backend
# base_dir = os.path.dirname(app_dir) # TUBES_KDS
# sys.path.append(os.path.join(base_dir, "backend", "app"))


class GAParameters(BaseModel):
    population_size: int = Field(50, gt=0)
    num_generations: int = Field(20, gt=0)
    crossover_prob: float = Field(0.8, ge=0.0, le=1.0)
    mutation_prob: float = Field(0.05, ge=0.0, le=1.0)
    # Tambahkan parameter fitness jika perlu diatur dari frontend
    # misal: svm_kernel: str = 'rbf', svm_c: float = 1.0, penalty_factor: float = 0.01

class SimulationRequest(BaseModel):
    user_feature_inputs: Dict[str, Any] # Dict fitur dari pengguna
    ga_params: GAParameters
    target_genus_specie: str
    # Tambahkan 'all_original_feature_names' jika ingin frontend mengirimkannya
    # Namun, ini lebih baik dikelola di backend berdasarkan FEATURE_ORDER

class FeatureEvolutionStep(BaseModel):
    generation: int
    fitness: float
    features: Dict[str, Any] # Kromosom terbaik generasi ini dalam format {nama_fitur: nilai}

class SimulationResponse(BaseModel):
    message: str
    target_genus_specie: str
    final_best_fitness: float
    final_best_features: Dict[str, Any]
    evolution_path: List[FeatureEvolutionStep] # Jejak evolusi
    input_features_processed: Dict[str, Any]

# --- Inisialisasi Aplikasi FastAPI ---
app = FastAPI(title="Evolution Simulation API")

# CORS Middleware (sesuaikan origins jika perlu)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"], # Alamat frontend Vue Anda
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Memuat Dataset (idealnya dimuat sekali saat startup) ---
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "Evolution_DataSets.csv") # Path dari backend/app/api.py ke TUBES_KDS/data/
evolution_df = None
data_load_error = None

# Identifikasi kolom numerik dan kategorikal asli berdasarkan FEATURE_DETAILS
# Ini akan digunakan oleh fungsi fitness
# Pastikan FEATURE_DETAILS sudah diimpor dengan benar
ORIGINAL_NUMERICAL_COLS = [f for f, details in FEATURE_DETAILS.items() if details['type'] == 'numerical' and f in FEATURE_ORDER]
ORIGINAL_CATEGORICAL_COLS = [f for f, details in FEATURE_DETAILS.items() if details['type'] == 'categorical' and f in FEATURE_ORDER]
LABEL_COL_IN_DATASET = 'Genus_&_Specie' # Sesuai dokumen

@app.on_event("startup")
async def load_dataset():
    global evolution_df, data_load_error
    try:
        # Pastikan path ini benar relatif terhadap lokasi di mana uvicorn dijalankan,
        # atau gunakan path absolut.
        # Struktur dari gambar Anda: TUBES_KDS/data/Evolution_DataSets.csv
        # Jika api.py ada di TUBES_KDS/backend/app/, maka path relatifnya: ../../data/Evolution_DataSets.csv
        
        # Cek apakah FEATURE_ORDER sudah terisi (artinya impor chromosome_setup berhasil)
        if not FEATURE_ORDER:
             data_load_error = "FEATURE_ORDER tidak terdefinisi, modul chromosome_setup mungkin gagal diimpor."
             print(data_load_error)
             return

        print(f"Mencoba memuat dataset dari: {DATASET_PATH}")
        if not os.path.exists(DATASET_PATH):
            data_load_error = f"Dataset tidak ditemukan di path: {DATASET_PATH}. Pastikan file ada dan path sudah benar. Current working directory: {os.getcwd()}"
            print(data_load_error)
            return

        evolution_df = pd.read_csv(DATASET_PATH)
        # Basic cleaning (sesuai dokumen, data seharusnya sudah bersih)
        evolution_df.dropna(subset=[LABEL_COL_IN_DATASET], inplace=True) # Hapus baris jika labelnya NaN
        # Anda mungkin perlu cleaning lebih lanjut atau imputasi jika data tidak sebersih yang diharapkan
        # evolution_df.fillna(method='ffill', inplace=True) # Contoh imputasi sederhana
        print("Dataset Evolution_DataSets.csv berhasil dimuat.")
    except Exception as e:
        data_load_error = f"Gagal memuat dataset Evolution_DataSets.csv: {str(e)}"
        print(data_load_error)
        evolution_df = None

# --- Endpoint API ---
@app.post("/simulate_evolution", response_model=SimulationResponse)
async def simulate_evolution_endpoint(request_data: SimulationRequest):
    global evolution_df, data_load_error

    if data_load_error or evolution_df is None:
        raise HTTPException(status_code=500, detail=f"Kesalahan internal server: Dataset tidak bisa dimuat. Detail: {data_load_error}")

    if not FEATURE_ORDER or not GeneticAlgorithmFeatureSelection: # Cek lagi jika modul GA tidak terimpor
        raise HTTPException(status_code=500, detail="Kesalahan internal server: Komponen Algoritma Genetik tidak terinisialisasi.")

    try:
        # 1. Proses input pengguna menjadi format kromosom (jika ada fitur yang hilang, akan diisi acak)
        # user_input_to_chromosome mengembalikan list, kita simpan juga dict aslinya untuk fitness
        processed_user_input_list = user_input_to_chromosome(request_data.user_feature_inputs)
        # Buat dict dari list yang sudah diproses untuk digunakan di fitness jika perlu
        # atau gunakan request_data.user_feature_inputs langsung jika itu yang diinginkan fitness.
        # Sesuai fitness.py terakhir, user_input_dict adalah parameter awal dari pengguna
        # dan harus berupa dict {nama_fitur: nilai}
        
        # Pastikan semua fitur dalam FEATURE_ORDER ada di user_feature_inputs yang diproses
        # untuk digunakan sebagai 'user_input_dict' dalam fitness
        user_params_for_fitness = {
            feature: processed_user_input_list[i] for i, feature in enumerate(FEATURE_ORDER)
        }

        # 2. Inisialisasi GA
        ga_simulator = GeneticAlgorithmFeatureSelection(
            original_df=evolution_df.copy(), # Berikan salinan bersih
            label_col=LABEL_COL_IN_DATASET,
            all_original_feature_names=list(FEATURE_ORDER), # list() untuk memastikan
            numerical_cols_original=ORIGINAL_NUMERICAL_COLS,
            categorical_cols_original=ORIGINAL_CATEGORICAL_COLS,
            population_size=request_data.ga_params.population_size,
            num_generations=request_data.ga_params.num_generations,
            crossover_prob=request_data.ga_params.crossover_prob,
            mutation_prob=request_data.ga_params.mutation_prob,
            target_genus_specie_for_ga=request_data.target_genus_specie,
            initial_user_params_for_ga=user_params_for_fitness,
            fitness_params= {}
        )

        # 3. Jalankan GA
        # Modifikasi ga_core.run() agar menerima target_genus_specie dan user_input_dict jika belum
        # Atau, jika GA secara internal sudah diset untuk ini.
        # Untuk sekarang, kita asumsikan ga_core.run() tidak butuh argumen tambahan ini secara langsung
        # karena sudah di-pass saat inisialisasi atau diambil dari self.
        # Namun, fungsi calculate_combined_fitness di dalam GA akan butuh argumen ini.
        # Cara paling bersih adalah memodifikasi ga_core agar bisa menyimpan/menerima ini.
        
        # Asumsi ga_core.run() sudah dimodifikasi untuk mengambil `target_genus_specie` dan `user_input_dict`
        # dari `self` (yang di-set saat `__init__`) atau menerimanya sebagai argumen `run`.
        # Kita akan mengasumsikan ini sudah di-set saat inisialisasi `ga_simulator`.

        best_chromosome_list, best_fitness, _, evolution_log_tuples = ga_simulator.run()
        # evolution_log_tuples adalah list of tuples (generation, fitness, best_chromosome_list_for_gen)

        # 4. Format hasil
        final_best_features_dict = {feature: best_chromosome_list[i] for i, feature in enumerate(FEATURE_ORDER)}
        
        evolution_path_formatted: List[FeatureEvolutionStep] = []
        for gen_data in evolution_log_tuples:
            gen_num, fit_val, chromo_list = gen_data
            chromo_dict = {feature: chromo_list[i] for i, feature in enumerate(FEATURE_ORDER)}
            evolution_path_formatted.append(
                FeatureEvolutionStep(generation=gen_num, fitness=fit_val, features=chromo_dict)
            )

        return SimulationResponse(
            message="Simulasi evolusi berhasil diselesaikan.",
            target_genus_specie=request_data.target_genus_specie,
            final_best_fitness=best_fitness,
            final_best_features=final_best_features_dict,
            evolution_path=evolution_path_formatted,
            input_features_processed=user_params_for_fitness
        )

    except ImportError as e: # Menangkap error impor modul GA jika terjadi di sini
        raise HTTPException(status_code=500, detail=f"Kesalahan impor modul internal: {str(e)}")
    except ValueError as ve: # Misal error dari Pydantic atau konversi data
        raise HTTPException(status_code=400, detail=f"Input tidak valid: {str(ve)}")
    except Exception as e:
        # Tangkap error spesifik lainnya jika perlu
        import traceback
        traceback.print_exc() # Untuk debugging di server
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan internal server: {str(e)}")


# --- Untuk menjalankan dengan Uvicorn (misal dari direktori 'backend'): ---
# uvicorn app.api:app --reload
# atau dari root TUBES_KDS: uvicorn backend.app.api:app --reload

if __name__ == "__main__":
    # Bagian ini hanya untuk pengujian jika file dijalankan langsung,
    # tapi FastAPI biasanya dijalankan dengan Uvicorn.
    # Untuk pengujian API, gunakan Postman atau curl.
    print("API server dapat dijalankan dengan Uvicorn.")
    print("Contoh: uvicorn backend.app.api:app --reload --port 8000")
    print(f"Dataset path yang akan dicoba: {DATASET_PATH}")
    print(f"CWD saat ini: {os.getcwd()}")
    # Coba load dataset di sini untuk debugging path
    # if not evolution_df and not data_load_error:
    #     loop = asyncio.get_event_loop()
    #     loop.run_until_complete(load_dataset())
    #     if data_load_error:
    #         print(f"Error saat load dataset di main: {data_load_error}")
    #     elif evolution_df is not None:
    #         print("Dataset berhasil di-load di main.")