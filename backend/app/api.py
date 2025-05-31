from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from .genetic_algorithm.core import genetic_algorithm_pipeline # Sesuaikan path jika perlu

app = FastAPI()

# CORS Middleware (penting untuk pengembangan lokal dengan frontend terpisah)
origins = [
    "http://localhost:8080", # Sesuaikan dengan port frontend Vue Anda
    "http://localhost:8081", # Jika port berbeda
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/run_ga")
async def run_ga_endpoint(
    file: UploadFile = File(...), # Input: Data fitur genetik manusia dalam format CSV [cite: 8]
    population_size: int = Form(...), # Parameter GA dari frontend [cite: 38]
    num_generations: int = Form(...), # Parameter GA dari frontend [cite: 38]
    crossover_probability: float = Form(...), # Parameter GA dari frontend [cite: 38]
    mutation_probability: float = Form(...), # Parameter GA dari frontend [cite: 38]
    # target_column: str = Form(...) # Nama kolom label/target
):
    try:
        contents = await file.read()
        # Asumsi file CSV "Evolution_DataSets.csv" [cite: 31]
        # Anda mungkin perlu memproses file CSV ini lebih lanjut
        # Misalnya, memisahkan fitur (X) dan label (y)
        data_df = pd.read_csv(io.BytesIO(contents)) 

        # Placeholder untuk data dan label
        # features_df = data_df.drop(columns=[target_column])
        # labels_series = data_df[target_column]

        # Panggil fungsi GA Anda
        # best_features_chromosome, best_fitness = genetic_algorithm_pipeline(
        #     data=features_df, # Ini harus berupa data yang sudah diproses (numerik)
        #     labels=labels_series, 
        #     pop_size=population_size,
        #     num_generations=num_generations,
        #     crossover_prob=crossover_probability,
        #     mutation_prob=mutation_probability
        # )

        # Simulasi hasil GA
        best_features_chromosome = [1,0,1,0,1] + [0]*22 # Contoh
        best_fitness = 0.95

        # Konversi kromosom terbaik ke daftar nama fitur (jika Anda punya daftar nama fitur)
        # column_names = features_df.columns.tolist()
        # selected_features = [col for i, col in enumerate(column_names) if best_features_chromosome[i] == 1]
        selected_features_placeholder = ['Current_Country', 'Habitat', 'Canine Size', 'Arms', 'Diet']


        return {
            "message": "GA process completed successfully!",
            "selected_features_chromosome": best_features_chromosome,
            "best_fitness": best_fitness,
            "selected_features_names": selected_features_placeholder, # Output: Hasil klasifikasi/fitur terpilih [cite: 9, 41]
            # Anda juga bisa mengembalikan metrik lain seperti grafik konvergensi (sebagai data JSON atau link) [cite: 42]
        }
    except Exception as e:
        return {"error": str(e)}

# Jalankan dengan: uvicorn backend.app.api:app --reload (dari direktori TuBes_KDS)