<template>
  <div class="home">
    <h1>Analisis Evolusi Genetik menggunakan Genetic Algorithm</h1>

    <section class="data-upload">
      <h2>1. Unggah Dataset</h2>
      <input type="file" @change="handleFileUpload" accept=".csv" />
      <div v-if="fileName">
        <p>File terpilih: {{ fileName }}</p>
        </div>
    </section>

    <section class="ga-parameters">
      <h2>2. Atur Parameter Genetic Algorithm</h2>
      <div>
        <label for="popSize">Ukuran Populasi (P): {{ gaParams.populationSize }}</label>
        <input type="range" id="popSize" min="10" max="200" step="10" v-model.number="gaParams.populationSize" /> 
        </div>
      <div>
        <label for="generations">Jumlah Generasi (Gmax): {{ gaParams.numGenerations }}</label>
        <input type="range" id="generations" min="10" max="100" step="10" v-model.number="gaParams.numGenerations" />
        </div>
      <div>
        <label for="crossoverProb">Probabilitas Crossover (pc): {{ gaParams.crossoverProbability }}</label>
        <input type="range" id="crossoverProb" min="0.1" max="1.0" step="0.05" v-model.number="gaParams.crossoverProbability" />
        </div>
      <div>
        <label for="mutationProb">Probabilitas Mutasi (pm): {{ gaParams.mutationProbability }}</label>
        <input type="range" id="mutationProb" min="0.01" max="0.2" step="0.01" v-model.number="gaParams.mutationProbability" />
        </div>
    </section>

    <section class="run-ga">
      <h2>3. Jalankan Analisis</h2>
      <button @click="runGA" :disabled="!file || isLoading">
        {{ isLoading ? 'Memproses...' : 'Run GA' }}
      </button>
      <div v-if="isLoading" class="progress-bar">
        <p>Loading... (Ini bisa diganti dengan progress bar aktual)</p>
      </div>
    </section>

    <section class="results" v-if="results">
      <h2>4. Hasil Analisis</h2>
      <h3>Fitur Terpilih:</h3>
      <ul v-if="results.selected_features_names && results.selected_features_names.length">
        <li v-for="feature in results.selected_features_names" :key="feature">{{ feature }}</li>
      </ul>
      <p v-else>Tidak ada fitur yang terpilih atau hasil belum tersedia.</p>

      <p><strong>Fitness Terbaik:</strong> {{ results.best_fitness }}</p>

      </section>

    <section class="logs" v-if="gaLog.length > 0">
        <h2>Log GA:</h2>
        <div v-for="(logEntry, index) in gaLog" :key="index">
            {{ logEntry }}
        </div>
    </section>

    <div v-if="error" class="error-message">
      <p>Error: {{ error }}</p>
    </div>
  </div>
</template>

<script>
import axios from 'axios'; // Untuk memanggil API backend

export default {
  name: 'HomeView',
  data() {
    return {
      file: null,
      fileName: '',
      gaParams: { // Nilai awal parameter GA [cite: 38]
        populationSize: 50, // Contoh nilai awal, sesuai debugging pipeline [cite: 46]
        numGenerations: 20, // Contoh nilai awal, sesuai debugging pipeline [cite: 46]
        crossoverProbability: 0.8, // Contoh nilai [cite: 47]
        mutationProbability: 0.01, // Contoh nilai [cite: 47]
      },
      isLoading: false,
      results: null,
      error: null,
      gaLog: [], // Untuk menyimpan log fitness per generasi [cite: 40]
    };
  },
  methods: {
    handleFileUpload(event) {
      this.file = event.target.files[0];
      this.fileName = this.file ? this.file.name : '';
      this.results = null; // Reset hasil jika file baru diunggah
      this.error = null;
      this.gaLog = [];
      // Idealnya, di sini juga bisa ada pratinjau data [cite: 37]
    },
    async runGA() {
      if (!this.file) {
        this.error = "Silakan unggah file dataset terlebih dahulu.";
        return;
      }
      this.isLoading = true;
      this.results = null;
      this.error = null;
      this.gaLog = ["Memulai proses GA..."]; // [cite: 40]

      const formData = new FormData();
      formData.append('file', this.file);
      formData.append('population_size', this.gaParams.populationSize);
      formData.append('num_generations', this.gaParams.numGenerations);
      formData.append('crossover_probability', this.gaParams.crossoverProbability);
      formData.append('mutation_probability', this.gaParams.mutationProbability);
      // formData.append('target_column', 'Genus_&_Specie'); // Sesuaikan dengan nama kolom label Anda [cite: 32]

      try {
        // Ganti URL dengan endpoint backend Anda
        const response = await axios.post('http://localhost:8000/run_ga', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });
        // Asumsi backend mengembalikan log atau progress yang bisa ditambahkan ke gaLog
        // this.gaLog.push("Proses GA selesai."); // [cite: 40]
        this.results = response.data;
        if (response.data.error) {
            this.error = response.data.error;
            this.results = null;
        }
        // Contoh: this.gaLog = response.data.logs; // Jika backend mengirim log [cite: 40]

      } catch (err) {
        this.error = `Gagal menjalankan GA: ${err.response ? err.response.data.error || err.message : err.message}`;
        this.results = null;
      } finally {
        this.isLoading = false;
      }
    }
  }
};
</script>

<style scoped>
.home {
  max-width: 800px;
  margin: 20px auto;
  padding: 20px;
  font-family: sans-serif;
}
section {
  margin-bottom: 30px;
  padding: 15px;
  border: 1px solid #eee;
  border-radius: 5px;
}
h1, h2, h3 {
  color: #333;
}
label {
  display: block;
  margin-bottom: 5px;
}
input[type="range"] {
  width: 100%;
}
button {
  padding: 10px 15px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}
button:disabled {
  background-color: #ccc;
}
.error-message {
  color: red;
  margin-top: 15px;
}
.progress-bar {
    margin-top: 10px;
    padding: 10px;
    background-color: #f0f0f0;
}
.logs div {
    padding: 2px 5px;
    border-bottom: 1px dotted #eee;
}
</style>