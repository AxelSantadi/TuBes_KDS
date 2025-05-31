import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
// import store from './store' // Jika Anda menggunakan Vuex

createApp(App).use(router).mount('#app') // Tambahkan .use(store) jika menggunakan Vuex