import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/Home.vue'

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomeView
  }
  // Tidak ada rute lain karena hanya satu halaman
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router