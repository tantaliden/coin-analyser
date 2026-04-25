import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import api from '../utils/api'

export const useAuthStore = create(
  persist(
    (set, get) => ({
      user: null,
      accessToken: null,
      refreshToken: null,
      isAuthenticated: false,

      login: async (email, password) => {
        try {
          const response = await api.post('/api/v1/auth/login', { email, password })
          const { access_token, refresh_token, user } = response.data
          set({ 
            user, 
            accessToken: access_token, 
            refreshToken: refresh_token, 
            isAuthenticated: true 
          })
          return { success: true }
        } catch (error) {
          return { 
            success: false, 
            error: error.response?.data?.detail || 'Login fehlgeschlagen' 
          }
        }
      },

      logout: async () => {
        try {
          await api.post('/api/v1/auth/logout')
        } catch {}
        set({ 
          user: null, 
          accessToken: null, 
          refreshToken: null, 
          isAuthenticated: false 
        })
      },

      checkAuth: async () => {
        const { accessToken } = get()
        if (!accessToken) {
          set({ isAuthenticated: false })
          return false
        }
        try {
          const response = await api.get('/api/v1/auth/me')
          set({ user: response.data, isAuthenticated: true })
          return true
        } catch {
          // Token abgelaufen - versuche Refresh
          const { refreshToken } = get()
          if (refreshToken) {
            try {
              const refreshResponse = await api.post('/api/v1/auth/refresh', null, {
                params: { refresh_token: refreshToken }
              })
              set({ accessToken: refreshResponse.data.access_token })
              return true
            } catch {}
          }
          get().logout()
          return false
        }
      },

      getToken: () => get().accessToken
    }),
    { 
      name: 'coin-auth', 
      partialize: (state) => ({ 
        accessToken: state.accessToken, 
        refreshToken: state.refreshToken, 
        user: state.user 
      }) 
    }
  )
)
