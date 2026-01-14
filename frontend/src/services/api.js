import axios from 'axios'

const API_BASE = '/api/v1'

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
})

// GeoJSON endpoints
export const fetchAquifersGeoJSON = () => api.get('/geojson/aquifers').then(r => r.data)
export const fetchPiezometersGeoJSON = () => api.get('/geojson/piezometers').then(r => r.data)
export const fetchVillagesGeoJSON = (limit = 500) => api.get(`/geojson/villages?limit=${limit}`).then(r => r.data)
export const fetchWellsGeoJSON = (limit = 1000) => api.get(`/geojson/wells?limit=${limit}`).then(r => r.data)
export const fetchBounds = () => api.get('/geojson/bounds').then(r => r.data)

// Predictions
export const predictVillage = (data) => api.post('/predictions/village', data).then(r => r.data)
export const fetchTimeSeries = (piezoId) => api.get(`/predictions/timeseries/${piezoId}`).then(r => r.data)

// Analytics
export const fetchSummary = () => api.get('/analytics/summary').then(r => r.data)
export const fetchTrends = (years = 5) => api.get(`/analytics/trends?years=${years}`).then(r => r.data)
export const fetchSeasonalAnalysis = (year) => api.get(`/analytics/seasonal${year ? `?year=${year}` : ''}`).then(r => r.data)
export const fetchAquiferComparison = () => api.get('/analytics/aquifer-comparison').then(r => r.data)

// Aquifers
export const fetchAquifers = () => api.get('/aquifers').then(r => r.data)
export const fetchAquiferStats = (code) => api.get(`/aquifers/${code}/statistics`).then(r => r.data)

// Villages
export const fetchVillages = (params = {}) => {
  const query = new URLSearchParams(params).toString()
  return api.get(`/villages${query ? `?${query}` : ''}`).then(r => r.data)
}
export const fetchMandals = () => api.get('/villages/mandals/list').then(r => r.data)

// Health - note: health endpoint is at root, not under /api/v1
export const fetchHealth = () => axios.get('/api/health').then(r => r.data)

export default api
