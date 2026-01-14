import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import MapContainer from './components/Map/MapContainer'
import Sidebar from './components/Sidebar/Sidebar'
import StatsBar from './components/Dashboard/StatsBar'
import { fetchSummary, fetchHealth } from './services/api'
import { Droplets, AlertCircle, Loader2 } from 'lucide-react'

function App() {
  const [selectedFeature, setSelectedFeature] = useState(null)
  const [activeLayer, setActiveLayer] = useState('villages')
  const [sidebarOpen, setSidebarOpen] = useState(true)

  // Fetch health status
  const { data: health, isLoading: healthLoading, error: healthError } = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    retry: 3,
    retryDelay: 1000,
  })

  // Fetch summary stats
  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: ['summary'],
    queryFn: fetchSummary,
    enabled: !!health,
  })

  // Handle feature selection from map
  const handleFeatureSelect = (feature) => {
    setSelectedFeature(feature)
    if (!sidebarOpen) setSidebarOpen(true)
  }

  // Show loading state
  if (healthLoading) {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-100">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-blue-500 mx-auto mb-4" />
          <p className="text-gray-600">Connecting to SmartJal API...</p>
        </div>
      </div>
    )
  }

  // Show error state if API is down
  if (healthError) {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-100">
        <div className="text-center max-w-md p-6 bg-white rounded-lg shadow-lg">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-800 mb-2">API Connection Error</h2>
          <p className="text-gray-600 mb-4">
            Unable to connect to the SmartJal API. Please ensure the backend server is running.
          </p>
          <code className="text-sm bg-gray-100 p-2 rounded block mb-4">
            cd backend && uvicorn app.main:app --reload
          </code>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Retry Connection
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="h-screen flex flex-col bg-gray-100">
      {/* Header */}
      <header className="bg-blue-600 text-white px-4 py-3 flex items-center justify-between shadow-lg z-10">
        <div className="flex items-center gap-3">
          <Droplets className="w-8 h-8" />
          <div>
            <h1 className="text-xl font-bold">SmartJal</h1>
            <p className="text-xs text-blue-200">Krishna District Groundwater Dashboard</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-sm">
            <span className={`inline-block w-2 h-2 rounded-full mr-2 ${health?.status === 'healthy' ? 'bg-green-400' : 'bg-red-400'}`}></span>
            {health?.status === 'healthy' ? 'Connected' : 'Disconnected'}
          </div>
        </div>
      </header>

      {/* Stats Bar */}
      <StatsBar summary={summary} loading={summaryLoading} />

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Map */}
        <div className="flex-1 relative">
          <MapContainer
            onFeatureSelect={handleFeatureSelect}
            activeLayer={activeLayer}
          />

          {/* Layer Controls */}
          <div className="absolute top-4 right-4 bg-white rounded-lg shadow-lg p-3 z-[1000]">
            <h3 className="text-sm font-semibold text-gray-700 mb-2">Layers</h3>
            <div className="space-y-2">
              {['villages', 'piezometers', 'aquifers', 'wells'].map(layer => (
                <label key={layer} className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name="layer"
                    checked={activeLayer === layer}
                    onChange={() => setActiveLayer(layer)}
                    className="text-blue-500"
                  />
                  <span className="text-sm capitalize">{layer}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Legend */}
          <div className="absolute bottom-4 left-4 bg-white rounded-lg shadow-lg p-3 z-[1000]">
            <h3 className="text-sm font-semibold text-gray-700 mb-2">Water Depth</h3>
            <div className="space-y-1 text-xs">
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-safe"></span>
                <span>Safe (0-3m)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-moderate"></span>
                <span>Moderate (3-8m)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-stress"></span>
                <span>Stress (8-20m)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-critical"></span>
                <span>Critical (&gt;20m)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Sidebar */}
        {sidebarOpen && (
          <Sidebar
            selectedFeature={selectedFeature}
            onClose={() => setSidebarOpen(false)}
          />
        )}
      </div>
    </div>
  )
}

export default App
