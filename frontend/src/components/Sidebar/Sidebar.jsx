import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { X, MapPin, Droplets, TrendingDown, TrendingUp, Minus, Loader2 } from 'lucide-react'
import { fetchTimeSeries, fetchAquiferComparison, fetchTrends } from '../../services/api'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

// Category badge component
function CategoryBadge({ category }) {
  const colors = {
    safe: 'bg-green-100 text-green-800',
    moderate: 'bg-yellow-100 text-yellow-800',
    stress: 'bg-orange-100 text-orange-800',
    critical: 'bg-red-100 text-red-800'
  }

  return (
    <span className={`px-2 py-1 rounded-full text-xs font-medium ${colors[category] || 'bg-gray-100 text-gray-800'}`}>
      {category}
    </span>
  )
}

// Village details panel
function VillageDetails({ feature }) {
  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold text-gray-800">{feature.village}</h3>
        <p className="text-sm text-gray-500">{feature.mandal}, {feature.district}</p>
      </div>

      <div className="bg-blue-50 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-600">Predicted Water Level</p>
            <p className="text-2xl font-bold text-gray-800">
              {feature.water_level_m?.toFixed(1) || '-'}
              <span className="text-sm font-normal text-gray-500 ml-1">m</span>
            </p>
          </div>
          <CategoryBadge category={feature.category} />
        </div>
        {feature.confidence && (
          <p className="text-xs text-gray-500 mt-2">
            Confidence: {(feature.confidence * 100).toFixed(0)}%
          </p>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="bg-gray-50 rounded p-3">
          <p className="text-xs text-gray-500">Wells</p>
          <p className="font-semibold">{feature.well_count || '-'}</p>
        </div>
        <div className="bg-gray-50 rounded p-3">
          <p className="text-xs text-gray-500">Avg Bore Depth</p>
          <p className="font-semibold">
            {feature.avg_bore_depth ? `${feature.avg_bore_depth.toFixed(0)}m` : '-'}
          </p>
        </div>
      </div>
    </div>
  )
}

// Piezometer details panel
function PiezometerDetails({ feature }) {
  const { data: timeSeries, isLoading } = useQuery({
    queryKey: ['timeseries', feature.piezo_id],
    queryFn: () => fetchTimeSeries(feature.piezo_id),
    enabled: !!feature.piezo_id,
  })

  // Prepare chart data
  const chartData = timeSeries?.data?.slice(-24).map(d => ({
    date: new Date(d.date).toLocaleDateString('en-US', { month: 'short', year: '2-digit' }),
    level: d.water_level
  })) || []

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold text-gray-800">Piezometer {feature.piezo_id}</h3>
        <p className="text-sm text-gray-500">{feature.village}, {feature.mandal}</p>
      </div>

      <div className="bg-blue-50 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-600">Latest Water Level</p>
            <p className="text-2xl font-bold text-gray-800">
              {feature.latest_water_level?.toFixed(1) || '-'}
              <span className="text-sm font-normal text-gray-500 ml-1">m</span>
            </p>
          </div>
          <Droplets className="w-8 h-8 text-blue-500" />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="bg-gray-50 rounded p-3">
          <p className="text-xs text-gray-500">Aquifer</p>
          <p className="font-semibold text-sm">{feature.aquifer || '-'}</p>
        </div>
        <div className="bg-gray-50 rounded p-3">
          <p className="text-xs text-gray-500">Total Depth</p>
          <p className="font-semibold">{feature.total_depth ? `${feature.total_depth}m` : '-'}</p>
        </div>
      </div>

      {/* Time series chart */}
      <div className="mt-4">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Historical Trend</h4>
        {isLoading ? (
          <div className="h-32 flex items-center justify-center">
            <Loader2 className="w-6 h-6 animate-spin text-gray-400" />
          </div>
        ) : chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={150}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis
                tick={{ fontSize: 10 }}
                domain={['auto', 'auto']}
                label={{ value: 'm', angle: -90, position: 'insideLeft', fontSize: 10 }}
              />
              <Tooltip
                contentStyle={{ fontSize: 12 }}
                formatter={(value) => [`${value.toFixed(1)}m`, 'Water Level']}
              />
              <Line
                type="monotone"
                dataKey="level"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-sm text-gray-500 text-center py-4">No historical data available</p>
        )}
      </div>

      {timeSeries?.statistics && (
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="bg-gray-50 rounded p-2">
            <p className="text-xs text-gray-500">Min</p>
            <p className="text-sm font-semibold">{timeSeries.statistics.min.toFixed(1)}m</p>
          </div>
          <div className="bg-gray-50 rounded p-2">
            <p className="text-xs text-gray-500">Mean</p>
            <p className="text-sm font-semibold">{timeSeries.statistics.mean.toFixed(1)}m</p>
          </div>
          <div className="bg-gray-50 rounded p-2">
            <p className="text-xs text-gray-500">Max</p>
            <p className="text-sm font-semibold">{timeSeries.statistics.max.toFixed(1)}m</p>
          </div>
        </div>
      )}
    </div>
  )
}

// Trends panel (shown when no feature selected)
function TrendsPanel() {
  const { data: trends, isLoading } = useQuery({
    queryKey: ['trends'],
    queryFn: () => fetchTrends(5),
  })

  const { data: comparison } = useQuery({
    queryKey: ['aquifer-comparison'],
    queryFn: fetchAquiferComparison,
  })

  const trendIcon = {
    improving: <TrendingUp className="w-5 h-5 text-green-500" />,
    stable: <Minus className="w-5 h-5 text-yellow-500" />,
    declining: <TrendingDown className="w-5 h-5 text-red-500" />
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-800">District Overview</h3>

      {/* Trend summary */}
      {trends && (
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">5-Year Trend</span>
            {trendIcon[trends.trend_direction]}
          </div>
          <p className="text-lg font-semibold capitalize">{trends.trend_direction}</p>
          <p className="text-xs text-gray-500">
            Change: {trends.change_rate_per_year > 0 ? '+' : ''}{trends.change_rate_per_year.toFixed(2)} m/year
          </p>
        </div>
      )}

      {/* Aquifer comparison */}
      {comparison?.comparison && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-2">Aquifer Comparison</h4>
          <div className="space-y-2">
            {comparison.comparison.slice(0, 5).map((aq, idx) => (
              <div key={idx} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                <div>
                  <p className="text-sm font-medium">{aq.aquifer_type}</p>
                  <p className="text-xs text-gray-500">{aq.piezometer_count} piezometers</p>
                </div>
                {aq.statistics?.mean_water_level && (
                  <span className="text-sm font-semibold">
                    {aq.statistics.mean_water_level.toFixed(1)}m
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <p className="text-xs text-gray-400 text-center">
        Click on a village or piezometer for details
      </p>
    </div>
  )
}

// Main Sidebar component
export default function Sidebar({ selectedFeature, onClose }) {
  return (
    <div className="w-80 bg-white border-l shadow-lg flex flex-col">
      {/* Header */}
      <div className="p-4 border-b flex items-center justify-between">
        <div className="flex items-center gap-2">
          <MapPin className="w-5 h-5 text-blue-500" />
          <span className="font-medium">
            {selectedFeature ? 'Details' : 'Overview'}
          </span>
        </div>
        <button
          onClick={onClose}
          className="p-1 hover:bg-gray-100 rounded"
        >
          <X className="w-5 h-5 text-gray-500" />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {selectedFeature ? (
          selectedFeature.type === 'village' ? (
            <VillageDetails feature={selectedFeature} />
          ) : selectedFeature.type === 'piezometer' ? (
            <PiezometerDetails feature={selectedFeature} />
          ) : (
            <TrendsPanel />
          )
        ) : (
          <TrendsPanel />
        )}
      </div>
    </div>
  )
}
