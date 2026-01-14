import React from 'react'
import { MapPin, Droplets, AlertTriangle, CheckCircle, Gauge } from 'lucide-react'

export default function StatsBar({ summary, loading }) {
  if (loading) {
    return (
      <div className="bg-white border-b px-4 py-2 flex gap-4 animate-pulse">
        {[1, 2, 3, 4, 5].map(i => (
          <div key={i} className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gray-200 rounded"></div>
            <div>
              <div className="w-16 h-3 bg-gray-200 rounded mb-1"></div>
              <div className="w-12 h-4 bg-gray-200 rounded"></div>
            </div>
          </div>
        ))}
      </div>
    )
  }

  const stats = [
    {
      icon: MapPin,
      label: 'Villages',
      value: summary?.total_villages || 0,
      color: 'text-blue-500'
    },
    {
      icon: Gauge,
      label: 'Piezometers',
      value: summary?.piezometer_count || 0,
      color: 'text-purple-500'
    },
    {
      icon: Droplets,
      label: 'Avg Depth',
      value: summary?.avg_water_level ? `${summary.avg_water_level.toFixed(1)}m` : '-',
      color: 'text-cyan-500'
    },
    {
      icon: CheckCircle,
      label: 'Safe',
      value: summary?.safe_villages || 0,
      color: 'text-green-500'
    },
    {
      icon: AlertTriangle,
      label: 'Critical',
      value: summary?.critical_villages || 0,
      color: 'text-red-500'
    }
  ]

  return (
    <div className="bg-white border-b px-4 py-2 flex gap-6 overflow-x-auto">
      {stats.map((stat, idx) => (
        <div key={idx} className="flex items-center gap-2 min-w-fit">
          <stat.icon className={`w-6 h-6 ${stat.color}`} />
          <div>
            <p className="text-xs text-gray-500">{stat.label}</p>
            <p className="font-semibold text-gray-800">{stat.value}</p>
          </div>
        </div>
      ))}
    </div>
  )
}
