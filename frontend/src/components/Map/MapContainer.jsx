import React, { useEffect, useState } from 'react'
import { MapContainer as LeafletMap, TileLayer, GeoJSON, CircleMarker, Popup, useMap } from 'react-leaflet'
import { useQuery } from '@tanstack/react-query'
import {
  fetchAquifersGeoJSON,
  fetchPiezometersGeoJSON,
  fetchVillagesGeoJSON,
  fetchWellsGeoJSON,
  fetchBounds
} from '../../services/api'

// Category colors
const categoryColors = {
  safe: '#22c55e',
  moderate: '#eab308',
  stress: '#f97316',
  critical: '#ef4444'
}

// Aquifer colors
const aquiferColors = {
  'BG': '#8b5cf6',
  'ST': '#06b6d4',
  'SH': '#f59e0b',
  'LS': '#10b981',
  'QZ': '#ec4899',
  'default': '#6b7280'
}

// Map bounds controller
function MapBoundsController({ bounds }) {
  const map = useMap()

  useEffect(() => {
    if (bounds?.bounds) {
      map.fitBounds(bounds.bounds, { padding: [20, 20] })
    }
  }, [bounds, map])

  return null
}

// Village Layer
function VillageLayer({ data, onFeatureSelect }) {
  if (!data?.features) return null

  return (
    <>
      {data.features.map((feature, idx) => {
        const { coordinates } = feature.geometry
        const props = feature.properties
        const color = categoryColors[props.category] || '#6b7280'

        return (
          <CircleMarker
            key={`village-${idx}`}
            center={[coordinates[1], coordinates[0]]}
            radius={6}
            pathOptions={{
              color: 'white',
              weight: 2,
              fillColor: color,
              fillOpacity: 0.8
            }}
            eventHandlers={{
              click: () => onFeatureSelect({ type: 'village', ...props, coordinates })
            }}
          >
            <Popup>
              <div className="text-sm">
                <h3 className="font-semibold">{props.village}</h3>
                <p className="text-gray-600">{props.mandal}</p>
                {props.water_level_m && (
                  <p className="mt-1">
                    Water Level: <strong>{props.water_level_m.toFixed(1)}m</strong>
                  </p>
                )}
                <p>
                  Status: <span style={{ color }}>{props.category}</span>
                </p>
              </div>
            </Popup>
          </CircleMarker>
        )
      })}
    </>
  )
}

// Piezometer Layer
function PiezometerLayer({ data, onFeatureSelect }) {
  if (!data?.features) return null

  return (
    <>
      {data.features.map((feature, idx) => {
        const { coordinates } = feature.geometry
        const props = feature.properties

        return (
          <CircleMarker
            key={`piezo-${idx}`}
            center={[coordinates[1], coordinates[0]]}
            radius={8}
            pathOptions={{
              color: 'white',
              weight: 2,
              fillColor: '#3b82f6',
              fillOpacity: 0.9
            }}
            eventHandlers={{
              click: () => onFeatureSelect({ type: 'piezometer', ...props, coordinates })
            }}
          >
            <Popup>
              <div className="text-sm">
                <h3 className="font-semibold">Piezometer {props.piezo_id}</h3>
                <p className="text-gray-600">{props.village}, {props.mandal}</p>
                {props.latest_water_level && (
                  <p className="mt-1">
                    Latest Level: <strong>{props.latest_water_level.toFixed(1)}m</strong>
                  </p>
                )}
                <p>Aquifer: {props.aquifer}</p>
              </div>
            </Popup>
          </CircleMarker>
        )
      })}
    </>
  )
}

// Aquifer Layer
function AquiferLayer({ data }) {
  if (!data?.features) return null

  const style = (feature) => ({
    color: aquiferColors[feature.properties.aquifer_code] || aquiferColors.default,
    weight: 2,
    fillOpacity: 0.2
  })

  const onEachFeature = (feature, layer) => {
    layer.bindPopup(`
      <div class="text-sm">
        <h3 class="font-semibold">${feature.properties.aquifer_type}</h3>
        <p>Code: ${feature.properties.aquifer_code}</p>
        ${feature.properties.area_sqkm ? `<p>Area: ${feature.properties.area_sqkm.toFixed(1)} km²</p>` : ''}
      </div>
    `)
  }

  return (
    <GeoJSON
      data={data}
      style={style}
      onEachFeature={onEachFeature}
    />
  )
}

// Wells Layer
function WellsLayer({ data }) {
  if (!data?.features) return null

  return (
    <>
      {data.features.slice(0, 500).map((feature, idx) => {
        const { coordinates } = feature.geometry
        const props = feature.properties

        return (
          <CircleMarker
            key={`well-${idx}`}
            center={[coordinates[1], coordinates[0]]}
            radius={3}
            pathOptions={{
              color: '#94a3b8',
              weight: 1,
              fillColor: '#64748b',
              fillOpacity: 0.6
            }}
          >
            <Popup>
              <div className="text-sm">
                <h3 className="font-semibold">{props.village}</h3>
                <p>Type: {props.well_type}</p>
                {props.bore_depth && <p>Depth: {props.bore_depth}m</p>}
                {props.crop_type && <p>Crop: {props.crop_type}</p>}
              </div>
            </Popup>
          </CircleMarker>
        )
      })}
    </>
  )
}

// Main Map Container
export default function MapContainer({ onFeatureSelect, activeLayer }) {
  const [mapReady, setMapReady] = useState(false)

  // Fetch bounds
  const { data: bounds } = useQuery({
    queryKey: ['bounds'],
    queryFn: fetchBounds,
  })

  // Fetch data based on active layer
  const { data: villagesData } = useQuery({
    queryKey: ['villages-geojson'],
    queryFn: () => fetchVillagesGeoJSON(500),
    enabled: activeLayer === 'villages',
  })

  const { data: piezometersData } = useQuery({
    queryKey: ['piezometers-geojson'],
    queryFn: fetchPiezometersGeoJSON,
    enabled: activeLayer === 'piezometers',
  })

  const { data: aquifersData } = useQuery({
    queryKey: ['aquifers-geojson'],
    queryFn: fetchAquifersGeoJSON,
    enabled: activeLayer === 'aquifers' || activeLayer === 'villages',
  })

  const { data: wellsData } = useQuery({
    queryKey: ['wells-geojson'],
    queryFn: () => fetchWellsGeoJSON(1000),
    enabled: activeLayer === 'wells',
  })

  const defaultCenter = bounds?.center || [16.25, 80.75]

  return (
    <LeafletMap
      center={defaultCenter}
      zoom={10}
      className="h-full w-full"
      whenReady={() => setMapReady(true)}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      {mapReady && bounds && <MapBoundsController bounds={bounds} />}

      {/* Always show aquifer boundaries as base layer */}
      {aquifersData && (activeLayer === 'aquifers' || activeLayer === 'villages') && (
        <AquiferLayer data={aquifersData} />
      )}

      {/* Show active layer */}
      {activeLayer === 'villages' && villagesData && (
        <VillageLayer data={villagesData} onFeatureSelect={onFeatureSelect} />
      )}

      {activeLayer === 'piezometers' && piezometersData && (
        <PiezometerLayer data={piezometersData} onFeatureSelect={onFeatureSelect} />
      )}

      {activeLayer === 'wells' && wellsData && (
        <WellsLayer data={wellsData} />
      )}
    </LeafletMap>
  )
}
