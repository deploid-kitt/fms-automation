import React, { useState, useEffect, useCallback } from 'react'

// Backend/size badges
const BackendBadge = ({ backend }) => {
  const config = {
    mediapipe: { bg: 'bg-green-100', text: 'text-green-700', label: 'MediaPipe' },
    rtmpose: { bg: 'bg-blue-100', text: 'text-blue-700', label: 'RTMPose' },
    yolo: { bg: 'bg-purple-100', text: 'text-purple-700', label: 'YOLO' },
  }
  const c = config[backend] || { bg: 'bg-gray-100', text: 'text-gray-700', label: backend }
  
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${c.bg} ${c.text}`}>
      {c.label}
    </span>
  )
}

const SizeBadge = ({ size }) => {
  const config = {
    nano: { bg: 'bg-green-50', text: 'text-green-600', icon: '⚡' },
    small: { bg: 'bg-blue-50', text: 'text-blue-600', icon: '🔹' },
    medium: { bg: 'bg-yellow-50', text: 'text-yellow-600', icon: '🔸' },
    large: { bg: 'bg-orange-50', text: 'text-orange-600', icon: '🔶' },
    heavy: { bg: 'bg-red-50', text: 'text-red-600', icon: '🔥' },
  }
  const c = config[size] || { bg: 'bg-gray-50', text: 'text-gray-600', icon: '•' }
  
  return (
    <span className={`px-2 py-0.5 rounded text-xs ${c.bg} ${c.text}`}>
      {c.icon} {size}
    </span>
  )
}

// Model card component
function ModelCard({ model, isSelected, onSelect, disabled }) {
  const isAvailable = model.is_available
  
  return (
    <button
      onClick={() => onSelect(model.model_id)}
      disabled={disabled || !isAvailable}
      className={`
        p-4 rounded-lg border-2 text-left transition-all
        ${isSelected 
          ? 'border-primary-500 bg-primary-50' 
          : 'border-gray-200 hover:border-gray-300 bg-white'}
        ${!isAvailable ? 'opacity-50 cursor-not-allowed' : 'hover:shadow-md'}
        ${disabled ? 'opacity-75' : ''}
      `}
    >
      <div className="flex items-start justify-between mb-2">
        <div>
          <h4 className="font-medium text-gray-800">{model.name}</h4>
          <div className="flex items-center gap-2 mt-1">
            <BackendBadge backend={model.backend} />
            <SizeBadge size={model.model_size} />
          </div>
        </div>
        {isSelected && (
          <div className="w-6 h-6 rounded-full bg-primary-500 flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
          </div>
        )}
      </div>
      
      <p className="text-sm text-gray-600 mb-2 line-clamp-2">
        {model.description}
      </p>
      
      <div className="flex flex-wrap gap-1">
        {model.recommended_for?.map(tag => (
          <span 
            key={tag} 
            className="text-xs px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded"
          >
            {tag === 'live' ? '🎥 Live' : 
             tag === 'upload' ? '📤 Upload' : 
             tag === 'speed' ? '⚡ Fast' : 
             tag === 'accuracy' ? '🎯 Accurate' : tag}
          </span>
        ))}
      </div>
      
      {model.estimated_fps > 0 && (
        <div className="mt-2 text-xs text-gray-500">
          ~{model.estimated_fps.toFixed(0)} FPS
        </div>
      )}
      
      {!isAvailable && (
        <div className="mt-2 text-xs text-red-500">
          Backend not installed
        </div>
      )}
    </button>
  )
}

// Benchmark results component
function BenchmarkResults({ benchmarks, loading }) {
  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
        <span className="ml-3 text-gray-600">Running benchmarks...</span>
      </div>
    )
  }
  
  if (!benchmarks || benchmarks.length === 0) {
    return null
  }
  
  // Sort by FPS
  const sorted = [...benchmarks].sort((a, b) => (b.fps || 0) - (a.fps || 0))
  
  return (
    <div className="space-y-2">
      <h4 className="font-medium text-gray-700">Performance Comparison</h4>
      <div className="grid gap-2">
        {sorted.map(b => (
          <div key={b.model_id} className="flex items-center gap-3 p-2 bg-gray-50 rounded">
            <span className="text-sm font-medium w-32 truncate">{b.model_id}</span>
            <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
              <div 
                className="h-full bg-primary-500 rounded-full"
                style={{ width: `${Math.min((b.fps / 60) * 100, 100)}%` }}
              />
            </div>
            <span className="text-sm font-mono w-16 text-right">
              {b.fps?.toFixed(0)} FPS
            </span>
            <span className="text-xs text-gray-500 w-20 text-right">
              {b.avg_ms?.toFixed(1)}ms
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

// Main PoseModelSettings component
export default function PoseModelSettings({ isOpen, onClose, onSave }) {
  const [models, setModels] = useState([])
  const [backends, setBackends] = useState([])
  const [preferences, setPreferences] = useState({
    live_model: 'mediapipe-medium',
    upload_model: 'rtmpose-large',
    fallback_model: 'mediapipe-medium',
    auto_select: true,
    prefer_gpu: false,
    prefer_accuracy: false,
  })
  const [stats, setStats] = useState(null)
  const [benchmarks, setBenchmarks] = useState([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [benchmarking, setBenchmarking] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('models')
  
  // Fetch initial data
  useEffect(() => {
    if (!isOpen) return
    
    const fetchData = async () => {
      setLoading(true)
      setError(null)
      
      try {
        const [modelsRes, backendsRes, prefsRes, statsRes] = await Promise.all([
          fetch('/api/pose/models'),
          fetch('/api/pose/backends'),
          fetch('/api/pose/preferences'),
          fetch('/api/pose/stats'),
        ])
        
        if (!modelsRes.ok || !backendsRes.ok || !prefsRes.ok) {
          throw new Error('Failed to fetch pose model settings')
        }
        
        const [modelsData, backendsData, prefsData, statsData] = await Promise.all([
          modelsRes.json(),
          backendsRes.json(),
          prefsRes.json(),
          statsRes.json(),
        ])
        
        setModels(modelsData)
        setBackends(backendsData.backends || [])
        setPreferences(prefsData)
        setStats(statsData)
      } catch (err) {
        console.error('Failed to load pose model settings:', err)
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    
    fetchData()
  }, [isOpen])
  
  // Save preferences
  const handleSave = async () => {
    setSaving(true)
    setError(null)
    
    try {
      const res = await fetch('/api/pose/preferences', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(preferences),
      })
      
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Failed to save preferences')
      }
      
      const saved = await res.json()
      setPreferences(saved)
      
      if (onSave) onSave(saved)
      onClose()
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }
  
  // Run benchmarks
  const handleBenchmark = async () => {
    setBenchmarking(true)
    setBenchmarks([])
    
    try {
      const res = await fetch('/api/pose/benchmark/all', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_frames: 30 }),
      })
      
      if (res.ok) {
        const data = await res.json()
        setBenchmarks(data.benchmarks || [])
      }
    } catch (err) {
      console.error('Benchmark failed:', err)
    } finally {
      setBenchmarking(false)
    }
  }
  
  // Update preference value
  const updatePreference = (key, value) => {
    setPreferences(prev => ({ ...prev, [key]: value }))
  }
  
  // Filter models by mode
  const liveModels = models.filter(m => 
    m.is_available && (m.recommended_for?.includes('live') || m.recommended_for?.includes('speed'))
  )
  const uploadModels = models.filter(m => 
    m.is_available && (m.recommended_for?.includes('upload') || m.recommended_for?.includes('accuracy'))
  )
  
  if (!isOpen) return null
  
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-3xl max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="bg-gray-50 px-6 py-4 border-b flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-gray-800">Pose Model Settings</h2>
            <p className="text-sm text-gray-500">Choose computer vision models for pose estimation</p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        {/* Tabs */}
        <div className="border-b px-6">
          <nav className="flex gap-6">
            {['models', 'backends', 'benchmark'].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`py-3 border-b-2 text-sm font-medium capitalize transition-colors
                  ${activeTab === tab 
                    ? 'border-primary-500 text-primary-600' 
                    : 'border-transparent text-gray-500 hover:text-gray-700'}`}
              >
                {tab}
              </button>
            ))}
          </nav>
        </div>
        
        {/* Content */}
        <div className="p-6 overflow-y-auto" style={{ maxHeight: 'calc(90vh - 200px)' }}>
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <div className="w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : error ? (
            <div className="bg-red-50 text-red-700 p-4 rounded-lg">
              <p className="font-medium">Error loading settings</p>
              <p className="text-sm">{error}</p>
            </div>
          ) : (
            <>
              {/* Models Tab */}
              {activeTab === 'models' && (
                <div className="space-y-6">
                  {/* Live Analysis Model */}
                  <div>
                    <h3 className="font-medium text-gray-800 mb-3 flex items-center gap-2">
                      <span className="text-lg">🎥</span>
                      Live Analysis Model
                      <span className="text-xs text-gray-500 font-normal">
                        (optimized for real-time)
                      </span>
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                      {models.filter(m => m.is_available).map(model => (
                        <ModelCard
                          key={model.model_id}
                          model={model}
                          isSelected={preferences.live_model === model.model_id}
                          onSelect={(id) => updatePreference('live_model', id)}
                          disabled={saving}
                        />
                      ))}
                    </div>
                  </div>
                  
                  {/* Upload Analysis Model */}
                  <div>
                    <h3 className="font-medium text-gray-800 mb-3 flex items-center gap-2">
                      <span className="text-lg">📤</span>
                      Upload Analysis Model
                      <span className="text-xs text-gray-500 font-normal">
                        (optimized for accuracy)
                      </span>
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                      {models.filter(m => m.is_available).map(model => (
                        <ModelCard
                          key={model.model_id}
                          model={model}
                          isSelected={preferences.upload_model === model.model_id}
                          onSelect={(id) => updatePreference('upload_model', id)}
                          disabled={saving}
                        />
                      ))}
                    </div>
                  </div>
                  
                  {/* Advanced Settings */}
                  <div className="border-t pt-4">
                    <h3 className="font-medium text-gray-800 mb-3">Advanced Settings</h3>
                    
                    <div className="space-y-3">
                      <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div>
                          <p className="font-medium text-gray-700">Auto-select Model</p>
                          <p className="text-xs text-gray-500">
                            Automatically choose best model based on context
                          </p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input
                            type="checkbox"
                            checked={preferences.auto_select}
                            onChange={(e) => updatePreference('auto_select', e.target.checked)}
                            className="sr-only peer"
                          />
                          <div className="w-11 h-6 bg-gray-200 peer-focus:ring-4 peer-focus:ring-primary-300 
                                          rounded-full peer peer-checked:after:translate-x-full 
                                          peer-checked:after:border-white after:content-[''] after:absolute 
                                          after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 
                                          after:border after:rounded-full after:h-5 after:w-5 after:transition-all 
                                          peer-checked:bg-primary-600" />
                        </label>
                      </div>
                      
                      <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div>
                          <p className="font-medium text-gray-700">Use GPU (if available)</p>
                          <p className="text-xs text-gray-500">
                            Enable GPU acceleration for faster processing
                          </p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input
                            type="checkbox"
                            checked={preferences.prefer_gpu}
                            onChange={(e) => updatePreference('prefer_gpu', e.target.checked)}
                            className="sr-only peer"
                          />
                          <div className="w-11 h-6 bg-gray-200 peer-focus:ring-4 peer-focus:ring-primary-300 
                                          rounded-full peer peer-checked:after:translate-x-full 
                                          peer-checked:after:border-white after:content-[''] after:absolute 
                                          after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 
                                          after:border after:rounded-full after:h-5 after:w-5 after:transition-all 
                                          peer-checked:bg-primary-600" />
                        </label>
                      </div>
                      
                      <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div>
                          <p className="font-medium text-gray-700">Prefer Accuracy</p>
                          <p className="text-xs text-gray-500">
                            Choose accuracy over speed when auto-selecting
                          </p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input
                            type="checkbox"
                            checked={preferences.prefer_accuracy}
                            onChange={(e) => updatePreference('prefer_accuracy', e.target.checked)}
                            className="sr-only peer"
                          />
                          <div className="w-11 h-6 bg-gray-200 peer-focus:ring-4 peer-focus:ring-primary-300 
                                          rounded-full peer peer-checked:after:translate-x-full 
                                          peer-checked:after:border-white after:content-[''] after:absolute 
                                          after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 
                                          after:border after:rounded-full after:h-5 after:w-5 after:transition-all 
                                          peer-checked:bg-primary-600" />
                        </label>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              
              {/* Backends Tab */}
              {activeTab === 'backends' && (
                <div className="space-y-4">
                  <p className="text-sm text-gray-600 mb-4">
                    Installed computer vision backends for pose estimation.
                  </p>
                  
                  {backends.map(b => (
                    <div 
                      key={b.backend}
                      className={`p-4 rounded-lg border-2 ${
                        b.available 
                          ? 'border-green-200 bg-green-50' 
                          : 'border-gray-200 bg-gray-50'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <BackendBadge backend={b.backend} />
                          <span className="font-medium text-gray-800 capitalize">
                            {b.backend}
                          </span>
                          {b.available && (
                            <span className="text-xs text-green-600">
                              v{b.version}
                            </span>
                          )}
                        </div>
                        <span className={`flex items-center gap-1 text-sm ${
                          b.available ? 'text-green-600' : 'text-gray-500'
                        }`}>
                          <span className={`w-2 h-2 rounded-full ${
                            b.available ? 'bg-green-500' : 'bg-gray-400'
                          }`} />
                          {b.available ? 'Available' : 'Not Installed'}
                        </span>
                      </div>
                      
                      <p className="text-sm text-gray-600 mt-2">{b.description}</p>
                      
                      {b.providers && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {b.providers.map(p => (
                            <span 
                              key={p}
                              className="text-xs px-2 py-0.5 bg-white rounded border"
                            >
                              {p.replace('ExecutionProvider', '')}
                            </span>
                          ))}
                        </div>
                      )}
                      
                      {b.cuda_available && (
                        <div className="mt-2 text-xs text-green-600">
                          ✓ CUDA acceleration available
                        </div>
                      )}
                    </div>
                  ))}
                  
                  <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                    <h4 className="text-sm font-medium text-blue-800 mb-2">
                      Installing Additional Backends
                    </h4>
                    <p className="text-xs text-blue-700">
                      Add new pose estimation backends via pip:
                    </p>
                    <pre className="mt-2 text-xs bg-blue-100 p-2 rounded overflow-x-auto">
{`# MediaPipe (default)
pip install mediapipe

# RTMPose (via ONNX Runtime)
pip install onnxruntime

# YOLO Pose
pip install ultralytics`}
                    </pre>
                  </div>
                </div>
              )}
              
              {/* Benchmark Tab */}
              {activeTab === 'benchmark' && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-medium text-gray-800">Performance Benchmarks</h3>
                      <p className="text-sm text-gray-500">
                        Test all models to find the fastest for your hardware
                      </p>
                    </div>
                    <button
                      onClick={handleBenchmark}
                      disabled={benchmarking}
                      className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700
                                 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                      {benchmarking ? (
                        <>
                          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                          Running...
                        </>
                      ) : (
                        <>
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                          </svg>
                          Run Benchmarks
                        </>
                      )}
                    </button>
                  </div>
                  
                  <BenchmarkResults benchmarks={benchmarks} loading={benchmarking} />
                  
                  {stats && (
                    <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-3">
                      <div className="p-3 bg-gray-50 rounded-lg text-center">
                        <p className="text-xs text-gray-500">Total Models</p>
                        <p className="text-xl font-semibold text-gray-800">
                          {stats.total_models}
                        </p>
                      </div>
                      <div className="p-3 bg-gray-50 rounded-lg text-center">
                        <p className="text-xs text-gray-500">Available</p>
                        <p className="text-xl font-semibold text-green-600">
                          {stats.available_models}
                        </p>
                      </div>
                      <div className="p-3 bg-gray-50 rounded-lg text-center">
                        <p className="text-xs text-gray-500">Loaded</p>
                        <p className="text-xl font-semibold text-primary-600">
                          {stats.loaded_models}
                        </p>
                      </div>
                      <div className="p-3 bg-gray-50 rounded-lg text-center">
                        <p className="text-xs text-gray-500">Backends</p>
                        <p className="text-xl font-semibold text-gray-800">
                          {stats.backends?.length || 0}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
        
        {/* Footer */}
        <div className="border-t px-6 py-4 bg-gray-50 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={saving || loading}
            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 
                       transition-colors disabled:opacity-50 disabled:cursor-not-allowed
                       flex items-center gap-2"
          >
            {saving && (
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            )}
            Save Settings
          </button>
        </div>
      </div>
    </div>
  )
}
