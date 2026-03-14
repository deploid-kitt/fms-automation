import React, { useState, useEffect } from 'react'

/**
 * Quick pose model selection widget for use during live analysis.
 * Allows switching models without opening full settings.
 */
export default function PoseModelQuickSelect({ 
  currentModelId,
  onModelChange,
  compact = false,
  disabled = false,
}) {
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)
  const [switching, setSwitching] = useState(false)
  
  // Fetch available models
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await fetch('/api/pose/models')
        if (res.ok) {
          const data = await res.json()
          setModels(data.filter(m => m.is_available))
        }
      } catch (err) {
        console.error('Failed to fetch pose models:', err)
      } finally {
        setLoading(false)
      }
    }
    
    fetchModels()
  }, [])
  
  const handleChange = async (modelId) => {
    if (modelId === currentModelId || disabled || switching) return
    
    setSwitching(true)
    try {
      if (onModelChange) {
        await onModelChange(modelId)
      }
    } finally {
      setSwitching(false)
    }
  }
  
  const getCurrentModelInfo = () => {
    return models.find(m => m.model_id === currentModelId)
  }
  
  if (loading) {
    return (
      <div className={`inline-flex items-center gap-2 ${compact ? 'text-xs' : 'text-sm'}`}>
        <div className="w-3 h-3 border-2 border-gray-300 border-t-primary-500 rounded-full animate-spin" />
      </div>
    )
  }
  
  const currentModel = getCurrentModelInfo()
  
  // Compact inline selector
  if (compact) {
    return (
      <div className="inline-flex items-center gap-2">
        <span className="text-xs text-gray-400">Pose:</span>
        <select
          value={currentModelId || ''}
          onChange={(e) => handleChange(e.target.value)}
          disabled={disabled || switching}
          className="text-xs px-2 py-0.5 bg-black/50 border border-gray-600 rounded text-white
                     focus:ring-1 focus:ring-primary-500 disabled:opacity-50"
        >
          {models.map(m => (
            <option key={m.model_id} value={m.model_id}>
              {m.name} ({m.model_size})
            </option>
          ))}
        </select>
        {switching && (
          <div className="w-3 h-3 border-2 border-gray-300 border-t-primary-500 rounded-full animate-spin" />
        )}
      </div>
    )
  }
  
  // Full selector with model info
  return (
    <div className="bg-white/90 backdrop-blur rounded-lg p-3 shadow-lg">
      <div className="flex items-center gap-3">
        <div className="flex-1 min-w-0">
          <label className="text-xs text-gray-500">Pose Model</label>
          <select
            value={currentModelId || ''}
            onChange={(e) => handleChange(e.target.value)}
            disabled={disabled || switching}
            className="w-full mt-1 px-2 py-1.5 border border-gray-300 rounded text-sm
                       focus:ring-2 focus:ring-primary-500 disabled:opacity-50"
          >
            {models.map(m => (
              <option key={m.model_id} value={m.model_id}>
                {m.name} - {m.model_size}
                {m.recommended_for?.includes('live') ? ' (🎥)' : ''}
                {m.recommended_for?.includes('accuracy') ? ' (🎯)' : ''}
              </option>
            ))}
          </select>
        </div>
        
        {switching && (
          <div className="w-6 h-6 border-2 border-primary-300 border-t-primary-600 rounded-full animate-spin" />
        )}
      </div>
      
      {currentModel && (
        <div className="mt-2 flex flex-wrap gap-1">
          <span className={`text-xs px-1.5 py-0.5 rounded-full ${
            currentModel.backend === 'mediapipe' ? 'bg-green-100 text-green-700' :
            currentModel.backend === 'rtmpose' ? 'bg-blue-100 text-blue-700' :
            'bg-purple-100 text-purple-700'
          }`}>
            {currentModel.backend}
          </span>
          {currentModel.recommended_for?.map(tag => (
            <span key={tag} className="text-xs px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded-full">
              {tag}
            </span>
          ))}
          {currentModel.estimated_fps > 0 && (
            <span className="text-xs text-gray-500">
              ~{currentModel.estimated_fps.toFixed(0)} FPS
            </span>
          )}
        </div>
      )}
    </div>
  )
}

/**
 * Inline model indicator showing current model with click to expand.
 */
export function PoseModelIndicator({ 
  currentModelId, 
  onClick,
  className = '' 
}) {
  const [model, setModel] = useState(null)
  
  useEffect(() => {
    if (!currentModelId) return
    
    fetch('/api/pose/models')
      .then(res => res.json())
      .then(data => {
        const m = data.find(m => m.model_id === currentModelId)
        setModel(m)
      })
      .catch(() => {})
  }, [currentModelId])
  
  if (!model) {
    return (
      <span className={`text-xs text-gray-400 ${className}`}>
        {currentModelId || 'No model'}
      </span>
    )
  }
  
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center gap-1.5 text-xs px-2 py-1 rounded-full
                 bg-black/50 text-white hover:bg-black/70 transition-colors ${className}`}
    >
      <span className={`w-2 h-2 rounded-full ${
        model.backend === 'mediapipe' ? 'bg-green-400' :
        model.backend === 'rtmpose' ? 'bg-blue-400' :
        'bg-purple-400'
      }`} />
      <span>{model.name}</span>
      <span className="text-gray-400">({model.model_size})</span>
      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
      </svg>
    </button>
  )
}
