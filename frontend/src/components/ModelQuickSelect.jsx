import React, { useState, useEffect } from 'react'

/**
 * Quick model selection widget for use in various parts of the app.
 * Shows current model and allows quick switching.
 */
export default function ModelQuickSelect({ 
  capability = 'report_generation',
  onChange,
  compact = false,
}) {
  const [models, setModels] = useState([])
  const [preferences, setPreferences] = useState(null)
  const [loading, setLoading] = useState(true)
  const [updating, setUpdating] = useState(false)
  
  // Fetch available models and current preferences
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [modelsRes, prefsRes] = await Promise.all([
          fetch(`/api/llm/models?capability=${capability}`),
          fetch('/api/llm/preferences'),
        ])
        
        if (modelsRes.ok && prefsRes.ok) {
          setModels(await modelsRes.json())
          setPreferences(await prefsRes.json())
        }
      } catch (err) {
        console.error('Failed to fetch model data:', err)
      } finally {
        setLoading(false)
      }
    }
    
    fetchData()
  }, [capability])
  
  const getCurrentModel = () => {
    if (!preferences) return null
    
    const capMap = {
      realtime_feedback: 'realtime_feedback',
      movement_analysis: 'movement_analysis',
      report_generation: 'report_generation',
      exercise_classification: 'exercise_classification',
      coaching_cues: 'coaching_cues',
    }
    
    const prefKey = capMap[capability] || 'fallback'
    return preferences[prefKey]
  }
  
  const handleChange = async (modelId) => {
    setUpdating(true)
    
    try {
      const capMap = {
        realtime_feedback: 'realtime_feedback',
        movement_analysis: 'movement_analysis',
        report_generation: 'report_generation',
        exercise_classification: 'exercise_classification',
        coaching_cues: 'coaching_cues',
      }
      
      const prefKey = capMap[capability] || 'fallback'
      
      const res = await fetch('/api/llm/preferences', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [prefKey]: modelId }),
      })
      
      if (res.ok) {
        const newPrefs = await res.json()
        setPreferences(newPrefs)
        if (onChange) onChange(modelId)
      }
    } catch (err) {
      console.error('Failed to update model:', err)
    } finally {
      setUpdating(false)
    }
  }
  
  if (loading) {
    return (
      <div className={`inline-flex items-center gap-2 ${compact ? 'text-xs' : 'text-sm'}`}>
        <div className="w-4 h-4 border-2 border-gray-300 border-t-primary-500 rounded-full animate-spin" />
        <span className="text-gray-500">Loading...</span>
      </div>
    )
  }
  
  if (!preferences || models.length === 0) {
    return null
  }
  
  const currentModel = getCurrentModel()
  const currentConfig = models.find(m => m.id.includes(currentModel))
  
  if (compact) {
    return (
      <div className="inline-flex items-center gap-2">
        <span className="text-xs text-gray-500">Model:</span>
        <select
          value={currentModel}
          onChange={(e) => handleChange(e.target.value)}
          disabled={updating}
          className="text-xs px-2 py-1 border border-gray-300 rounded bg-white 
                     focus:ring-1 focus:ring-primary-500 disabled:opacity-50"
        >
          {models.map(m => (
            <option key={m.id} value={m.id.split('/')[1]}>
              {m.display_name}
            </option>
          ))}
        </select>
        {updating && (
          <div className="w-3 h-3 border-2 border-gray-300 border-t-primary-500 rounded-full animate-spin" />
        )}
      </div>
    )
  }
  
  return (
    <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-700">AI Model</span>
          {currentConfig && (
            <span className={`px-1.5 py-0.5 text-xs rounded-full 
              ${currentConfig.tier === 'fast' ? 'bg-green-100 text-green-700' :
                currentConfig.tier === 'premium' ? 'bg-purple-100 text-purple-700' :
                'bg-blue-100 text-blue-700'}`}>
              {currentConfig.tier}
            </span>
          )}
        </div>
        <p className="text-xs text-gray-500">
          {capability === 'report_generation' ? 'For generating reports' :
           capability === 'realtime_feedback' ? 'For live coaching' :
           capability === 'movement_analysis' ? 'For detailed analysis' :
           'For AI features'}
        </p>
      </div>
      
      <select
        value={currentModel}
        onChange={(e) => handleChange(e.target.value)}
        disabled={updating}
        className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white
                   focus:ring-2 focus:ring-primary-500 disabled:opacity-50"
      >
        {models.map(m => (
          <option key={m.id} value={m.id.split('/')[1]}>
            {m.display_name} ({m.provider})
          </option>
        ))}
      </select>
      
      {updating && (
        <div className="w-5 h-5 border-2 border-gray-300 border-t-primary-500 rounded-full animate-spin" />
      )}
    </div>
  )
}
