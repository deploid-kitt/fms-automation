import React, { useState, useEffect, useCallback } from 'react'

// Model tier badges
const TierBadge = ({ tier }) => {
  const config = {
    fast: { bg: 'bg-green-100', text: 'text-green-700', label: 'Fast' },
    standard: { bg: 'bg-blue-100', text: 'text-blue-700', label: 'Standard' },
    premium: { bg: 'bg-purple-100', text: 'text-purple-700', label: 'Premium' },
  }
  const c = config[tier] || config.standard
  
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${c.bg} ${c.text}`}>
      {c.label}
    </span>
  )
}

// Provider icon
const ProviderIcon = ({ provider }) => {
  const icons = {
    openai: '🟢',
    anthropic: '🔵',
    google: '🔴',
    ollama: '🟣',
  }
  return <span className="text-lg">{icons[provider] || '⚪'}</span>
}

// Model selector dropdown
function ModelSelector({ 
  label, 
  description, 
  value, 
  onChange, 
  models, 
  capability,
  disabled 
}) {
  // Filter models by capability
  const filteredModels = models.filter(m => 
    !capability || m.capabilities.includes(capability)
  )
  
  return (
    <div className="mb-4">
      <label className="block text-sm font-medium text-gray-700 mb-1">
        {label}
      </label>
      {description && (
        <p className="text-xs text-gray-500 mb-2">{description}</p>
      )}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm
                   focus:ring-2 focus:ring-primary-500 focus:border-primary-500
                   disabled:bg-gray-100 disabled:cursor-not-allowed"
      >
        {filteredModels.map(model => (
          <option key={model.id} value={model.id.split('/')[1]}>
            {model.display_name} ({model.provider}) - {model.tier}
          </option>
        ))}
      </select>
    </div>
  )
}

// Provider status indicator
function ProviderStatus({ provider, healthy }) {
  return (
    <div className="flex items-center gap-2 px-3 py-2 bg-gray-50 rounded-lg">
      <ProviderIcon provider={provider} />
      <span className="text-sm font-medium capitalize">{provider}</span>
      <span className={`ml-auto w-2 h-2 rounded-full ${healthy ? 'bg-green-500' : 'bg-red-500'}`} />
      <span className="text-xs text-gray-500">{healthy ? 'Connected' : 'Unavailable'}</span>
    </div>
  )
}

// Main ModelSettings component
export default function ModelSettings({ isOpen, onClose, onSave }) {
  const [models, setModels] = useState([])
  const [providers, setProviders] = useState([])
  const [preferences, setPreferences] = useState({
    realtime_feedback: 'gpt-4o-mini',
    movement_analysis: 'claude-sonnet-4-20250514',
    report_generation: 'claude-sonnet-4-20250514',
    exercise_classification: 'gpt-4o-mini',
    coaching_cues: 'gpt-4o-mini',
    fallback: 'gpt-3.5-turbo',
    enable_llm: true,
    enable_caching: true,
    cache_ttl_seconds: 3600,
  })
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('models')
  
  // Fetch initial data
  useEffect(() => {
    if (!isOpen) return
    
    const fetchData = async () => {
      setLoading(true)
      setError(null)
      
      try {
        const [modelsRes, providersRes, prefsRes, statsRes] = await Promise.all([
          fetch('/api/llm/models'),
          fetch('/api/llm/providers'),
          fetch('/api/llm/preferences'),
          fetch('/api/llm/stats'),
        ])
        
        if (!modelsRes.ok || !providersRes.ok || !prefsRes.ok) {
          throw new Error('Failed to fetch LLM settings')
        }
        
        const [modelsData, providersData, prefsData, statsData] = await Promise.all([
          modelsRes.json(),
          providersRes.json(),
          prefsRes.json(),
          statsRes.json(),
        ])
        
        setModels(modelsData)
        setProviders(providersData)
        setPreferences(prefsData)
        setStats(statsData)
      } catch (err) {
        console.error('Failed to load LLM settings:', err)
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
      const res = await fetch('/api/llm/preferences', {
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
  
  // Clear cache
  const handleClearCache = async () => {
    try {
      await fetch('/api/llm/cache/clear', { method: 'POST' })
      // Refresh stats
      const statsRes = await fetch('/api/llm/stats')
      setStats(await statsRes.json())
    } catch (err) {
      console.error('Failed to clear cache:', err)
    }
  }
  
  // Update preference value
  const updatePreference = (key, value) => {
    setPreferences(prev => ({ ...prev, [key]: value }))
  }
  
  if (!isOpen) return null
  
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-2xl max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="bg-gray-50 px-6 py-4 border-b flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-gray-800">AI Model Settings</h2>
            <p className="text-sm text-gray-500">Configure models for different analysis tasks</p>
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
            {['models', 'providers', 'stats'].map(tab => (
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
                  {/* Enable LLM toggle */}
                  <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                    <div>
                      <p className="font-medium text-gray-800">Enable AI Enhancement</p>
                      <p className="text-sm text-gray-500">Use LLMs for coaching cues and reports</p>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={preferences.enable_llm}
                        onChange={(e) => updatePreference('enable_llm', e.target.checked)}
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
                  
                  {preferences.enable_llm && (
                    <>
                      <div className="grid grid-cols-1 gap-4">
                        <ModelSelector
                          label="Real-time Coaching"
                          description="Fast model for live feedback during exercises"
                          value={preferences.realtime_feedback}
                          onChange={(v) => updatePreference('realtime_feedback', v)}
                          models={models}
                          capability="realtime_feedback"
                        />
                        
                        <ModelSelector
                          label="Movement Analysis"
                          description="Detailed analysis of movement patterns"
                          value={preferences.movement_analysis}
                          onChange={(v) => updatePreference('movement_analysis', v)}
                          models={models}
                          capability="movement_analysis"
                        />
                        
                        <ModelSelector
                          label="Report Generation"
                          description="Comprehensive assessment reports"
                          value={preferences.report_generation}
                          onChange={(v) => updatePreference('report_generation', v)}
                          models={models}
                          capability="report_generation"
                        />
                        
                        <ModelSelector
                          label="Fallback Model"
                          description="Used when primary model is unavailable"
                          value={preferences.fallback}
                          onChange={(v) => updatePreference('fallback', v)}
                          models={models}
                        />
                      </div>
                      
                      {/* Caching settings */}
                      <div className="border-t pt-4 mt-4">
                        <h3 className="font-medium text-gray-800 mb-3">Performance</h3>
                        
                        <div className="flex items-center justify-between mb-3">
                          <div>
                            <p className="text-sm font-medium text-gray-700">Response Caching</p>
                            <p className="text-xs text-gray-500">Cache similar requests to reduce latency</p>
                          </div>
                          <label className="relative inline-flex items-center cursor-pointer">
                            <input
                              type="checkbox"
                              checked={preferences.enable_caching}
                              onChange={(e) => updatePreference('enable_caching', e.target.checked)}
                              className="sr-only peer"
                            />
                            <div className="w-9 h-5 bg-gray-200 peer-focus:ring-2 peer-focus:ring-primary-300 
                                            rounded-full peer peer-checked:after:translate-x-full 
                                            peer-checked:after:border-white after:content-[''] after:absolute 
                                            after:top-[2px] after:left-[2px] after:bg-white after:border 
                                            after:rounded-full after:h-4 after:w-4 after:transition-all 
                                            peer-checked:bg-primary-600" />
                          </label>
                        </div>
                        
                        {preferences.enable_caching && (
                          <div className="flex items-center gap-3">
                            <label className="text-sm text-gray-700">Cache TTL:</label>
                            <select
                              value={preferences.cache_ttl_seconds}
                              onChange={(e) => updatePreference('cache_ttl_seconds', parseInt(e.target.value))}
                              className="px-2 py-1 border border-gray-300 rounded text-sm"
                            >
                              <option value={600}>10 minutes</option>
                              <option value={1800}>30 minutes</option>
                              <option value={3600}>1 hour</option>
                              <option value={7200}>2 hours</option>
                            </select>
                          </div>
                        )}
                      </div>
                    </>
                  )}
                </div>
              )}
              
              {/* Providers Tab */}
              {activeTab === 'providers' && (
                <div className="space-y-4">
                  <p className="text-sm text-gray-600 mb-4">
                    Connected AI providers. Configure API keys in your environment variables.
                  </p>
                  
                  {providers.map(p => (
                    <ProviderStatus 
                      key={p.provider} 
                      provider={p.provider} 
                      healthy={p.healthy} 
                    />
                  ))}
                  
                  <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                    <h4 className="text-sm font-medium text-blue-800 mb-2">Adding Providers</h4>
                    <p className="text-xs text-blue-700">
                      Set API keys as environment variables:
                    </p>
                    <pre className="mt-2 text-xs bg-blue-100 p-2 rounded overflow-x-auto">
{`OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...`}
                    </pre>
                    <p className="text-xs text-blue-700 mt-2">
                      For local models, ensure Ollama is running at the configured URL.
                    </p>
                  </div>
                </div>
              )}
              
              {/* Stats Tab */}
              {activeTab === 'stats' && stats && (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-50 rounded-lg p-4">
                      <p className="text-sm text-gray-500">Total Requests</p>
                      <p className="text-2xl font-semibold text-gray-800">{stats.request_count}</p>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-4">
                      <p className="text-sm text-gray-500">Cache Hits</p>
                      <p className="text-2xl font-semibold text-gray-800">{stats.cache_hits}</p>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-4">
                      <p className="text-sm text-gray-500">Cache Hit Rate</p>
                      <p className="text-2xl font-semibold text-gray-800">
                        {(stats.cache_rate * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-4">
                      <p className="text-sm text-gray-500">Avg Latency</p>
                      <p className="text-2xl font-semibold text-gray-800">
                        {stats.avg_latency_ms.toFixed(0)}ms
                      </p>
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-sm text-gray-500 mb-1">Cache Size</p>
                    <p className="text-lg font-medium text-gray-800">{stats.cache_size} entries</p>
                    <button
                      onClick={handleClearCache}
                      className="mt-2 text-sm text-red-600 hover:text-red-700"
                    >
                      Clear Cache
                    </button>
                  </div>
                  
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-sm text-gray-500 mb-2">Active Providers</p>
                    <div className="flex flex-wrap gap-2">
                      {stats.available_providers.map(p => (
                        <span key={p} className="px-2 py-1 bg-green-100 text-green-700 text-xs rounded-full capitalize">
                          {p}
                        </span>
                      ))}
                    </div>
                  </div>
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
