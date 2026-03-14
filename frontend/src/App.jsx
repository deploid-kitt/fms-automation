import React, { useState, useCallback, useEffect } from 'react'
import LiveAnalysis from './components/LiveAnalysis'

const API_BASE = '/api'

// Analysis mode types
const MODE = {
  SELECT: 'select',
  UPLOAD: 'upload',
  LIVE: 'live'
}

// Score badge component
function ScoreBadge({ score }) {
  if (score === null || score === undefined) {
    return <span className="score-badge score-0">-</span>
  }
  return <span className={`score-badge score-${score}`}>{score}</span>
}

// Progress bar component
function ProgressBar({ progress, status }) {
  return (
    <div className="w-full">
      <div className="flex justify-between mb-1">
        <span className="text-sm font-medium text-primary-700">{status}</span>
        <span className="text-sm font-medium text-primary-700">{Math.round(progress)}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-3">
        <div 
          className="bg-primary-600 h-3 rounded-full transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  )
}

// Upload zone component
function UploadZone({ onUpload, disabled }) {
  const [dragging, setDragging] = useState(false)

  const handleDrag = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const handleDragIn = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragging(true)
  }, [])

  const handleDragOut = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragging(false)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragging(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUpload(e.dataTransfer.files[0])
    }
  }, [onUpload])

  const handleClick = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'video/*'
    input.onchange = (e) => {
      if (e.target.files && e.target.files[0]) {
        onUpload(e.target.files[0])
      }
    }
    input.click()
  }

  return (
    <div
      className={`upload-zone ${dragging ? 'dragging' : ''} ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      onDragEnter={handleDragIn}
      onDragLeave={handleDragOut}
      onDragOver={handleDrag}
      onDrop={disabled ? undefined : handleDrop}
      onClick={disabled ? undefined : handleClick}
    >
      <div className="flex flex-col items-center gap-4">
        <svg className="w-16 h-16 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} 
            d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
        </svg>
        <div>
          <p className="text-lg font-medium text-gray-700">
            {dragging ? 'Drop your video here' : 'Drag & drop your FMS video'}
          </p>
          <p className="text-sm text-gray-500 mt-1">or click to browse</p>
        </div>
        <p className="text-xs text-gray-400">Supports MP4, MOV, AVI, WebM (max 500MB)</p>
      </div>
    </div>
  )
}

// Report component
function Report({ report, onDownloadPDF, onNewAssessment }) {
  const getQualityColor = (score) => {
    if (score >= 18) return 'text-green-600'
    if (score >= 14) return 'text-yellow-600'
    if (score >= 10) return 'text-orange-600'
    return 'text-red-600'
  }

  const tests = [
    { name: 'Deep Squat', score: report.deep_squat?.score },
    { name: 'Hurdle Step', score: report.hurdle_step },
    { name: 'Inline Lunge', score: report.inline_lunge },
    { name: 'Shoulder Mobility', score: report.shoulder_mobility },
    { name: 'Active Straight Leg Raise', score: report.active_straight_leg_raise },
    { name: 'Trunk Stability Push-up', score: report.trunk_stability_pushup?.score },
    { name: 'Rotary Stability', score: report.rotary_stability },
  ]

  return (
    <div className="bg-white rounded-2xl shadow-lg p-8">
      {/* Header */}
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-gray-800">FMS Assessment Report</h2>
        <p className="text-gray-500 mt-1">
          {new Date(report.created_at).toLocaleDateString('en-US', {
            year: 'numeric', month: 'long', day: 'numeric'
          })}
        </p>
      </div>

      {/* Total Score */}
      <div className="bg-gray-50 rounded-xl p-6 mb-8 text-center">
        <p className="text-gray-600 mb-2">Total Score</p>
        <p className={`text-5xl font-bold ${getQualityColor(report.total_score)}`}>
          {report.total_score}
          <span className="text-2xl text-gray-400">/21</span>
        </p>
      </div>

      {/* Score Breakdown */}
      <div className="mb-8">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Score Breakdown</h3>
        <div className="space-y-3">
          {tests.map((test) => (
            <div key={test.name} className="flex items-center justify-between py-2 border-b border-gray-100">
              <span className="text-gray-700">{test.name}</span>
              <ScoreBadge score={test.score} />
            </div>
          ))}
        </div>
      </div>

      {/* Summary */}
      {report.summary && (
        <div className="mb-8">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Summary</h3>
          <p className="text-gray-600">{report.summary}</p>
        </div>
      )}

      {/* Recommendations */}
      {report.recommendations && report.recommendations.length > 0 && (
        <div className="mb-8">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Recommendations</h3>
          <ul className="list-disc list-inside space-y-1 text-gray-600">
            {report.recommendations.map((rec, idx) => (
              <li key={idx}>{rec}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-4">
        <button
          onClick={onDownloadPDF}
          className="flex-1 bg-primary-600 text-white py-3 px-6 rounded-lg font-medium
                     hover:bg-primary-700 transition-colors flex items-center justify-center gap-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
              d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          Download PDF
        </button>
        <button
          onClick={onNewAssessment}
          className="flex-1 border border-gray-300 text-gray-700 py-3 px-6 rounded-lg font-medium
                     hover:bg-gray-50 transition-colors"
        >
          New Assessment
        </button>
      </div>
    </div>
  )
}

// Mode selector component
function ModeSelector({ onSelectMode }) {
  return (
    <div className="bg-white rounded-2xl shadow-lg p-8">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">FMS Analysis</h2>
        <p className="text-gray-600">
          Choose how you want to perform your Functional Movement Screen assessment
        </p>
      </div>
      
      <div className="grid md:grid-cols-2 gap-6">
        {/* Live Analysis Option */}
        <button
          onClick={() => onSelectMode(MODE.LIVE)}
          className="group relative bg-gradient-to-br from-primary-50 to-blue-100 rounded-xl p-6 text-left
                     border-2 border-transparent hover:border-primary-500 transition-all duration-300
                     hover:shadow-lg"
        >
          <div className="absolute top-4 right-4 bg-green-500 text-white text-xs font-medium px-2 py-1 rounded-full">
            NEW
          </div>
          
          <div className="w-16 h-16 bg-primary-600 rounded-xl flex items-center justify-center mb-4
                        group-hover:scale-110 transition-transform">
            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          </div>
          
          <h3 className="text-xl font-semibold text-gray-800 mb-2">Live Analysis</h3>
          <p className="text-gray-600 text-sm mb-4">
            Real-time coaching with your webcam. Get instant feedback as you perform each movement.
          </p>
          
          <ul className="space-y-2 text-sm text-gray-600">
            <li className="flex items-center gap-2">
              <svg className="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              Real-time pose tracking
            </li>
            <li className="flex items-center gap-2">
              <svg className="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              Instant coaching cues
            </li>
            <li className="flex items-center gap-2">
              <svg className="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              Audio feedback
            </li>
            <li className="flex items-center gap-2">
              <svg className="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              Visual skeleton overlay
            </li>
          </ul>
        </button>
        
        {/* Video Upload Option */}
        <button
          onClick={() => onSelectMode(MODE.UPLOAD)}
          className="group bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-6 text-left
                     border-2 border-transparent hover:border-gray-400 transition-all duration-300
                     hover:shadow-lg"
        >
          <div className="w-16 h-16 bg-gray-600 rounded-xl flex items-center justify-center mb-4
                        group-hover:scale-110 transition-transform">
            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
          </div>
          
          <h3 className="text-xl font-semibold text-gray-800 mb-2">Upload Video</h3>
          <p className="text-gray-600 text-sm mb-4">
            Upload a pre-recorded video for comprehensive analysis and detailed PDF report.
          </p>
          
          <ul className="space-y-2 text-sm text-gray-600">
            <li className="flex items-center gap-2">
              <svg className="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              Full 7-test battery
            </li>
            <li className="flex items-center gap-2">
              <svg className="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              Detailed PDF report
            </li>
            <li className="flex items-center gap-2">
              <svg className="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              Comprehensive recommendations
            </li>
            <li className="flex items-center gap-2">
              <svg className="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              Auto exercise detection
            </li>
          </ul>
        </button>
      </div>
    </div>
  )
}

// Upload Mode Component
function UploadMode({ onBack }) {
  const [state, setState] = useState('idle') // idle, uploading, processing, complete, error
  const [jobId, setJobId] = useState(null)
  const [progress, setProgress] = useState(0)
  const [statusMessage, setStatusMessage] = useState('')
  const [report, setReport] = useState(null)
  const [error, setError] = useState(null)

  // Poll for status
  useEffect(() => {
    if (state !== 'processing' || !jobId) return

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/status/${jobId}`)
        const data = await res.json()

        setProgress(data.progress)
        setStatusMessage(data.message)

        if (data.status === 'completed') {
          clearInterval(interval)
          // Fetch report
          const reportRes = await fetch(`${API_BASE}/report/${jobId}`)
          const reportData = await reportRes.json()
          setReport(reportData)
          setState('complete')
        } else if (data.status === 'failed') {
          clearInterval(interval)
          setError(data.error || 'Processing failed')
          setState('error')
        }
      } catch (err) {
        console.error('Status poll error:', err)
      }
    }, 1000)

    return () => clearInterval(interval)
  }, [state, jobId])

  const handleUpload = async (file) => {
    setState('uploading')
    setError(null)
    setProgress(0)
    setStatusMessage('Uploading video...')

    try {
      const formData = new FormData()
      formData.append('file', file)

      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Upload failed')
      }

      const data = await res.json()
      setJobId(data.job_id)
      setState('processing')
      setStatusMessage('Processing started...')
    } catch (err) {
      setError(err.message)
      setState('error')
    }
  }

  const handleDownloadPDF = async () => {
    if (!jobId) return
    
    const res = await fetch(`${API_BASE}/report/${jobId}/pdf`)
    const blob = await res.blob()
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `fms_report_${jobId.slice(0, 8)}.pdf`
    a.click()
    window.URL.revokeObjectURL(url)
  }

  const handleNewAssessment = () => {
    setState('idle')
    setJobId(null)
    setProgress(0)
    setStatusMessage('')
    setReport(null)
    setError(null)
  }

  return (
    <>
      {state === 'idle' && (
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <div className="flex items-center justify-between mb-8">
            <div className="text-center flex-1">
              <h2 className="text-2xl font-bold text-gray-800 mb-2">Upload FMS Video</h2>
              <p className="text-gray-600">
                Upload a video of your patient performing the FMS battery for automated scoring
              </p>
            </div>
            <button
              onClick={onBack}
              className="text-gray-600 hover:text-gray-800 flex items-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              Back
            </button>
          </div>
          <UploadZone onUpload={handleUpload} disabled={false} />
          
          <div className="mt-8 grid grid-cols-3 gap-4 text-center">
            <div className="p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl mb-1">🎯</div>
              <p className="text-sm font-medium text-gray-700">7 FMS Tests</p>
              <p className="text-xs text-gray-500">Automatically detected</p>
            </div>
            <div className="p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl mb-1">🤖</div>
              <p className="text-sm font-medium text-gray-700">AI Scoring</p>
              <p className="text-xs text-gray-500">0-3 per test</p>
            </div>
            <div className="p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl mb-1">📄</div>
              <p className="text-sm font-medium text-gray-700">PDF Reports</p>
              <p className="text-xs text-gray-500">Download & share</p>
            </div>
          </div>
        </div>
      )}

      {(state === 'uploading' || state === 'processing') && (
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <div className="text-center mb-8">
            <div className="w-16 h-16 mx-auto mb-4 relative">
              <div className="absolute inset-0 border-4 border-primary-200 rounded-full" />
              <div className="absolute inset-0 border-4 border-primary-600 rounded-full border-t-transparent animate-spin" />
            </div>
            <h2 className="text-2xl font-bold text-gray-800 mb-2">Analyzing Video</h2>
            <p className="text-gray-600">Please wait while we process your FMS assessment</p>
          </div>
          <ProgressBar progress={progress} status={statusMessage} />
          
          <div className="mt-8 text-center text-sm text-gray-500">
            <p>This may take a few minutes depending on video length</p>
          </div>
        </div>
      )}

      {state === 'complete' && report && (
        <Report 
          report={report}
          onDownloadPDF={handleDownloadPDF}
          onNewAssessment={handleNewAssessment}
        />
      )}

      {state === 'error' && (
        <div className="bg-white rounded-2xl shadow-lg p-8 text-center">
          <div className="w-16 h-16 mx-auto mb-4 bg-red-100 rounded-full flex items-center justify-center">
            <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                d="M6 18L18 6M6 6l12 12" />
            </svg>
          </div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2">Processing Failed</h2>
          <p className="text-red-600 mb-6">{error}</p>
          <button
            onClick={handleNewAssessment}
            className="bg-primary-600 text-white py-3 px-6 rounded-lg font-medium
                       hover:bg-primary-700 transition-colors"
          >
            Try Again
          </button>
        </div>
      )}
    </>
  )
}

// Main App
export default function App() {
  const [mode, setMode] = useState(MODE.SELECT)

  const handleBack = useCallback(() => {
    setMode(MODE.SELECT)
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 to-blue-100">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center gap-3">
          <div className="w-10 h-10 bg-primary-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold">FMS</span>
          </div>
          <div>
            <h1 className="text-xl font-semibold text-gray-800">FMS Automation</h1>
            <p className="text-sm text-gray-500">Functional Movement Screen Analysis</p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 py-12">
        {mode === MODE.SELECT && (
          <ModeSelector onSelectMode={setMode} />
        )}
        
        {mode === MODE.UPLOAD && (
          <UploadMode onBack={handleBack} />
        )}
        
        {mode === MODE.LIVE && (
          <LiveAnalysis onBack={handleBack} />
        )}
      </main>

      {/* Footer */}
      <footer className="text-center py-8 text-sm text-gray-500">
        <p>FMS Automation v2.0 • Powered by MediaPipe • Live Analysis Enabled</p>
      </footer>
    </div>
  )
}
