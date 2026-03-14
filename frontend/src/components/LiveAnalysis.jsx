import React, { useRef, useEffect, useState, useCallback } from 'react'
import ModelSettings from './ModelSettings'

// WebSocket connection states
const WS_STATE = {
  DISCONNECTED: 'disconnected',
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  ERROR: 'error'
}

// Skeleton connections for drawing
const POSE_CONNECTIONS = [
  // Torso
  [11, 12], // shoulders
  [11, 23], [12, 24], // shoulder to hip
  [23, 24], // hips
  // Arms
  [11, 13], [13, 15], // left arm
  [12, 14], [14, 16], // right arm
  // Legs
  [23, 25], [25, 27], // left leg
  [24, 26], [26, 28], // right leg
  // Face
  [0, 11], [0, 12], // nose to shoulders
]

// Interpolate between two skeleton poses for smooth animation
function interpolateSkeleton(prevSkeleton, nextSkeleton, t) {
  if (!prevSkeleton || !nextSkeleton || prevSkeleton.length !== nextSkeleton.length) {
    return nextSkeleton || prevSkeleton || []
  }
  
  return nextSkeleton.map((point, idx) => {
    const prev = prevSkeleton[idx]
    if (!prev || prev.visibility < 0.3 || point.visibility < 0.3) {
      return point
    }
    
    return {
      ...point,
      x: prev.x + (point.x - prev.x) * t,
      y: prev.y + (point.y - prev.y) * t,
      z: prev.z + (point.z - prev.z) * t,
    }
  })
}

// Draw pose skeleton on canvas with optimizations
function drawSkeleton(ctx, skeleton, problemJoints, width, height) {
  if (!skeleton || skeleton.length === 0) return
  
  // Use willReadFrequently for performance
  ctx.clearRect(0, 0, width, height)
  
  // Draw connections with single path for performance
  ctx.lineWidth = 3
  ctx.strokeStyle = '#00ff00'
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  
  POSE_CONNECTIONS.forEach(([i, j]) => {
    const p1 = skeleton[i]
    const p2 = skeleton[j]
    if (p1 && p2 && p1.visibility > 0.5 && p2.visibility > 0.5) {
      ctx.beginPath()
      ctx.moveTo(p1.x * width, p1.y * height)
      ctx.lineTo(p2.x * width, p2.y * height)
      ctx.stroke()
    }
  })
  
  // Draw joints - batch normal joints first
  ctx.fillStyle = '#00ff00'
  skeleton.forEach((point, idx) => {
    if (point.visibility > 0.5 && !problemJoints.includes(idx)) {
      ctx.beginPath()
      ctx.arc(point.x * width, point.y * height, 5, 0, 2 * Math.PI)
      ctx.fill()
    }
  })
  
  // Draw problem joints with glow effect
  ctx.fillStyle = '#ff4444'
  ctx.shadowColor = '#ff4444'
  ctx.shadowBlur = 15
  
  skeleton.forEach((point, idx) => {
    if (point.visibility > 0.5 && problemJoints.includes(idx)) {
      ctx.beginPath()
      ctx.arc(point.x * width, point.y * height, 8, 0, 2 * Math.PI)
      ctx.fill()
    }
  })
  
  ctx.shadowBlur = 0
}

// Score indicator component
function ScoreIndicator({ score, confidence }) {
  const colors = {
    0: 'bg-red-500',
    1: 'bg-orange-500',
    2: 'bg-yellow-500',
    3: 'bg-green-500'
  }
  
  return (
    <div className="flex items-center gap-2">
      <div className={`w-16 h-16 rounded-full ${colors[score]} flex items-center justify-center text-white text-2xl font-bold shadow-lg`}>
        {score}
      </div>
      <div className="text-sm text-gray-600">
        <div className="font-medium">Score</div>
        <div className="text-xs">{Math.round(confidence * 100)}% confident</div>
      </div>
    </div>
  )
}

// Form quality indicator
function FormQualityIndicator({ quality }) {
  const config = {
    excellent: { color: 'text-green-600', bg: 'bg-green-100', label: 'Excellent' },
    good: { color: 'text-blue-600', bg: 'bg-blue-100', label: 'Good' },
    needs_work: { color: 'text-yellow-600', bg: 'bg-yellow-100', label: 'Needs Work' },
    poor: { color: 'text-red-600', bg: 'bg-red-100', label: 'Poor' }
  }
  
  const c = config[quality] || config.poor
  
  return (
    <span className={`px-3 py-1 rounded-full text-sm font-medium ${c.bg} ${c.color}`}>
      {c.label}
    </span>
  )
}

// Coaching cue component with LLM support
function CoachingCue({ primary, secondary, llmCue, llmModel }) {
  const displayCue = llmCue || primary
  const isLLMCue = !!llmCue
  
  return (
    <div className="bg-white/90 backdrop-blur rounded-lg p-4 shadow-lg">
      <div className="flex items-start gap-2">
        <p className="text-lg font-medium text-gray-800 flex-1">{displayCue}</p>
        {isLLMCue && (
          <span className="px-1.5 py-0.5 bg-purple-100 text-purple-700 text-xs rounded-full flex-shrink-0">
            AI
          </span>
        )}
      </div>
      {secondary && secondary.length > 0 && (
        <ul className="mt-2 space-y-1">
          {secondary.map((cue, idx) => (
            <li key={idx} className="text-sm text-gray-600 flex items-center gap-2">
              <span className="w-1.5 h-1.5 bg-yellow-500 rounded-full"></span>
              {cue}
            </li>
          ))}
        </ul>
      )}
      {isLLMCue && llmModel && (
        <p className="mt-2 text-xs text-gray-400">via {llmModel}</p>
      )}
    </div>
  )
}

// Joint angles display
function JointAngles({ angles }) {
  if (!angles || Object.keys(angles).length === 0) return null
  
  const formatLabel = (key) => {
    return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
  }
  
  return (
    <div className="bg-black/50 text-white px-3 py-2 rounded-lg text-sm">
      {Object.entries(angles).map(([key, value]) => (
        <div key={key} className="flex justify-between gap-4">
          <span>{formatLabel(key)}:</span>
          <span className="font-mono">{value}°</span>
        </div>
      ))}
    </div>
  )
}

// Performance monitor component
function PerformanceMonitor({ fps, latency, serverProcessMs, visible }) {
  if (!visible) return null
  
  return (
    <div className="bg-black/70 text-white text-xs px-2 py-1 rounded font-mono space-y-0.5">
      <div className="flex justify-between gap-3">
        <span>Send FPS:</span>
        <span className={fps < 10 ? 'text-red-400' : fps < 15 ? 'text-yellow-400' : 'text-green-400'}>
          {fps}
        </span>
      </div>
      <div className="flex justify-between gap-3">
        <span>Latency:</span>
        <span className={latency > 200 ? 'text-red-400' : latency > 100 ? 'text-yellow-400' : 'text-green-400'}>
          {Math.round(latency)}ms
        </span>
      </div>
      <div className="flex justify-between gap-3">
        <span>Server:</span>
        <span>{Math.round(serverProcessMs)}ms</span>
      </div>
    </div>
  )
}

// Exercise selector
function ExerciseSelector({ exercises, selected, onChange, disabled }) {
  return (
    <div className="flex flex-wrap gap-2">
      {exercises.map(ex => (
        <button
          key={ex.id}
          onClick={() => onChange(ex.id)}
          disabled={disabled}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors
            ${selected === ex.id 
              ? 'bg-primary-600 text-white' 
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {ex.name}
        </button>
      ))}
    </div>
  )
}

// Main LiveAnalysis component
export default function LiveAnalysis({ onBack }) {
  // Refs
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const wsRef = useRef(null)
  const streamRef = useRef(null)
  const animationRef = useRef(null)
  const renderLoopRef = useRef(null)
  
  // State
  const [wsState, setWsState] = useState(WS_STATE.DISCONNECTED)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [exercises, setExercises] = useState([])
  const [selectedExercise, setSelectedExercise] = useState('deep_squat')
  const [feedback, setFeedback] = useState(null)
  const [finalResult, setFinalResult] = useState(null)
  const [cameraError, setCameraError] = useState(null)
  const [showPerformance, setShowPerformance] = useState(true)
  const [showModelSettings, setShowModelSettings] = useState(false)
  
  // Performance metrics
  const [fps, setFps] = useState(0)
  const [latency, setLatency] = useState(0)
  const [serverProcessMs, setServerProcessMs] = useState(0)
  
  // FPS and timing tracking
  const frameCountRef = useRef(0)
  const lastFpsTimeRef = useRef(Date.now())
  const frameSequenceRef = useRef(0)
  const frameSendTimesRef = useRef(new Map()) // Map sequence -> send timestamp
  
  // Skeleton interpolation state
  const prevSkeletonRef = useRef(null)
  const currentSkeletonRef = useRef(null)
  const targetSkeletonRef = useRef(null)
  const problemJointsRef = useRef([])
  const lastSkeletonUpdateRef = useRef(0)
  const interpolationStartRef = useRef(0)
  
  // Session ID
  const sessionIdRef = useRef(`live-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`)
  
  // Render loop for smooth skeleton animation
  const startRenderLoop = useCallback(() => {
    if (renderLoopRef.current) return
    
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d', { willReadFrequently: false, alpha: true })
    const INTERPOLATION_DURATION = 66 // ms between pose updates (adjust based on send FPS)
    
    const render = () => {
      const now = performance.now()
      
      if (targetSkeletonRef.current && currentSkeletonRef.current) {
        // Calculate interpolation progress
        const elapsed = now - interpolationStartRef.current
        const t = Math.min(elapsed / INTERPOLATION_DURATION, 1)
        
        // Ease-out interpolation for smoother motion
        const easedT = 1 - Math.pow(1 - t, 2)
        
        const interpolated = interpolateSkeleton(
          currentSkeletonRef.current,
          targetSkeletonRef.current,
          easedT
        )
        
        drawSkeleton(
          ctx,
          interpolated,
          problemJointsRef.current,
          canvas.width,
          canvas.height
        )
      } else if (currentSkeletonRef.current) {
        drawSkeleton(
          ctx,
          currentSkeletonRef.current,
          problemJointsRef.current,
          canvas.width,
          canvas.height
        )
      }
      
      renderLoopRef.current = requestAnimationFrame(render)
    }
    
    renderLoopRef.current = requestAnimationFrame(render)
  }, [])
  
  const stopRenderLoop = useCallback(() => {
    if (renderLoopRef.current) {
      cancelAnimationFrame(renderLoopRef.current)
      renderLoopRef.current = null
    }
  }, [])
  
  // Fetch available exercises
  useEffect(() => {
    fetch('/ws/exercises')
      .then(res => res.json())
      .then(data => setExercises(data.exercises || []))
      .catch(err => console.error('Failed to fetch exercises:', err))
  }, [])
  
  // Start camera with optimized settings
  const startCamera = useCallback(async () => {
    try {
      setCameraError(null)
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640, max: 640 },
          height: { ideal: 480, max: 480 },
          frameRate: { ideal: 30, max: 30 },
          facingMode: 'user'
        },
        audio: false
      })
      
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
    } catch (err) {
      console.error('Camera error:', err)
      setCameraError(err.message || 'Failed to access camera')
    }
  }, [])
  
  // Stop camera
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
  }, [])
  
  // Connect WebSocket
  const connectWebSocket = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/live/${sessionIdRef.current}`
    
    setWsState(WS_STATE.CONNECTING)
    
    const ws = new WebSocket(wsUrl)
    
    ws.onopen = () => {
      setWsState(WS_STATE.CONNECTED)
      console.log('WebSocket connected')
    }
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      if (data.type === 'started') {
        setIsAnalyzing(true)
        startRenderLoop()
      } else if (data.type === 'feedback') {
        setFeedback(data)
        
        // Calculate round-trip latency
        const sequence = data.frame_sequence
        if (frameSendTimesRef.current.has(sequence)) {
          const sendTime = frameSendTimesRef.current.get(sequence)
          const rtt = performance.now() - sendTime
          setLatency(prev => prev * 0.8 + rtt * 0.2) // Smooth latency
          frameSendTimesRef.current.delete(sequence)
          
          // Clean up old entries
          if (frameSendTimesRef.current.size > 100) {
            const keys = Array.from(frameSendTimesRef.current.keys()).slice(0, 50)
            keys.forEach(k => frameSendTimesRef.current.delete(k))
          }
        }
        
        // Update server processing time
        if (data.server_process_ms) {
          setServerProcessMs(prev => prev * 0.8 + data.server_process_ms * 0.2)
        }
        
        // Update skeleton for interpolation
        if (data.skeleton && data.skeleton.length > 0) {
          prevSkeletonRef.current = currentSkeletonRef.current
          currentSkeletonRef.current = targetSkeletonRef.current || data.skeleton
          targetSkeletonRef.current = data.skeleton
          interpolationStartRef.current = performance.now()
        }
        
        problemJointsRef.current = data.problem_joints || []
        
        // Handle audio cues (using Web Speech API)
        if (data.audio_cue && data.audio_priority >= 2) {
          speakCue(data.audio_cue)
        }
      } else if (data.type === 'stopped') {
        setIsAnalyzing(false)
        stopRenderLoop()
        setFinalResult(data)
      } else if (data.type === 'exercise_changed') {
        console.log('Exercise changed to:', data.exercise)
        // Reset skeleton state
        prevSkeletonRef.current = null
        currentSkeletonRef.current = null
        targetSkeletonRef.current = null
      }
    }
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setWsState(WS_STATE.ERROR)
    }
    
    ws.onclose = () => {
      setWsState(WS_STATE.DISCONNECTED)
      setIsAnalyzing(false)
      stopRenderLoop()
    }
    
    wsRef.current = ws
  }, [startRenderLoop, stopRenderLoop])
  
  // Disconnect WebSocket
  const disconnectWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
  }, [])
  
  // Reusable canvas for frame capture
  const captureCanvasRef = useRef(null)
  
  // Send frame to server with timing info
  const sendFrame = useCallback(() => {
    if (!videoRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return
    }
    
    // Reuse canvas for performance
    if (!captureCanvasRef.current) {
      captureCanvasRef.current = document.createElement('canvas')
      captureCanvasRef.current.width = 640
      captureCanvasRef.current.height = 480
    }
    
    const tempCanvas = captureCanvasRef.current
    const ctx = tempCanvas.getContext('2d', { willReadFrequently: false })
    
    // Draw mirrored video frame
    ctx.setTransform(-1, 0, 0, 1, tempCanvas.width, 0)
    ctx.drawImage(videoRef.current, 0, 0, tempCanvas.width, tempCanvas.height)
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    
    // Convert to JPEG with reduced quality for faster transfer
    const frameData = tempCanvas.toDataURL('image/jpeg', 0.6)
    
    // Track timing
    const sequence = frameSequenceRef.current++
    const sendTime = performance.now()
    frameSendTimesRef.current.set(sequence, sendTime)
    
    wsRef.current.send(JSON.stringify({
      type: 'frame',
      data: frameData,
      timestamp: Date.now(),
      sequence: sequence
    }))
    
    // Track send FPS
    frameCountRef.current++
    const now = Date.now()
    if (now - lastFpsTimeRef.current >= 1000) {
      setFps(frameCountRef.current)
      frameCountRef.current = 0
      lastFpsTimeRef.current = now
    }
  }, [])
  
  // Animation loop for sending frames at controlled rate
  const startFrameLoop = useCallback(() => {
    let lastFrameTime = 0
    const targetFPS = 15 // Send 15 frames per second for real-time performance
    const frameInterval = 1000 / targetFPS
    
    const loop = (timestamp) => {
      if (timestamp - lastFrameTime >= frameInterval) {
        sendFrame()
        lastFrameTime = timestamp
      }
      animationRef.current = requestAnimationFrame(loop)
    }
    
    animationRef.current = requestAnimationFrame(loop)
  }, [sendFrame])
  
  // Stop frame loop
  const stopFrameLoop = useCallback(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
      animationRef.current = null
    }
  }, [])
  
  // Text-to-speech for audio cues (with rate limiting)
  const lastSpeakTimeRef = useRef(0)
  const speakCue = useCallback((text) => {
    const now = Date.now()
    if (now - lastSpeakTimeRef.current < 2000) return // 2 second cooldown
    
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel()
      
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.rate = 1.1
      utterance.pitch = 1.0
      utterance.volume = 0.8
      
      window.speechSynthesis.speak(utterance)
      lastSpeakTimeRef.current = now
    }
  }, [])
  
  // Start analysis
  const handleStart = useCallback(async () => {
    setFinalResult(null)
    setFeedback(null)
    
    // Reset skeleton state
    prevSkeletonRef.current = null
    currentSkeletonRef.current = null
    targetSkeletonRef.current = null
    problemJointsRef.current = []
    frameSequenceRef.current = 0
    frameSendTimesRef.current.clear()
    
    // Start camera if not already started
    if (!streamRef.current) {
      await startCamera()
    }
    
    // Connect WebSocket if not connected
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      connectWebSocket()
      // Wait for connection
      await new Promise(resolve => setTimeout(resolve, 500))
    }
    
    // Send start command
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'start',
        exercise: selectedExercise
      }))
      startFrameLoop()
    }
  }, [startCamera, connectWebSocket, selectedExercise, startFrameLoop])
  
  // Stop analysis
  const handleStop = useCallback(() => {
    stopFrameLoop()
    stopRenderLoop()
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'stop' }))
    }
    
    setIsAnalyzing(false)
    
    // Clear canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d')
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    }
    
    // Reset skeleton state
    prevSkeletonRef.current = null
    currentSkeletonRef.current = null
    targetSkeletonRef.current = null
  }, [stopFrameLoop, stopRenderLoop])
  
  // Change exercise
  const handleExerciseChange = useCallback((exerciseId) => {
    setSelectedExercise(exerciseId)
    
    if (isAnalyzing && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'change_exercise',
        exercise: exerciseId
      }))
    }
  }, [isAnalyzing])
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopFrameLoop()
      stopRenderLoop()
      stopCamera()
      disconnectWebSocket()
    }
  }, [stopFrameLoop, stopRenderLoop, stopCamera, disconnectWebSocket])
  
  // Initialize camera on mount
  useEffect(() => {
    startCamera()
  }, [startCamera])
  
  // Get current exercise info
  const currentExercise = exercises.find(e => e.id === selectedExercise)
  
  return (
    <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
      {/* Header */}
      <div className="bg-gray-50 border-b px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-gray-800">Live FMS Analysis</h2>
            <p className="text-sm text-gray-500">Real-time movement coaching</p>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setShowPerformance(!showPerformance)}
              className="text-xs text-gray-500 hover:text-gray-700"
            >
              {showPerformance ? 'Hide' : 'Show'} Stats
            </button>
            <button
              onClick={() => setShowModelSettings(true)}
              className="text-gray-600 hover:text-gray-800 flex items-center gap-2"
              title="AI Model Settings"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </button>
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
        </div>
      </div>
      
      <div className="p-6">
        {/* Exercise Selector */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">Select Exercise</label>
          <ExerciseSelector
            exercises={exercises}
            selected={selectedExercise}
            onChange={handleExerciseChange}
            disabled={false}
          />
        </div>
        
        {/* Video Area */}
        <div className="relative bg-black rounded-xl overflow-hidden" style={{ aspectRatio: '4/3' }}>
          {/* Video feed */}
          <video
            ref={videoRef}
            className="w-full h-full object-cover"
            style={{ transform: 'scaleX(-1)' }}
            autoPlay
            playsInline
            muted
          />
          
          {/* Pose overlay canvas */}
          <canvas
            ref={canvasRef}
            width={640}
            height={480}
            className="absolute inset-0 w-full h-full pointer-events-none"
            style={{ transform: 'scaleX(-1)' }}
          />
          
          {/* Camera error overlay */}
          {cameraError && (
            <div className="absolute inset-0 bg-black/80 flex items-center justify-center">
              <div className="text-center text-white p-6">
                <svg className="w-16 h-16 mx-auto mb-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <p className="text-lg font-medium mb-2">Camera Access Required</p>
                <p className="text-sm text-gray-400 mb-4">{cameraError}</p>
                <button
                  onClick={startCamera}
                  className="px-4 py-2 bg-primary-600 rounded-lg hover:bg-primary-700"
                >
                  Retry
                </button>
              </div>
            </div>
          )}
          
          {/* Status indicators */}
          <div className="absolute top-4 left-4 flex items-center gap-3">
            {/* Connection status */}
            <div className={`px-3 py-1 rounded-full text-xs font-medium flex items-center gap-2
              ${wsState === WS_STATE.CONNECTED ? 'bg-green-500/80' : 
                wsState === WS_STATE.CONNECTING ? 'bg-yellow-500/80' : 'bg-red-500/80'}
              text-white`}
            >
              <span className={`w-2 h-2 rounded-full ${isAnalyzing ? 'bg-white animate-pulse' : 'bg-white/50'}`}></span>
              {isAnalyzing ? 'Analyzing' : wsState === WS_STATE.CONNECTED ? 'Ready' : wsState}
            </div>
            
            {/* Performance Monitor */}
            {isAnalyzing && (
              <PerformanceMonitor
                fps={fps}
                latency={latency}
                serverProcessMs={serverProcessMs}
                visible={showPerformance}
              />
            )}
          </div>
          
          {/* Score indicator */}
          {isAnalyzing && feedback && (
            <div className="absolute top-4 right-4">
              <ScoreIndicator score={feedback.current_score} confidence={feedback.score_confidence} />
            </div>
          )}
          
          {/* Joint angles */}
          {isAnalyzing && feedback && feedback.joint_angles && (
            <div className="absolute bottom-4 left-4">
              <JointAngles angles={feedback.joint_angles} />
            </div>
          )}
          
          {/* Coaching cue */}
          {isAnalyzing && feedback && (
            <div className="absolute bottom-4 right-4 max-w-xs">
              <CoachingCue 
                primary={feedback.primary_cue} 
                secondary={feedback.secondary_cues}
                llmCue={feedback.llm_cue}
                llmModel={feedback.llm_model}
              />
            </div>
          )}
          
          {/* Form quality */}
          {isAnalyzing && feedback && (
            <div className="absolute top-4 left-1/2 -translate-x-1/2">
              <FormQualityIndicator quality={feedback.form_quality} />
            </div>
          )}
        </div>
        
        {/* Exercise Instructions */}
        {currentExercise && !isAnalyzing && (
          <div className="mt-4 bg-blue-50 rounded-lg p-4">
            <h3 className="font-medium text-blue-900 mb-2">{currentExercise.name}</h3>
            <p className="text-sm text-blue-700 mb-2">{currentExercise.description}</p>
            <ol className="text-sm text-blue-600 space-y-1">
              {currentExercise.instructions.map((step, idx) => (
                <li key={idx} className="flex items-start gap-2">
                  <span className="font-medium">{idx + 1}.</span>
                  <span>{step}</span>
                </li>
              ))}
            </ol>
          </div>
        )}
        
        {/* Controls */}
        <div className="mt-6 flex gap-4">
          {!isAnalyzing ? (
            <button
              onClick={handleStart}
              disabled={cameraError || wsState === WS_STATE.CONNECTING}
              className="flex-1 bg-primary-600 text-white py-3 px-6 rounded-lg font-medium
                       hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed
                       flex items-center justify-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Start Analysis
            </button>
          ) : (
            <button
              onClick={handleStop}
              className="flex-1 bg-red-600 text-white py-3 px-6 rounded-lg font-medium
                       hover:bg-red-700 transition-colors flex items-center justify-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
              </svg>
              Stop Analysis
            </button>
          )}
        </div>
        
        {/* Final Result */}
        {finalResult && (
          <div className="mt-6 bg-gray-50 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Analysis Complete</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-white rounded-lg p-4 text-center">
                <p className="text-sm text-gray-600">Final Score</p>
                <p className="text-3xl font-bold text-primary-600">{finalResult.score}/3</p>
              </div>
              <div className="bg-white rounded-lg p-4 text-center">
                <p className="text-sm text-gray-600">Frames Analyzed</p>
                <p className="text-3xl font-bold text-gray-800">{finalResult.frames_analyzed}</p>
              </div>
            </div>
            <div className="mt-4 text-center">
              <p className="text-sm text-gray-500">
                Confidence: {Math.round(finalResult.confidence * 100)}%
              </p>
            </div>
          </div>
        )}
      </div>
      
      {/* Model Settings Modal */}
      <ModelSettings
        isOpen={showModelSettings}
        onClose={() => setShowModelSettings(false)}
        onSave={(prefs) => {
          console.log('Model preferences saved:', prefs)
        }}
      />
    </div>
  )
}
