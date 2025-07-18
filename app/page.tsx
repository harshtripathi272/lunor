"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import {
  Loader2,
  Volume2,
  Languages,
  Moon,
  Sun,
  Upload,
  FileText,
  X,
  Settings,
  Folder,
  Info,
  Download,
  Trash2,
  HardDrive,
  CheckCircle,
  AlertCircle,
  Cloud,
  CloudOff,
  ExternalLink,
} from "lucide-react"
import { toast } from "@/hooks/use-toast"
import { useTheme } from "next-themes"

// Comprehensive NLLB language codes mapping
const LANGUAGES = {
  // Major World Languages
  eng_Latn: "English",
  spa_Latn: "Spanish",
  fra_Latn: "French",
  deu_Latn: "German",
  ita_Latn: "Italian",
  por_Latn: "Portuguese",
  rus_Cyrl: "Russian",
  jpn_Jpan: "Japanese",
  kor_Hang: "Korean",
  zho_Hans: "Chinese (Simplified)",
  zho_Hant: "Chinese (Traditional)",
  arb_Arab: "Arabic",

  // Indian Languages
  hin_Deva: "Hindi",
  ben_Beng: "Bengali",
  tel_Telu: "Telugu",
  mar_Deva: "Marathi",
  tam_Taml: "Tamil",
  urd_Arab: "Urdu",
  guj_Gujr: "Gujarati",
  kan_Knda: "Kannada",
  mal_Mlym: "Malayalam",
  pan_Guru: "Punjabi",
  ori_Orya: "Odia",
  asm_Beng: "Assamese",
  nep_Deva: "Nepali",
  sin_Sinh: "Sinhala",

  // Southeast Asian Languages
  tha_Thai: "Thai",
  vie_Latn: "Vietnamese",
  ind_Latn: "Indonesian",
  msa_Latn: "Malay",
  tgl_Latn: "Filipino",
  mya_Mymr: "Burmese",
  khm_Khmr: "Khmer",
  lao_Laoo: "Lao",

  // European Languages
  nld_Latn: "Dutch",
  swe_Latn: "Swedish",
  dan_Latn: "Danish",
  nor_Latn: "Norwegian",
  fin_Latn: "Finnish",
  pol_Latn: "Polish",
  ces_Latn: "Czech",
  hun_Latn: "Hungarian",
  ron_Latn: "Romanian",
  bul_Cyrl: "Bulgarian",
  hrv_Latn: "Croatian",
  srp_Cyrl: "Serbian",
  slk_Latn: "Slovak",
  slv_Latn: "Slovenian",
  est_Latn: "Estonian",
  lav_Latn: "Latvian",
  lit_Latn: "Lithuanian",
  ell_Grek: "Greek",

  // Middle Eastern Languages
  fas_Arab: "Persian",
  tur_Latn: "Turkish",
  heb_Hebr: "Hebrew",
  kur_Arab: "Kurdish",
  aze_Latn: "Azerbaijani",

  // African Languages
  swa_Latn: "Swahili",
  hau_Latn: "Hausa",
  yor_Latn: "Yoruba",
  ibo_Latn: "Igbo",
  amh_Ethi: "Amharic",
  som_Latn: "Somali",
  afr_Latn: "Afrikaans",

  // Other Languages
  ukr_Cyrl: "Ukrainian",
  bel_Cyrl: "Belarusian",
  kaz_Cyrl: "Kazakh",
  uzb_Latn: "Uzbek",
  tgk_Cyrl: "Tajik",
  mon_Cyrl: "Mongolian",
  cat_Latn: "Catalan",
  eus_Latn: "Basque",
  glg_Latn: "Galician",
  isl_Latn: "Icelandic",
  mlt_Latn: "Maltese",

  // Additional South Asian
  pus_Arab: "Pashto",
  snd_Arab: "Sindhi",
  bod_Tibt: "Tibetan",

  // Additional European
  gle_Latn: "Irish",
  cym_Latn: "Welsh",
  bre_Latn: "Breton",

  // Additional African
  xho_Latn: "Xhosa",
  zul_Latn: "Zulu",

  // Additional Asian
  tuk_Latn: "Turkmen",
  kir_Cyrl: "Kyrgyz",
}

// Available NLLB models - Enhanced with more options
const AVAILABLE_MODELS = {
  "facebook/nllb-200-distilled-600M": {
    id: "facebook/nllb-200-distilled-600M",
    name: "NLLB-200 Distilled 600M",
    size: "2.4GB",
    description: "Fastest, good quality, lower memory usage",
    recommended: "Recommended for most users",
    downloadUrl: "https://huggingface.co/facebook/nllb-200-distilled-600M",
    files: [
      { name: "pytorch_model.bin", size: "2.1GB" },
      { name: "config.json", size: "1KB" },
      { name: "tokenizer.json", size: "17MB" },
      { name: "tokenizer_config.json", size: "2KB" },
    ],
  },
  "facebook/nllb-200-1.3B": {
    id: "facebook/nllb-200-1.3B",
    name: "NLLB-200 1.3B",
    size: "5.2GB",
    description: "Balanced speed and quality",
    recommended: "Good balance of speed and accuracy",
    downloadUrl: "https://huggingface.co/facebook/nllb-200-1.3B",
    files: [
      { name: "pytorch_model.bin", size: "4.8GB" },
      { name: "config.json", size: "1KB" },
      { name: "tokenizer.json", size: "17MB" },
      { name: "tokenizer_config.json", size: "2KB" },
    ],
  },
  "facebook/nllb-200-3.3B": {
    id: "facebook/nllb-200-3.3B",
    name: "NLLB-200 3.3B",
    size: "13GB",
    description: "Higher quality, slower inference",
    recommended: "Best quality, requires more memory",
    downloadUrl: "https://huggingface.co/facebook/nllb-200-3.3B",
    files: [
      { name: "pytorch_model.bin", size: "12.5GB" },
      { name: "config.json", size: "1KB" },
      { name: "tokenizer.json", size: "17MB" },
      { name: "tokenizer_config.json", size: "2KB" },
    ],
  },
  "facebook/nllb-200-distilled-1.3B": {
    id: "facebook/nllb-200-distilled-1.3B",
    name: "NLLB-200 Distilled 1.3B",
    size: "5.2GB",
    description: "Distilled version with good performance",
    recommended: "Good alternative to standard 1.3B",
    downloadUrl: "https://huggingface.co/facebook/nllb-200-distilled-1.3B",
    files: [
      { name: "pytorch_model.bin", size: "4.8GB" },
      { name: "config.json", size: "1KB" },
      { name: "tokenizer.json", size: "17MB" },
      { name: "tokenizer_config.json", size: "2KB" },
    ],
  },
}

interface TranslationResponse {
  translated_text: string
  source_language: string
  target_language: string
}

interface ModelStatus {
  model_loaded: boolean
  current_model: string
  current_device: string
  gpu_available: boolean
  gpu_memory?: string
  model_info?: {
    name: string
    size: string
    parameters: string
  }
}

interface LocalModel {
  id: string
  name: string
  path: string
  size: string
  isDownloaded: boolean
  lastUsed?: string
}

interface SetupStatus {
  setup_needed: boolean
  downloaded_models: string[]
  models_count: number
}

// Detect if running in Tauri and if invoke is available
const isTauri = () => {
  if (typeof window === "undefined") return false
  // Use optional chaining to safely check for nested properties
  return !!(window as any).__TAURI__?.tauri?.invoke
}

// Add new state for audio management
// Add new state for audio management
// All state hooks at the top level - always called
const TranslationApp = () => {
  const [currentAudioFile, setCurrentAudioFile] = useState<string | null>(null)
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null)

  const [inputText, setInputText] = useState("")
  const [translatedText, setTranslatedText] = useState("")
  const [sourceLanguage, setSourceLanguage] = useState("")
  const [targetLanguage, setTargetLanguage] = useState("")
  const [isTranslating, setIsTranslating] = useState(false)
  const [isPlayingAudio, setIsPlayingAudio] = useState(false)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [isProcessingFile, setIsProcessingFile] = useState(false)
  const [showModelSettings, setShowModelSettings] = useState(false)
  const [customModelPath, setCustomModelPath] = useState("")
  const [isLoadingModel, setIsLoadingModel] = useState(false)
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null)
  const [localModels, setLocalModels] = useState<LocalModel[]>([])
  const [selectedDevice, setSelectedDevice] = useState("auto")
  const [showFirstTimeSetup, setShowFirstTimeSetup] = useState(false)
  const [setupStep, setSetupStep] = useState<"welcome" | "selection" | "downloading" | "complete" | "skipped">(
    "welcome",
  )
  const [selectedModelsForSetup, setSelectedModelsForSetup] = useState<string[]>(["facebook/nllb-200-1.3B"])
  const [isCheckingSetup, setIsCheckingSetup] = useState(true)
  const [modelAvailability, setModelAvailability] = useState<Record<string, boolean>>({})

  // Refs and theme hook
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { theme, setTheme } = useTheme()

  // API abstraction layer for web/Tauri compatibility
  const apiCall = async (endpoint: string, options: any = {}) => {
    if (isTauri()) {
      // Log the state of window.__TAURI__ before attempting to invoke
      console.log("Attempting Tauri API call:", endpoint)

      // Safely get invoke, if it's not available, isTauri() should have returned false
      const invoke = (window as any).__TAURI__.tauri.invoke

      try {
        switch (endpoint) {
          case "/model-status":
            return await invoke("get_model_status")
          case "/load-model":
            return await invoke("load_model", options.body ? JSON.parse(options.body) : {})
          case "/translate":
            return await invoke("translate_text", options.body ? JSON.parse(options.body) : {})
          case "/text-to-speech":
            return await invoke("text_to_speech", options.body ? JSON.parse(options.body) : {})
          case "/play-audio":
            return await invoke("play_audio_file", { filename: options.filename })
          case "/delete-audio":
            return await invoke("delete_audio_file", { filename: options.filename })
          case "/check-model-exists":
            return await invoke("check_model_exists", options.body ? JSON.parse(options.body) : {})
          case "/get-local-models":
            return await invoke("get_local_models")
          case "/delete-model":
            return await invoke("delete_model", options.body ? JSON.parse(options.body) : {})
          case "/setup-status":
            return await invoke("get_setup_status")
          default:
            throw new Error(`Unknown Tauri command: ${endpoint}`)
        }
      } catch (error) {
        console.error(`Tauri invoke error for ${endpoint}:`, error)
        throw error // Re-throw to be caught by the calling function
      }
    } else {
      // Web API calls
      const hasBody = !!options.body
      const method = options.method || (hasBody ? "POST" : "GET")

      // If method resolves to GET, ensure body is not sent to avoid fetch errors
      const fetchOptions: RequestInit = {
        method,
        headers: {
          "Content-Type": "application/json",
          ...options.headers,
        },
      }

      if (method !== "GET" && hasBody) {
        fetchOptions.body = options.body
      }

      const response = await fetch(`http://localhost:8000${endpoint}`, fetchOptions)

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ message: response.statusText }))
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorData.detail || errorData.message}`)
      }

      switch (endpoint) {
        case "/check-model-exists":
          const result = await response.json()
          return result.exists // Extract the exists field from the response
        default:
          return await response.json()
      }
    }
  }

  // All useEffect hooks at the top level - always called
  useEffect(() => {
    // Add a small delay to give Tauri API time to fully initialize
    const timer = setTimeout(() => {
      checkFirstTimeSetup()
    }, 500) // 500ms delay

    return () => clearTimeout(timer)
  }, []) // Only run once on mount

  // This useEffect should trigger checkModelAvailability when the app is ready
  // and when model settings are shown.
  useEffect(() => {
    // Only run if Tauri is detected and not in the middle of initial setup check
    if (isTauri() && !isCheckingSetup) {
      checkModelAvailability()
    }
  }, [isCheckingSetup, isTauri]) // Depend on isCheckingSetup to trigger after initial load

  // All function definitions (no hooks inside these)
  const checkFirstTimeSetup = async () => {
    try {
      setIsCheckingSetup(true)

      if (isTauri()) {
        const setupStatus: SetupStatus = await apiCall("/setup-status")

        if (setupStatus.setup_needed) {
          setShowFirstTimeSetup(true)
          setSetupStep("welcome")
        } else {
          // If setup not needed, load local models and check status
          await loadLocalModels()
          await checkModelStatus()
          // Also check model availability immediately after loading local models
          await checkModelAvailability()
        }
      } else {
        // For web, just check model status
        await checkModelStatus()
      }
    } catch (error) {
      console.error("Failed to check setup status:", error)
      // If setup check fails, assume setup is needed or proceed to main app
      // This ensures the app doesn't get stuck on a blank screen
      setShowFirstTimeSetup(true) // Force setup flow if initial check fails
      setSetupStep("welcome")
    } finally {
      setIsCheckingSetup(false)
    }
  }

  const checkModelAvailability = async () => {
    if (!isTauri()) return

    const availability: Record<string, boolean> = {}

    for (const modelId of Object.keys(AVAILABLE_MODELS)) {
      try {
        const exists = await apiCall("/check-model-exists", {
          body: JSON.stringify({ modelId }),
        })
        availability[modelId] = exists
      } catch (error) {
        console.error(`Failed to check model ${modelId}:`, error)
        availability[modelId] = false
      }
    }

    setModelAvailability(availability)
  }

  // Handle model selection for setup
  const handleSetupModelSelection = (modelId: string, selected: boolean) => {
    setSelectedModelsForSetup((prev) => (selected ? [...prev, modelId] : prev.filter((id) => id !== modelId)))
  }

  // Skip first-time setup
  const skipSetup = () => {
    setSetupStep("skipped")
    setTimeout(() => {
      setShowFirstTimeSetup(false)
      checkModelStatus()
    }, 1000)
  }

  // Check model status
  const checkModelStatus = async () => {
    try {
      const status = await apiCall("/model-status")
      setModelStatus(status)
    } catch (error) {
      console.error("Failed to check model status:", error)
    }
  }

  // Load local models
  const loadLocalModels = async () => {
    try {
      if (isTauri()) {
        const models = await apiCall("/get-local-models")
        setLocalModels(models)
      }
    } catch (error) {
      console.error("Failed to load local models:", error)
    }
  }

  // Check if model exists locally
  const checkModelExists = async (modelId: string): Promise<boolean> => {
    try {
      if (isTauri()) {
        const exists = await apiCall("/check-model-exists", {
          body: JSON.stringify({ modelId }),
        })
        console.log(`Model ${modelId} exists locally:`, exists) // Log existence check
        return exists
      } else {
        return modelStatus?.current_model === modelId && modelStatus?.model_loaded
      }
    } catch (error) {
      console.error("Failed to check model existence:", error)
      return false
    }
  }

  // Open Hugging Face model page
  const openModelPage = (modelId: string) => {
    const modelInfo = AVAILABLE_MODELS[modelId as keyof typeof AVAILABLE_MODELS]
    if (modelInfo?.downloadUrl) {
      window.open(modelInfo.downloadUrl, "_blank")
      toast({
        title: "Opening Hugging Face",
        description: `Opening ${modelInfo.name} page on Hugging Face for download.`,
      })
    }
  }

  // Delete model
  const deleteModel = async (modelId: string) => {
    if (!isTauri()) return

    // Confirm deletion
    const confirmed = window.confirm(
      `Are you sure you want to delete ${AVAILABLE_MODELS[modelId as keyof typeof AVAILABLE_MODELS]?.name || modelId}? This will free up ${AVAILABLE_MODELS[modelId as keyof typeof AVAILABLE_MODELS]?.size || "unknown"} of storage space.`,
    )

    if (!confirmed) return

    try {
      await apiCall("/delete-model", {
        body: JSON.stringify({ modelId }),
      })

      toast({
        title: "Model Deleted",
        description: "Model has been removed from your device.",
      })

      await loadLocalModels()
      await checkModelAvailability()

      // If the deleted model was currently loaded, clear the model status
      if (modelStatus?.current_model === modelId) {
        setModelStatus((prev) => (prev ? { ...prev, model_loaded: false, current_model: "" } : null))
      }
    } catch (error) {
      console.error("Delete error:", error)
      toast({
        title: "Delete Failed",
        description: "Failed to delete model.",
        variant: "destructive",
      })
    }
  }

  // Handle model change/loading
  const handleModelChange = async () => {
    const modelPath = customModelPath.trim()

    if (!modelPath) {
      toast({
        title: "Invalid Model Path",
        description: "Please provide a valid model path.",
        variant: "destructive",
      })
      return
    }

    setIsLoadingModel(true)
    console.log(`Attempting to load model: ${modelPath} on device: ${selectedDevice}`)

    try {
      const result = await apiCall("/load-model", {
        body: JSON.stringify({
          model_path: modelPath,
          device: selectedDevice,
        }),
      })

      toast({
        title: "Model Loaded Successfully",
        description: `${result.model_name} loaded on ${result.device.toUpperCase()}.`,
      })

      await checkModelStatus()
      setShowModelSettings(false)
      console.log(`Model ${modelPath} loaded and settings closed.`)
    } catch (error) {
      const errMsg =
        (error as Error)?.message || "Failed to load the selected model. Please check the path and device availability."
      console.error("Model loading error:", error)
      toast({
        title: "Model Loading Failed",
        description: errMsg,
        variant: "destructive",
      })
    } finally {
      setIsLoadingModel(false)
    }
  }

  // Handle file upload
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    // Check file size (limit to 10MB)
    if (file.size > 10 * 1024 * 1024) {
      toast({
        title: "File Too Large",
        description: "Please select a file smaller than 10MB.",
        variant: "destructive",
      })
      return
    }

    // Check file type
    const allowedTypes = [
      "text/plain",
      "application/pdf",
      "application/msword",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "text/csv",
      "application/json",
      "text/html",
      "text/xml",
    ]

    const fileExtension = file.name.split(".").pop()?.toLowerCase()
    const allowedExtensions = ["txt", "pdf", "doc", "docx", "csv", "json", "html", "xml", "md"]

    if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension || "")) {
      toast({
        title: "Unsupported File Type",
        description: "Please select a text file (.txt, .pdf, .doc, .docx, .csv, .json, .html, .xml, .md).",
        variant: "destructive",
      })
      return
    }

    setIsProcessingFile(true)
    setUploadedFile(file)

    try {
      let content = ""

      if (file.type === "text/plain" || fileExtension === "txt" || fileExtension === "md") {
        content = await file.text()
      } else if (file.type === "application/json" || fileExtension === "json") {
        const jsonContent = await file.text()
        try {
          const parsed = JSON.parse(jsonContent)
          content = JSON.stringify(parsed, null, 2)
        } catch {
          content = jsonContent
        }
      } else if (file.type === "text/html" || fileExtension === "html") {
        const htmlContent = await file.text()
        content = htmlContent
          .replace(/<[^>]*>/g, " ")
          .replace(/\s+/g, " ")
          .trim()
      } else if (file.type === "text/csv" || fileExtension === "csv") {
        content = await file.text()
      } else if (file.type === "application/pdf" || fileExtension === "pdf") {
        toast({
          title: "PDF File Detected",
          description:
            "PDF text extraction requires backend support. Please convert to .txt first or implement PDF parsing in the backend.",
          variant: "destructive",
        })
        setIsProcessingFile(false)
        setUploadedFile(null)
        return
      } else if (file.type.includes("word") || fileExtension === "doc" || fileExtension === "docx") {
        toast({
          title: "Word Document Detected",
          description:
            "Word document parsing requires backend support. Please convert to .txt first or implement document parsing in the backend.",
          variant: "destructive",
        })
        setIsProcessingFile(false)
        setUploadedFile(null)
        return
      } else {
        content = await file.text()
      }

      if (content.length > 50000) {
        content = content.substring(0, 50000) + "..."
        toast({
          title: "File Content Truncated",
          description: "File content was truncated to 50,000 characters for translation.",
        })
      }

      setInputText(content)
      toast({
        title: "File Loaded Successfully",
        description: `Loaded ${file.name} with ${content.length} characters.`,
      })
    } catch (error) {
      console.error("File processing error:", error)
      toast({
        title: "File Processing Failed",
        description: "Failed to read the file content. Please try a different file.",
        variant: "destructive",
      })
    } finally {
      setIsProcessingFile(false)
    }
  }

  // Remove uploaded file
  const handleRemoveFile = () => {
    setUploadedFile(null)
    setInputText("")
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  // Handle translation
  const handleTranslate = async () => {
    if (!inputText.trim() || !sourceLanguage || !targetLanguage) {
      toast({
        title: "Missing Information",
        description: "Please fill in all fields before translating.",
        variant: "destructive",
      })
      return
    }

    if (sourceLanguage === targetLanguage) {
      toast({
        title: "Same Languages",
        description: "Source and target languages cannot be the same.",
        variant: "destructive",
      })
      return
    }

    if (!modelStatus?.model_loaded) {
      toast({
        title: "No Model Loaded",
        description: "Please load a model first from the model settings.",
        variant: "destructive",
      })
      return
    }

    setIsTranslating(true)

    try {
      const data: TranslationResponse = await apiCall("/translate", {
        body: JSON.stringify({
          text: inputText,
          source_lang: sourceLanguage,
          target_lang: targetLanguage,
        }),
      })

      setTranslatedText(data.translated_text)

      toast({
        title: "Translation Complete",
        description: "Text has been successfully translated!",
      })
    } catch (error) {
      console.error("Translation error:", error)
      toast({
        title: "Translation Failed",
        description: "Failed to translate text. Please check if the backend server is running.",
        variant: "destructive",
      })
    } finally {
      setIsTranslating(false)
    }
  }

  // Handle text-to-speech
  const handlePlayAudio = async () => {
    if (!translatedText.trim()) {
      toast({
        title: "No Text to Play",
        description: "Please translate some text first.",
        variant: "destructive",
      })
      return
    }

    if (targetLanguage !== "eng_Latn" && targetLanguage !== "hin_Deva") {
      toast({
        title: "TTS Not Available",
        description: "Text-to-speech is only available for English and Hindi.",
        variant: "destructive",
      })
      return
    }

    // Check text length
    if (translatedText.length > 500) {
      toast({
        title: "Text Too Long",
        description: "Text-to-speech supports up to 500 characters. Please use shorter text.",
        variant: "destructive",
      })
      return
    }

    setIsPlayingAudio(true)

    try {
      // Generate TTS
      const ttsResponse = await apiCall("/text-to-speech", {
        body: JSON.stringify({
          text: translatedText,
          language: targetLanguage,
        }),
      })

      if (isTauri()) {
        // For Tauri, use system audio player
        await apiCall("/play-audio", { filename: ttsResponse.audio_file })

        toast({
          title: "Playing Audio",
          description: "Text-to-speech is now playing!",
        })

        // No immediate cleanup - let backend handle it after 60 seconds
      } else {
        // For web, use HTML5 audio
        const audioUrl = `http://localhost:8000/audio/${ttsResponse.audio_file}`

        // Clean up previous audio
        if (audioElement) {
          audioElement.pause()
          audioElement.src = ""
        }
        // Clean up previous audio
        if (audioElement) {
          audioElement.pause()
          audioElement.src = ""
          setAudioElement(null)
        }

// Create new audio element
        const audio = new Audio(audioUrl)
        audio.crossOrigin = "anonymous"

        audio.onloadstart = () => {
          toast({
            title: "Loading Audio",
            description: "Preparing text-to-speech...",
          })
        }

        audio.oncanplay = () => {
          toast({
            title: "Playing Audio",
            description: "Text-to-speech is now playing!",
          })
          audio.play()
        }

        audio.onended = () => {
          setIsPlayingAudio(false)
          
        }

        audio.onerror = (e) => {
          console.error("Audio playback error:", e, audio.error)
          setIsPlayingAudio(false)
          toast({
            title: "Audio Playback Failed",
            description: `Failed to play audio: ${audio.error?.message || "Unknown error"}`,
            variant: "destructive",
          })
        }

        setAudioElement(audio)
        setCurrentAudioFile(ttsResponse.audio_file)
      }
    } catch (error) {
      console.error("TTS error:", error)
      toast({
        title: "Audio Generation Failed",
        description: "Failed to generate or play audio. Please check if the backend server is running.",
        variant: "destructive",
      })
    } finally {
      if (isTauri()) {
        // For Tauri, stop loading immediately since system handles playback
        setIsPlayingAudio(false)
      }
      // For web, isPlayingAudio will be set to false in audio.onended
    }
  }

  // Add cleanup effect
  useEffect(() => {
    return () => {
      // Cleanup audio on component unmount
      if (audioElement) {
        audioElement.pause()
        audioElement.src = ""
      }
    }
  }, [audioElement, currentAudioFile])

  // Model Card Component - Simplified with Hugging Face redirect
  const ModelCard = ({
    modelId,
    model,
  }: {
    modelId: string
    model: any
  }) => {
    const isDownloaded = modelAvailability[modelId] || false

    return (
      <Card className="border-slate-600 hover:border-slate-500 transition-all duration-300">
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <h3 className="font-semibold text-slate-200">{model.name}</h3>
                <Badge variant="outline" className="text-xs">
                  {model.size}
                </Badge>
                {isDownloaded ? (
                  <Badge variant="default" className="text-xs bg-green-600">
                    <CheckCircle className="w-3 h-3 mr-1" />
                    Downloaded
                  </Badge>
                ) : (
                  <Badge variant="outline" className="text-xs text-orange-400 border-orange-400">
                    <Cloud className="w-3 h-3 mr-1" />
                    Not Downloaded
                  </Badge>
                )}
              </div>
              <p className="text-sm text-slate-400 mb-1">{model.description}</p>
              <p className="text-xs text-purple-400">{model.recommended}</p>
            </div>

            <div className="flex flex-col items-end gap-2 ml-4">
              <div className="flex gap-2">
                {isTauri() && isDownloaded && (
                  <Button
                    onClick={(e) => {
                      e.stopPropagation()
                      deleteModel(modelId)
                    }}
                    variant="outline"
                    size="sm"
                    className="text-red-400 hover:text-red-300 hover:border-red-400"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                )}
                <Button
                  onClick={() => openModelPage(modelId)}
                  variant="outline"
                  size="sm"
                  className="min-w-[120px] text-blue-400 hover:text-blue-300 hover:border-blue-400"
                >
                  <Download className="w-4 h-4 mr-1" />
                  Download
                  <ExternalLink className="w-3 h-3 ml-1" />
                </Button>
                {isDownloaded && (
                  <Badge variant="secondary" className="text-xs">
                    Ready to Use
                  </Badge>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  // Show loading screen while checking setup
  if (isCheckingSetup) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-purple-400 mx-auto mb-4" />
          <p className="text-slate-400">Initializing Lunor Translator...</p>
        </div>
      </div>
    )
  }

  if (showFirstTimeSetup) {
    return (
      // First time setup JSX - simplified without download functionality
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 dark:from-slate-950 dark:via-purple-950 dark:to-slate-950 p-4 relative overflow-hidden">
        {/* Animated background elements */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse"></div>
          <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse animation-delay-2000"></div>
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-80 h-80 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-10 animate-pulse animation-delay-4000"></div>
        </div>

        <div className="max-w-4xl mx-auto relative z-10 flex items-center justify-center min-h-screen">
          <Card className="glass-card border-slate-700/50 shadow-2xl shadow-purple-500/10 w-full max-w-2xl">
            <CardHeader className="text-center">
              <div className="flex items-center justify-center gap-3 mb-4">
                <Languages className="w-12 h-12 text-purple-400" />
                <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent">
                  Lunor Translator
                </h1>
              </div>
              <Badge variant="outline" className="mx-auto">
                Welcome
              </Badge>
            </CardHeader>

            <CardContent className="space-y-6">
              <div className="text-center space-y-6">
                <div className="space-y-4">
                  <h2 className="text-2xl font-bold text-slate-200">Welcome to Lunor Translator!</h2>
                  <p className="text-slate-400 leading-relaxed">
                    A powerful translation application using Meta's NLLB-200 model. Download models from Hugging Face
                    and load them locally for fast, private translation.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  <div className="p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                    <div className="text-green-400 font-semibold mb-2">🔒 Private</div>
                    <div className="text-slate-400">All translation happens locally on your device</div>
                  </div>
                  <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                    <div className="text-blue-400 font-semibold mb-2">⚡ Fast</div>
                    <div className="text-slate-400">No internet required after model download</div>
                  </div>
                  <div className="p-4 bg-purple-500/10 border border-purple-500/20 rounded-lg">
                    <div className="text-purple-400 font-semibold mb-2">🌍 200+ Languages</div>
                    <div className="text-slate-400">Support for major world languages</div>
                  </div>
                </div>

                <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700/50">
                  <div className="flex items-center gap-2 mb-2">
                    <Info className="w-4 h-4 text-blue-400" />
                    <span className="text-sm font-medium text-slate-300">Getting Started</span>
                  </div>
                  <p className="text-sm text-slate-400">
                    Use the model settings to download models from Hugging Face, then load them to start translating.
                  </p>
                </div>

                <div className="flex gap-4 justify-center">
                  <Button
                    onClick={skipSetup}
                    className="glow-button bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-semibold px-8"
                  >
                    Get Started
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  // Main app JSX
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 dark:from-slate-950 dark:via-purple-950 dark:to-slate-950 p-4 relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse animation-delay-2000"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-80 h-80 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-10 animate-pulse animation-delay-4000"></div>
      </div>

      <div className="max-w-4xl mx-auto relative z-10">
        {/* Header with theme toggle and model settings */}
        <div className="flex justify-between items-center mb-8">
          <div className="text-center flex-1">
            <div className="flex items-center justify-center gap-3 mb-4">
              <div className="relative">
                <Languages className="w-10 h-10 text-purple-400 drop-shadow-lg" />
                <div className="absolute inset-0 w-10 h-10 bg-purple-400 rounded-full blur-md opacity-30 animate-pulse"></div>
              </div>
              <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent drop-shadow-2xl">
                Lunor Translator
              </h1>
              {isTauri() && (
                <Badge variant="outline" className="ml-2 text-xs">
                  Desktop
                </Badge>
              )}
            </div>
            <p className="text-slate-300 dark:text-slate-400 text-lg font-medium">
              Translate text between 200+ languages using Meta's NLLB-200 model
            </p>
            {modelStatus && (
              <div className="mt-2 flex items-center justify-center gap-2">
                <Badge variant={modelStatus.model_loaded ? "default" : "destructive"} className="text-xs">
                  {modelStatus.model_loaded ? (
                    <>
                      <CheckCircle className="w-3 h-3 mr-1" />
                      Model Ready
                    </>
                  ) : (
                    <>
                      <AlertCircle className="w-3 h-3 mr-1" />
                      No Model Loaded
                    </>
                  )}
                </Badge>
                {modelStatus.model_info && (
                  <Badge variant="outline" className="text-xs">
                    {modelStatus.model_info.name}
                  </Badge>
                )}
                {isTauri() && modelStatus.model_loaded && modelAvailability[modelStatus.current_model] && (
                  <Badge variant="outline" className="text-xs text-green-400 border-green-400">
                    <CloudOff className="w-3 h-3 mr-1" />
                    Offline
                  </Badge>
                )}
              </div>
            )}
          </div>

          <div className="flex gap-2">
            <Button
              variant="outline"
              size="icon"
              onClick={() => setShowModelSettings(!showModelSettings)}
              className="glass-card border-slate-600 hover:border-purple-400 transition-all duration-300 hover:shadow-lg hover:shadow-purple-500/25"
            >
              <Settings className="h-5 w-5 text-purple-400" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              className="glass-card border-slate-600 hover:border-purple-400 transition-all duration-300 hover:shadow-lg hover:shadow-purple-500/25"
            >
              {theme === "dark" ? (
                <Sun className="h-5 w-5 text-yellow-400" />
              ) : (
                <Moon className="h-5 w-5 text-purple-400" />
              )}
            </Button>
          </div>
        </div>

        {/* Model Settings Card */}
        {showModelSettings && (
          <Card className="glass-card border-slate-700/50 shadow-2xl shadow-purple-500/10 mb-8 animate-in slide-in-from-top-4 duration-300">
            <CardHeader>
              <CardTitle className="text-2xl bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent flex items-center gap-2">
                <Settings className="w-6 h-6 text-purple-400" />
                Model Management
                <Badge variant="outline" className="ml-2 text-xs">
                  Download from Hugging Face
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Storage Info for Tauri */}
              {isTauri() && (
                <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700/50">
                  <div className="flex items-center gap-2 mb-2">
                    <HardDrive className="w-4 h-4 text-slate-400" />
                    <span className="text-sm font-medium text-slate-300">Storage Information</span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-400">
                    <div>
                      <span>Downloaded Models: </span>
                      <span className="text-slate-300">{Object.values(modelAvailability).filter(Boolean).length}</span>
                    </div>
                    <div>
                      <span>Available Models: </span>
                      <span className="text-slate-300">{Object.keys(AVAILABLE_MODELS).length}</span>
                    </div>
                  </div>
                </div>
              )}

              <div className="space-y-4">
                <Label className="text-sm font-semibold text-slate-300 dark:text-slate-400 tracking-wide">
                  Available NLLB Models
                </Label>
                <p className="text-xs text-slate-400">
                  Click "Download" to visit the Hugging Face model page. After downloading, use the custom model path
                  below to load them.
                </p>

                {Object.entries(AVAILABLE_MODELS).map(([modelId, modelInfo]) => (
                  <ModelCard key={modelId} modelId={modelId} model={modelInfo} />
                ))}

                {/* Custom Model Path Section */}
                <div className="space-y-3 pt-4 border-t border-slate-700/50">
                  <Label className="text-sm font-semibold text-slate-300 dark:text-slate-400 tracking-wide">
                    Load Model (Custom Path or Downloaded Model)
                  </Label>
                  <Input
                    placeholder="Enter model path (e.g., /path/to/model or facebook/nllb-200-1.3B)"
                    value={customModelPath}
                    onChange={(e) => setCustomModelPath(e.target.value)}
                    className="glass-input border-slate-600 hover:border-purple-400 focus:border-purple-400 transition-all duration-300"
                  />
                  <div className="flex items-start gap-2 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                    <Info className="w-4 h-4 text-blue-400 mt-0.5 flex-shrink-0" />
                    <div className="text-xs text-blue-300">
                      <p className="font-medium mb-1">Model Path Examples:</p>
                      <p>• Downloaded model: facebook/nllb-200-1.3B</p>
                      <p>• Local path: /home/user/models/nllb-custom</p>
                      <p>• Hugging Face: facebook/nllb-200-distilled-1.3B</p>
                      <p>• Fine-tuned model: your-username/nllb-finetuned</p>
                    </div>
                  </div>
                </div>

                {/* Device Selection Section */}
                <div className="space-y-3 pt-4 border-t border-slate-700/50">
                  <Label className="text-sm font-semibold text-slate-300 dark:text-slate-400 tracking-wide">
                    Compute Device
                  </Label>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    <div
                      className={`p-3 rounded-lg border transition-all duration-300 cursor-pointer ${
                        selectedDevice === "auto"
                          ? "border-purple-400 bg-purple-500/10"
                          : "border-slate-600 hover:border-slate-500"
                      }`}
                      onClick={() => setSelectedDevice("auto")}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <input
                          type="radio"
                          checked={selectedDevice === "auto"}
                          onChange={() => setSelectedDevice("auto")}
                          className="text-purple-400"
                        />
                        <h4 className="font-medium text-slate-200 text-sm">Auto</h4>
                      </div>
                      <p className="text-xs text-slate-400">Use GPU if available, fallback to CPU</p>
                    </div>

                    <div
                      className={`p-3 rounded-lg border transition-all duration-300 cursor-pointer ${
                        selectedDevice === "gpu"
                          ? "border-purple-400 bg-purple-500/10"
                          : "border-slate-600 hover:border-slate-500"
                      }`}
                      onClick={() => setSelectedDevice("gpu")}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <input
                          type="radio"
                          checked={selectedDevice === "gpu"}
                          onChange={() => setSelectedDevice("gpu")}
                          className="text-purple-400"
                        />
                        <h4 className="font-medium text-slate-200 text-sm">GPU</h4>
                      </div>
                      <p className="text-xs text-slate-400">Force GPU usage (faster inference)</p>
                    </div>

                    <div
                      className={`p-3 rounded-lg border transition-all duration-300 cursor-pointer ${
                        selectedDevice === "cpu"
                          ? "border-purple-400 bg-purple-500/10"
                          : "border-slate-600 hover:border-slate-500"
                      }`}
                      onClick={() => setSelectedDevice("cpu")}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <input
                          type="radio"
                          checked={selectedDevice === "cpu"}
                          onChange={() => setSelectedDevice("cpu")}
                          className="text-purple-400"
                        />
                        <h4 className="font-medium text-slate-200 text-sm">CPU</h4>
                      </div>
                      <p className="text-xs text-slate-400">Use CPU only (lower memory usage)</p>
                    </div>
                  </div>
                </div>

                <div className="flex gap-3 pt-4">
                  <Button
                    onClick={() => handleModelChange()}
                    disabled={isLoadingModel || !customModelPath.trim()}
                    className="glow-button bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-semibold transition-all duration-300"
                  >
                    {isLoadingModel ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Loading Model...
                      </>
                    ) : (
                      <>
                        <Folder className="w-4 h-4 mr-2" />
                        Load Model
                      </>
                    )}
                  </Button>
                  <Button
                    onClick={() => setShowModelSettings(false)}
                    variant="outline"
                    className="glass-card border-slate-600 hover:border-purple-400 transition-all duration-300"
                  >
                    Close
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Main translation card */}
        <Card className="glass-card border-slate-700/50 shadow-2xl shadow-purple-500/10 hover:shadow-purple-500/20 transition-all duration-500">
          <CardHeader className="text-center">
            <CardTitle className="text-3xl bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              Language Translation
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-8">
            {/* Language Selection */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <label className="text-sm font-semibold text-slate-300 dark:text-slate-400 tracking-wide">
                  Source Language
                </label>
                <Select value={sourceLanguage} onValueChange={setSourceLanguage}>
                  <SelectTrigger className="glass-input border-slate-600 hover:border-purple-400 focus:border-purple-400 transition-all duration-300 h-12">
                    <SelectValue placeholder="Select source language" />
                  </SelectTrigger>
                  <SelectContent className="glass-card border-slate-700 z-50 max-h-60 overflow-y-auto">
                    {Object.entries(LANGUAGES).map(([code, name]) => (
                      <SelectItem
                        key={code}
                        value={code}
                        className="hover:bg-purple-500/20 focus:bg-purple-500/20 transition-colors"
                      >
                        {name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-3">
                <label className="text-sm font-semibold text-slate-300 dark:text-slate-400 tracking-wide">
                  Target Language
                </label>
                <Select value={targetLanguage} onValueChange={setTargetLanguage}>
                  <SelectTrigger className="glass-input border-slate-600 hover:border-purple-400 focus:border-purple-400 transition-all duration-300 h-12">
                    <SelectValue placeholder="Select target language" />
                  </SelectTrigger>
                  <SelectContent className="glass-card border-slate-700 z-50 max-h-60 overflow-y-auto">
                    {Object.entries(LANGUAGES).map(([code, name]) => (
                      <SelectItem
                        key={code}
                        value={code}
                        className="hover:bg-purple-500/20 focus:bg-purple-500/20 transition-colors"
                      >
                        {name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* File Upload Section */}
            <div className="space-y-3">
              <label className="text-sm font-semibold text-slate-300 dark:text-slate-400 tracking-wide">
                Upload File (Optional)
              </label>
              <div className="flex items-center gap-4">
                <Input
                  ref={fileInputRef}
                  type="file"
                  accept=".txt,.pdf,.doc,.docx,.csv,.json,.html,.xml,.md"
                  onChange={handleFileUpload}
                  className="glass-input border-slate-600 hover:border-purple-400 focus:border-purple-400 transition-all duration-300 file:bg-purple-600 file:text-white file:border-0 file:rounded-md file:px-4 file:py-2 file:mr-4 file:hover:bg-purple-700"
                  disabled={isProcessingFile}
                />
                <Button
                  onClick={() => fileInputRef.current?.click()}
                  variant="outline"
                  size="sm"
                  className="glass-card border-slate-600 hover:border-purple-400 transition-all duration-300 hover:shadow-lg hover:shadow-purple-500/25 bg-transparent"
                  disabled={isProcessingFile}
                >
                  {isProcessingFile ? (
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <Upload className="w-4 h-4 mr-2" />
                  )}
                  Upload File
                </Button>
              </div>

              {uploadedFile && (
                <div className="flex items-center gap-2 p-3 glass-card border-slate-700/50 rounded-lg">
                  <FileText className="w-4 h-4 text-purple-400" />
                  <span className="text-sm text-slate-300 flex-1">{uploadedFile.name}</span>
                  <Button
                    onClick={handleRemoveFile}
                    variant="ghost"
                    size="sm"
                    className="h-6 w-6 p-0 hover:bg-red-500/20"
                  >
                    <X className="w-3 h-3 text-red-400" />
                  </Button>
                </div>
              )}

              <p className="text-xs text-slate-400">
                Supported formats: .txt, .csv, .json, .html, .xml, .md (PDF and Word docs require backend support)
              </p>
            </div>

            {/* Input Text */}
            <div className="space-y-3">
              <label className="text-sm font-semibold text-slate-300 dark:text-slate-400 tracking-wide">
                Text to Translate
              </label>
              <Textarea
                placeholder="Enter your text here or upload a file..."
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                maxLength={50000}
                className="glass-input border-slate-600 hover:border-purple-400 focus:border-purple-400 transition-all duration-300 min-h-[140px] resize-none text-slate-100 placeholder:text-slate-500"
              />
              <div className="flex justify-between text-xs text-slate-400">
                <span>{inputText.length} characters</span>
                <span>Max: 50,000 characters</span>
              </div>
            </div>

            {/* Translate Button */}
            <Button
              onClick={handleTranslate}
              disabled={isTranslating || !modelStatus?.model_loaded}
              className="w-full glow-button bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white py-4 text-lg font-semibold transition-all duration-300 transform hover:scale-[1.02] disabled:hover:scale-100"
            >
              {isTranslating ? (
                <>
                  <Loader2 className="w-6 h-6 mr-3 animate-spin" />
                  Translating...
                </>
              ) : !modelStatus?.model_loaded ? (
                <>
                  <Settings className="w-6 h-6 mr-3" />
                  Load Model First
                </>
              ) : (
                <>
                  <Languages className="w-6 h-6 mr-3" />
                  Translate
                </>
              )}
            </Button>

            {/* Translated Text */}
            {translatedText && (
              <div className="space-y-4 animate-in slide-in-from-bottom-4 duration-500">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-semibold text-slate-300 dark:text-slate-400 tracking-wide">
                    Translated Text
                  </label>
                  <div className="flex items-center gap-2">
                    {(targetLanguage === "eng_Latn" || targetLanguage === "hin_Deva") && (
                      <>
                        <Badge variant={translatedText.length <= 500 ? "default" : "destructive"} className="text-xs">
                          TTS: {translatedText.length}/500
                        </Badge>
                        <Button
                          onClick={handlePlayAudio}
                          disabled={isPlayingAudio || !translatedText.trim() || translatedText.length > 500}
                          variant="outline"
                          size="sm"
                          className="glass-card border-slate-600 hover:border-purple-400 transition-all duration-300 hover:shadow-lg hover:shadow-purple-500/25 bg-transparent"
                        >
                          {isPlayingAudio ? (
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          ) : (
                            <Volume2 className="w-4 h-4 mr-2" />
                          )}
                          {isPlayingAudio ? "Generating..." : "Play Audio"}
                        </Button>
                        <Button
                          onClick={async () => {
                            if (currentAudioFile) {
                              try {
                                if (isTauri()) {
                                  await apiCall("/delete-audio", { filename: currentAudioFile })
                                } else {
                                  await fetch(`http://localhost:8000/audio/${currentAudioFile}`, {
                                    method: "DELETE",
                                  })
                                }
                                setCurrentAudioFile(null)
                                setAudioElement(null)
                                toast({
                                  title: "Audio Deleted",
                                  description: "The audio file has been deleted.",
                                })
                              } catch (error) {
                                console.error("Audio deletion error:", error)
                                toast({
                                  title: "Audio Deletion Failed",
                                  description: "Failed to delete the audio file.",
                                  variant: "destructive",
                                })
                              }
                            }
                          }}
                          disabled={!currentAudioFile || isPlayingAudio}
                          variant="outline"
                          size="sm"
                          className="glass-card border-slate-600 hover:border-red-400 transition-all duration-300 hover:shadow-lg hover:shadow-red-500/25 bg-transparent"
                        >
                          <Trash2 className="w-4 h-4 mr-2" />
                          Delete Audio
                        </Button>
                      </>
                    )}
                  </div>
                </div>
                <div className="glass-card border-slate-700/50 p-6 rounded-xl">
                  <p className="text-slate-100 dark:text-slate-200 leading-relaxed text-lg font-medium">
                    {translatedText}
                  </p>
                </div>
                <div className="text-xs text-slate-400">
                  {translatedText.length} characters translated
                  {(targetLanguage === "eng_Latn" || targetLanguage === "hin_Deva") && (
                    <span className={translatedText.length > 500 ? "text-red-400 ml-2" : "text-green-400 ml-2"}>
                      • TTS {translatedText.length <= 500 ? "available" : "unavailable (too long)"}
                    </span>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Instructions Card */}
        <Card className="mt-8 glass-card border-slate-700/50 shadow-xl shadow-purple-500/5">
          <CardContent className="pt-6">
            <h3 className="font-bold text-xl text-slate-200 dark:text-slate-300 mb-4 flex items-center gap-2">
              <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></div>
              Instructions
            </h3>
            <ul className="text-slate-300 dark:text-slate-400 space-y-2 text-sm leading-relaxed">
              <li className="flex items-start gap-2">
                <span className="text-purple-400 mt-1">•</span>
                Click the settings icon to access model management and download models from Hugging Face
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-400 mt-1">•</span>
                Download models from Hugging Face by clicking the "Download" button, then load them using the custom
                model path
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-400 mt-1">•</span>
                Select your source and target languages from the dropdowns (200+ languages supported)
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-400 mt-1">•</span>
                Upload a text file (.txt, .csv, .json, .html, .xml, .md) or enter text manually (up to 50,000
                characters)
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-400 mt-1">•</span>
                Use "Play Audio" for English and Hindi translations (requires backend TTS)
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-400 mt-1">•</span>
                Choose your compute device (Auto/GPU/CPU) based on your hardware and performance needs
              </li>
            </ul>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default function Page() {
  return <TranslationApp />
}
