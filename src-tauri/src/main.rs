// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use tauri::{Manager, State};
use serde::{Deserialize, Serialize};
use tokio::fs;

// Model management structures
#[derive(Debug, Serialize, Deserialize, Clone)]
struct ModelInfo {
    id: String,
    name: String,
    size: String,
    description: String,
    recommended: String,
    files: Vec<ModelFile>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ModelFile {
    name: String,
    size: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct LocalModel {
    id: String,
    name: String,
    path: String,
    size: String,
    #[serde(rename = "isDownloaded")]
    is_downloaded: bool,
    #[serde(rename = "lastUsed")]
    last_used: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DownloadProgress {
    #[serde(rename = "modelId")]
    model_id: String,
    progress: f64,
    status: String,
    #[serde(rename = "downloadedSize")]
    downloaded_size: Option<String>,
    #[serde(rename = "totalSize")]
    total_size: Option<String>,
    speed: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelStatus {
    #[serde(rename = "model_loaded")]
    model_loaded: bool,
    #[serde(rename = "current_model")]
    current_model: String,
    #[serde(rename = "current_device")]
    current_device: String,
    #[serde(rename = "gpu_available")]
    gpu_available: bool,
    #[serde(rename = "gpu_memory")]
    gpu_memory: Option<String>,
    #[serde(rename = "model_info")]
    model_info: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SetupStatus {
    setup_needed: bool,
    downloaded_models: Vec<String>,
    models_count: i32,
}

#[derive(Debug, Serialize, Deserialize)]
struct TranslationRequest {
    text: String,
    source_lang: String,
    target_lang: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct TranslationResponse {
    translated_text: String,
    source_language: String,
    target_language: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct TTSRequest {
    text: String,
    language: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelLoadRequest {
    model_path: String,
    device: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelLoadResponse {
    success: bool,
    model_name: String,
    model_size: String,
    device: String,
    message: String,
}

// Application state
struct AppState {
    models_dir: PathBuf,
    backend_url: String,
    download_progress: Mutex<HashMap<String, DownloadProgress>>,
}

impl AppState {
    fn new() -> Self {
        let models_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".nllb_translator")
            .join("models");

        Self {
            models_dir,
            backend_url: "http://localhost:8000".to_string(),
            download_progress: Mutex::new(HashMap::new()),
        }
    }

    async fn ensure_models_dir(&self) -> Result<(), Box<dyn std::error::Error>> {
        fs::create_dir_all(&self.models_dir).await?;
        Ok(())
    }

    fn get_model_dir(&self, model_id: &str) -> PathBuf {
        self.models_dir.join(model_id.replace("/", "_"))
    }
}

// Tauri commands
#[tauri::command]
async fn get_setup_status(state: State<'_, AppState>) -> Result<SetupStatus, String> {
    let client = reqwest::Client::new();
    let response = client
        .get(&format!("{}/setup-status", state.backend_url))
        .send()
        .await
        .map_err(|e| format!("Failed to get setup status: {}", e))?;

    let status: SetupStatus = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse setup status: {}", e))?;

    Ok(status)
}

#[tauri::command]
async fn get_model_status(state: State<'_, AppState>) -> Result<ModelStatus, String> {
    let client = reqwest::Client::new();
    let response = client
        .get(&format!("{}/model-status", state.backend_url))
        .send()
        .await
        .map_err(|e| format!("Failed to get model status: {}", e))?;

    let status: ModelStatus = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse model status: {}", e))?;

    Ok(status)
}

#[tauri::command]
async fn load_model(
    request: ModelLoadRequest,
    state: State<'_, AppState>,
) -> Result<ModelLoadResponse, String> {
    let client = reqwest::Client::new();
    let response = client
        .post(&format!("{}/load-model", state.backend_url))
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("Failed to load model: {}", e))?;

    let result: ModelLoadResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse load model response: {}", e))?;

    Ok(result)
}

#[tauri::command]
async fn translate_text(
    request: TranslationRequest,
    state: State<'_, AppState>,
) -> Result<TranslationResponse, String> {
    let client = reqwest::Client::new();
    let response = client
        .post(&format!("{}/translate", state.backend_url))
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("Failed to translate: {}", e))?;

    let result: TranslationResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse translation response: {}", e))?;

    Ok(result)
}

#[tauri::command]
async fn text_to_speech(
    request: TTSRequest,
    state: State<'_, AppState>,
) -> Result<HashMap<String, String>, String> {
    let client = reqwest::Client::new();
    let response = client
        .post(&format!("{}/text-to-speech", state.backend_url))
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("Failed to generate TTS: {}", e))?;

    let result: HashMap<String, String> = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse TTS response: {}", e))?;

    Ok(result)
}

#[tauri::command]
async fn play_audio_file(
    filename: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let audio_url = format!("{}/audio/{}", state.backend_url, filename);
    
    // Use system default audio player
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(&["/C", "start", "", &audio_url])
            .spawn()
            .map_err(|e| format!("Failed to play audio: {}", e))?;
    }
    
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(&audio_url)
            .spawn()
            .map_err(|e| format!("Failed to play audio: {}", e))?;
    }
    
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(&audio_url)
            .spawn()
            .map_err(|e| format!("Failed to play audio: {}", e))?;
    }
    
    Ok(())
}

#[tauri::command]
async fn delete_audio_file(
    filename: String,
    state: State<'_, AppState>,
) -> Result<HashMap<String, String>, String> {
    let client = reqwest::Client::new();
    let response = client
        .delete(&format!("{}/audio/{}", state.backend_url, filename))
        .send()
        .await
        .map_err(|e| format!("Failed to delete audio: {}", e))?;

    let result: HashMap<String, String> = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse delete response: {}", e))?;

    Ok(result)
}

#[tauri::command]
async fn check_model_exists(
    model_id: String,
    state: State<'_, AppState>,
) -> Result<bool, String> {
    let model_dir = state.get_model_dir(&model_id);
    
    if !model_dir.exists() {
        return Ok(false);
    }

    // Check for required files
    let config_file = model_dir.join("config.json");
    let tokenizer_file = model_dir.join("tokenizer.json");
    
    if !config_file.exists() || !tokenizer_file.exists() {
        return Ok(false);
    }

    // Check for model files
    let pytorch_model = model_dir.join("pytorch_model.bin");
    let safetensors_model = model_dir.join("model.safetensors");
    
    Ok(pytorch_model.exists() || safetensors_model.exists())
}

#[tauri::command]
async fn get_local_models(state: State<'_, AppState>) -> Result<Vec<LocalModel>, String> {
    state.ensure_models_dir().await.map_err(|e| e.to_string())?;
    
    let mut models = Vec::new();
    let mut entries = fs::read_dir(&state.models_dir)
        .await
        .map_err(|e| format!("Failed to read models directory: {}", e))?;

    while let Some(entry) = entries.next_entry().await.map_err(|e| e.to_string())? {
        if entry.file_type().await.map_err(|e| e.to_string())?.is_dir() {
            let model_dir = entry.path();
            let config_file = model_dir.join("config.json");
            
            if config_file.exists() {
                // Check for model files
                let pytorch_model = model_dir.join("pytorch_model.bin");
                let safetensors_model = model_dir.join("model.safetensors");
                
                if pytorch_model.exists() || safetensors_model.exists() {
                    // Calculate directory size
                    let size = calculate_dir_size(&model_dir).await.unwrap_or(0);
                    let size_gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
                    
                    // Get model ID from directory name
                    let model_id = model_dir
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .replace("_", "/");
                    
                    let model_name = get_model_display_name(&model_id);
                    
                    models.push(LocalModel {
                        id: model_id,
                        name: model_name,
                        path: model_dir.to_string_lossy().to_string(),
                        size: format!("{:.1}GB", size_gb),
                        is_downloaded: true,
                        last_used: None,
                    });
                }
            }
        }
    }

    Ok(models)
}

#[tauri::command]
async fn download_model(
    model_id: String,
    model_info: ModelInfo,
    app_handle: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<HashMap<String, String>, String> {
    let model_dir = state.get_model_dir(&model_id);
    
    // Create model directory
    fs::create_dir_all(&model_dir)
        .await
        .map_err(|e| format!("Failed to create model directory: {}", e))?;

    // Emit initial progress
    let progress = DownloadProgress {
        model_id: model_id.clone(),
        progress: 0.0,
        status: "downloading".to_string(),
        downloaded_size: Some("0MB".to_string()),
        total_size: Some(model_info.size.clone()),
        speed: None,
    };
    
    let _ = app_handle.emit_all("download_progress", &progress);

    // Use the backend to download the model
    let client = reqwest::Client::new();
    let download_request = serde_json::json!({
        "modelId": model_id,
        "modelInfo": model_info
    });

    let response = client
        .post(&format!("{}/download-model", state.backend_url))
        .json(&download_request)
        .send()
        .await
        .map_err(|e| format!("Failed to download model: {}", e))?;

    if response.status().is_success() {
        // Emit completion progress
        let progress = DownloadProgress {
            model_id: model_id.clone(),
            progress: 100.0,
            status: "completed".to_string(),
            downloaded_size: Some(model_info.size.clone()),
            total_size: Some(model_info.size.clone()),
            speed: None,
        };
        
        let _ = app_handle.emit_all("download_progress", &progress);

        let result: HashMap<String, String> = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse download response: {}", e))?;

        Ok(result)
    } else {
        Err(format!("Download failed with status: {}", response.status()))
    }
}

#[tauri::command]
async fn delete_model(
    model_id: String,
    state: State<'_, AppState>,
) -> Result<HashMap<String, String>, String> {
    let model_dir = state.get_model_dir(&model_id);
    
    if model_dir.exists() {
        fs::remove_dir_all(&model_dir)
            .await
            .map_err(|e| format!("Failed to delete model: {}", e))?;
        
        let mut result = HashMap::new();
        result.insert("success".to_string(), "true".to_string());
        result.insert("message".to_string(), format!("Model {} deleted successfully", model_id));
        Ok(result)
    } else {
        Err("Model not found".to_string())
    }
}

// Helper functions
fn calculate_dir_size(dir: &PathBuf) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<u64, Box<dyn std::error::Error>>> + Send>> {
    let dir = dir.clone();
    Box::pin(async move {
        let mut size = 0;
        let mut entries = fs::read_dir(&dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let metadata = entry.metadata().await?;
            if metadata.is_file() {
                size += metadata.len();
            } else if metadata.is_dir() {
                size += calculate_dir_size(&entry.path()).await?;
            }
        }
        
        Ok(size)
    })
}

fn get_model_display_name(model_id: &str) -> String {
    match model_id {
        "facebook/nllb-200-distilled-600M" => "NLLB-200 Distilled 600M".to_string(),
        "facebook/nllb-200-1.3B" => "NLLB-200 1.3B".to_string(),
        "facebook/nllb-200-3.3B" => "NLLB-200 3.3B".to_string(),
        "facebook/nllb-200-distilled-1.3B" => "NLLB-200 Distilled 1.3B".to_string(),
        _ => format!("Custom Model: {}", model_id.split('/').last().unwrap_or(model_id)),
    }
}

fn main() {
    tauri::Builder::default()
        .manage(AppState::new())
        .invoke_handler(tauri::generate_handler![
            get_setup_status,
            get_model_status,
            load_model,
            translate_text,
            text_to_speech,
            play_audio_file,
            delete_audio_file,
            check_model_exists,
            get_local_models,
            download_model,
            delete_model
        ])
        .setup(|app| {
            // Initialize app state and create models directory
            let app_handle = app.handle();
            tauri::async_runtime::spawn(async move {
                let state: State<AppState> = app_handle.state();
                let _ = state.ensure_models_dir().await;
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
