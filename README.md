# Lunor Translator 🔮

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React"/>
  <img src="https://img.shields.io/badge/Tauri-FFC131?style=for-the-badge&logo=tauri&logoColor=black" alt="Tauri"/>
  <img src="https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white" alt="Rust"/>
</p>

<p align="center">
  <img src="[INSERT YOUR APP SCREENSHOT OR LOGO HERE]" alt="Lunor Translator Screenshot" width="700"/>
</p>

<p align="center">
  <strong>A cross-platform, offline, AI-powered translation application for fast, private, and accurate multilingual communication.</strong>
</p>

---

## Table of Contents

1.  [Overview](#overview)
2.  [Key Features](#key-features)
3.  [Core Technology & Performance](#core-technology--performance)
4.  [Technology Stack](#technology-stack)
5.  [Repository Structure](#repository-structure)
6.  [Installation and Usage](#installation-and-usage)
    * [Prerequisites](#prerequisites)
    * [Backend Setup](#backend-setup)
    * [Frontend Setup](#frontend-setup)
7.  [Future Work](#future-work)
8.  [Contact](#contact)

---

## Overview

In an era where information is critical, language barriers pose a significant challenge, especially in sectors requiring high security and data privacy. Commercial online translation services, while powerful, are not viable solutions due to their reliance on internet connectivity and the inherent security risks of sending sensitive data to third-party servers.

**Lunor Translator** was built to solve this problem. It's a desktop application that brings the power of state-of-the-art neural machine translation directly to your machine. It operates **entirely offline**, ensuring that your data never leaves your computer. The application is powered by Meta's groundbreaking **NLLB-200 (No Language Left Behind)** model, providing high-quality, nuanced translations for over 200 languages.

---

## Key Features

* 🔒 **Completely Offline:** All translation is processed locally. No data ever leaves your computer.
* ⚡ **State-of-the-Art Accuracy:** Powered by the NLLB-200 model for high-quality translation.
* 🌐 **Cross-Platform:** Designed to run natively on Windows, macOS, and Linux.
* ✨ **Modern UI:** A clean, intuitive, and responsive user interface built with React and Tauri.
* 📂 **Custom Model Support:** Load any compatible Hugging Face model from your local filesystem.

---

## Core Technology & Performance

Lunor Translator is powered by Meta's **NLLB-200 (No Language Left Behind)** model family, the state-of-the-art in open-source multilingual translation. This model was chosen after a comprehensive evaluation against other models like M2M100 and online baselines.

#### State-of-the-Art Burmese Translation
A key success of this project was demonstrating the application's powerful capabilities for low-resource languages. In benchmark tests on a custom-curated dataset, Lunor Translator achieved a **BLEU score of 25.02** for Burmese-to-English translation. This result significantly surpassed the Google Translate baseline, which scored **9.14**, proving the immense potential of using high-quality offline models for languages that are often underserved by major online platforms.

| Model                 | Burmese-English BLEU Score |
| :-------------------- | :------------------------- |
| Google Translate      | 9.14                       |
| **NLLB-200 1.3B** | **25.02** |

---

## Technology Stack

| Category          | Technology                                   |
| :---------------- | :------------------------------------------- |
| **AI/ML Model** | Facebook NLLB-200                            |
| **Backend** | Python, FastAPI, Hugging Face `transformers` |
| **Frontend** | React.js, Next.js, shadcn/ui                 |
| **Desktop Framework**| Tauri (with Rust Core)                       |

---

## Repository Structure

The project is organized with a clear separation between the frontend application and the backend scripts.


nllb/
├── app/                  # The React/Tauri frontend application
│   ├── components/
│   └── ...
├── scripts/
│   └── backend/          # The Python backend server
│       ├── main.py
│       ├── run_server.py
│       └── requirements.txt
└── ... (other config files)


---

## Installation and Usage

Follow these instructions to set up and run the application from the source code.

### Prerequisites

Ensure you have the following installed on your system:
* **Python** (version 3.10 or higher)
* **Node.js** (LTS version)
* **Rust** (stable toolchain, installed via `rustup`)
* **Git**
* **(Windows Only)** **WiX Toolset v3**: Required by Tauri to build the Windows installer (`.msi`). [Download the `wix314.exe` installer here](https://github.com/wixtoolset/wix3/releases/tag/wix3141rtm).

### Backend Setup

1.  **Navigate to the backend directory:**
    ```bash
    cd nllb/scripts/backend
    ```
2.  **(Recommended) Create and activate a virtual environment:**
    * **Windows:** `python -m venv venv` followed by `.\venv\Scripts\activate`
    * **macOS / Linux:** `python3 -m venv venv` followed by `source venv/bin/activate`
3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the backend server:**
    ```bash
    python run_server.py
    ```
    The server will start on `http://localhost:8000`. **Keep this terminal window open.**

### Frontend Setup

Open a **new terminal window**.

1.  **Navigate to the frontend directory:**
    ```bash
    cd nllb/app
    ```
2.  **Install Node.js dependencies:**
    ```bash
    npm install
    ```
3.  **Launch the Lunor Translator application:**
    ```bash
    npm run tauri dev
    ```
    The desktop application window will launch. You can now load a model via the settings panel and start translating.

---

## Future Work

* **Standalone Executable:** Solve the Tauri sidecar challenge to bundle the Python backend into a single, double-clickable application.
* **Model Fine-Tuning:** Create a larger, cleaner corpus for specific language pairs and fine-tune NLLB to reduce errors like named entity hallucinations.
* **Feature Expansion:** Add more features, such as full document translation and expanding Text-to-Speech (TTS) support.

---

## Contact

Harsh Tripathi - [harsht@iitbhilai.ac.in](mailto:harsht@iitbhilai.ac.in)
