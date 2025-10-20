# 🧠 Debunking Fashion using LLaVA (Vision-Language Model)

## 📖 Overview
**Debunking Fashion** is a multimodal AI project that uses **[LLaVA (Large Language and Vision Assistant)](https://ollama.com/library/llava)** to analyze and interpret visual content from fashion-related images.  
The model combines a **vision encoder** and **Vicuna-based language model** to provide intelligent, descriptive, and context-aware insights about clothing, style, and aesthetics — enabling creative and analytical applications in the fashion domain.

---

## 🧩 What is LLaVA?
**LLaVA** is a **Large Multimodal Model (LMM)** designed for:
- General-purpose **visual question answering (VQA)**  
- **Image captioning** and **visual reasoning**
- **Cross-modal understanding** — connecting text and vision

Under the hood:
- 🖼️ **Vision Encoder** → extracts rich visual features  
- 💬 **Vicuna LLM** → interprets them through natural language  

Together, they deliver high-quality visual-language reasoning.

---

## ⚙️ Setup & Installation

### 1️ Prerequisites
- **Python 3.10+**
- **[Ollama](https://ollama.com/download)** installed on your system
- A GPU (recommended) or CPU for local inference

### 2️ Pull the LLaVA model
```bash
ollama pull llava:7b
```

### 3 Install dependencies with uv pacakage manager and run app.py to launch GradIO app in your browser with localhost and a port.

