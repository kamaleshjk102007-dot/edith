# 🤖 EDITH AI – Multi-Provider Intelligent AI Assistant

EDITH AI is an intelligent desktop AI assistant developed as a capstone project. Unlike traditional AI assistants that rely on a single AI provider, EDITH integrates multiple AI providers and automatically switches between them when one becomes unavailable or reaches its usage limit.

This ensures a seamless, reliable, and uninterrupted AI experience.

---

## 🎥 Project Demo

📺 **Watch the Demo Video**

https://youtu.be/oIT5tEaFCSU

---

## ✨ Features

* 🤖 Multi-AI Provider Integration
* 🔄 Automatic Provider Fallback
* 💬 AI Chat Assistant
* 🎙️ Voice Interaction (Speech-to-Text & Text-to-Speech)
* 📄 Document Analysis
* 🖼️ Image Generation
* 🎥 Video Generation
* 🌤️ Weather Information
* 🗺️ Maps Integration
* 📧 Email Access
* 📂 Open Applications with Voice Commands
* 💻 Real-Time System Monitor

  * CPU Usage
  * Memory Usage
  * Battery Status
* 🛡️ Virus Shield (File Scanning)
* 🧠 Conversation Memory
* ⚡ Fast & Responsive User Interface

---

## 🏗️ Architecture

```
                +----------------+
                |     User       |
                +--------+-------+
                         |
                         v
                +----------------+
                |    EDITH AI    |
                +--------+-------+
                         |
     +-------------------+-------------------+
     |                   |                   |
     v                   v                   v
 Gemini API         Groq API         OpenRouter
     |                   |                   |
     +-------------------+-------------------+
                         |
               Automatic Provider
                    Fallback
                         |
                         v
                 Final AI Response
```

---

## 🛠️ Technologies Used

* Python
* Tkinter
* Gemini API
* Groq API
* OpenRouter
* Hugging Face
* HTML
* CSS
* JavaScript
* JSON
* REST APIs

---

## 📌 How It Works

1. User enters a prompt or uses voice input.
2. EDITH sends the request to the active AI provider.
3. If the provider is unavailable or rate-limited, EDITH automatically switches to the next available provider.
4. The response is displayed without interrupting the user experience.
5. Conversation history is maintained for context-aware interactions.

---

## 📷 Screenshots

Add screenshots of:

* Home Screen
* AI Chat
* Voice Assistant
* Image Generator
* Document Analysis
* System Monitor
* Virus Shield

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/kamaleshjk102007-dot/edith.git
```

### Navigate to the project

```bash
cd edith
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python main.py
```

---

## 📂 Project Structure

```
edith/
│
├── assets/
├── config/
├── providers/
├── modules/
├── utils/
├── models/
├── main.py
├── requirements.txt
└── README.md
```

---

## 🎯 Future Improvements

* Mobile Application
* Offline AI Models
* Face Recognition Login
* Smart Home Integration
* AI Agent Automation
* Multi-language Support
* Cloud Synchronization
* Enhanced Security Features

---

## 👨‍💻 Developer

**Kamalesh JK**

AI & Data Science Student

GitHub:
https://github.com/kamaleshjk102007-dot

---

## 📜 License

This project is developed for educational and research purposes as a capstone project.

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

Your support is greatly appreciated!
