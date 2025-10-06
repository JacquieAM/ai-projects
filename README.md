# AI Projects

A curated collection of AI and machine learning projects exploring practical applications, experiments, and techniques. This repository showcases hands-on implementations, interactive tools, and accessible AI solutions. Each project demonstrates problem-solving with modern AI models—including **OpenAI’s GPT, Ollama, and Claude**—and highlights technical, interactive, and inclusive design principles.

---

## Table of Contents

1. [Website Summary](#1-website-summary)  
2. [Website Summary with Gradio](#2-website-summary-with-gradio)  
3. [Quantum Tutor: Voice Assistant for Visually Impaired Learners](#3-quantum-tutor-voice-assistant-for-visually-impaired-learners)  
4. [Quantum Computing Tutor with GPT & Claude](#4-quantum-computing-tutor-with-gpt--claude)  
5. [OpenAI vs Ollama Streaming](#5-openai-vs-ollama-streaming)
6. [Quantum Synthetic Data Generator](#6-quantum-synthetic-data-generator)

---

## 1. Website Summary

**Project Overview:**  
This project explores the capabilities of AI models to summarize online training content. It extracts material from Microsoft’s **Quantum Computing Fundamentals** learning path, cleans it, and generates concise summaries per lesson as well as an overall course summary, allowing for model comparison.

**Key Features:**  
- Fetches and parses webpages using **requests** and **BeautifulSoup**  
- Generates summaries in multiple languages and levels of detail  
- Compares outputs from different AI models for clarity and accuracy  
- Produces clean, structured **Markdown** summaries  

**Tech Stack:**  
- **Models:** GPT-4o-mini, Ollama  
- **Language:** Python  
- **Libraries:** BeautifulSoup, OpenAI  

**Purpose:**  
Demonstrates how AI can streamline understanding of technical content and highlight performance differences across models.  

---

## 2. Website Summary with Gradio

**Project Overview:**  
An interactive extension of the previous notebook. Users can now summarize lessons or entire modules in real time through a **Gradio interface**, choosing models, languages, and summary lengths.

**Key Features:**  
- Lesson-level and full-course summaries  
- Dual model support (**GPT** and **Ollama**)  
- User-friendly **Gradio interface** for interactive exploration  
- Clean Markdown output for readability  

**Tech Stack:**  
- **Models:** GPT-4o-mini, Ollama  
- **Interface:** Gradio  
- **Language:** Python  
- **Libraries:** BeautifulSoup, OpenAI  

**Purpose:**  
Offers a hands-on way to explore AI-generated summaries and compare outputs interactively.  

---

## 3. Quantum Tutor: Voice Assistant for Visually Impaired Learners

**Project Overview:**  
This project transforms the summarization tool into a **voice-enabled AI assistant**, designed with accessibility in mind. Users can navigate, ask questions, and receive answers entirely via **voice**, making it especially useful for visually impaired learners.

**Key Features:**  
- **Accessible Voice Interaction:** speech-to-text input, text-to-speech output  
- **Screen Reader Optimization:** leverages Gradio’s ARIA-label support  
- Lesson-level and full-course summaries  
- Customizable summary length and language  
- Interactive chatbot flow for hands-free navigation  

**Tech Stack:**  
- **Models:** GPT-4o-mini  
- **Interface:** Gradio (ARIA-label enabled)  
- **Language:** Python  
- **Libraries:** BeautifulSoup, OpenAI, Speech-to-Text, Text-to-Speech  

**Purpose:**  
Demonstrates inclusive AI design by combining summarization with voice-first interaction, making technical education accessible to all learners.  

---

## 4. Quantum Computing Tutor with GPT & Claude

**Project Overview:**  
An interactive dual-AI notebook where GPT teaches and Claude asks questions, creating a dynamic, engaging learning experience for users.

**Key Features:**  
- Combines instruction and questioning from two AI models  
- Interactive learning that adapts to user responses  
- Highlights strengths of multi-AI collaboration for education  

**Tech Stack:**  
- **Models:** GPT-4o-mini, Claude  
- **Language:** Python  
- **Libraries:** OpenAI API  

**Purpose:**  
Showcases the synergy of multiple AI models in teaching complex topics in an engaging and adaptive way.  

---

## 5. OpenAI vs Ollama Streaming

**Project Overview:**  
This notebook demonstrates **real-time streaming** of LLM responses from OpenAI and Ollama. Users see tokens as they are generated instead of waiting for a complete response.

**Key Features:**  
- Streams AI responses in real time  
- Compares cloud-based vs local model inference  
- Highlights performance and latency differences  

**Tech Stack:**  
- **Models:** GPT-4o-mini, Ollama  
- **Language:** Python  
- **Libraries:** OpenAI, Ollama API  

**Purpose:**  
Illustrates the technical and UX differences between streaming cloud and local AI responses, emphasizing responsiveness in AI applications.  

---
## 6. Quantum Synthetic Data Generator

**Project Overview:**  
A synthetic data generator powered by open-source LLMs that creates realistic datasets for quantum computing use cases, including simulated circuits, experiment logs, and research abstracts. Users can interactively generate JSON or CSV datasets for experimentation, testing, or educational purposes.

**Key Features:**  
- Gradio-based interactive interface for dataset generation  
- Configurable dataset type and record counts (e.g., Quantum Circuits, Experiment Logs, Research Abstracts)  
- Downloadable JSON or CSV outputs ready for ML pipelines  
- Quantum-themed prompts for realistic and coherent outputs  
- Handles edge cases and extra text from model outputs, ensuring robust JSON parsing  

**Tech Stack:**  
- **Model:** Meta-Llama-3.1-8B-Instruct  
- **Interface:** Gradio  
- **Language:** Python  
- **Libraries:** Transformers, Hugging Face, Torch  

**Purpose:**  
Expands the Quantum AI learning ecosystem with synthetic datasets for experimentation, model testing, and educational content creation.

---
