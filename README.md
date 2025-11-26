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
7. [LLM Benchmark: Generating and Explaining Quantum Code](#7-llm-benchmark-generating-and-explaining-quantum-code)
8. [Retrieval-Augmented Generation (RAG) Chatbot for Quantum Computing](#8-retrieval-augmented-generation-rag-chatbot-for-quantum-computing)
9. [RAG and Fine-Tuning Comparison on a Linux Commands Dataset](#9-rag-and-fine-tuning-comparison-on-a-linux-commands-dataset)
10. [RAG Linux Command Assistant + Automated Evaluation](#10-rag-linux-command-assistant--automated-evaluation)



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

## 7. LLM Benchmark: Generating and Explaining Quantum Code

**Project Overview:**
This project benchmarks AI models for quantum computing code generation, specifically OpenAI’s GPT-4o and Anthropic’s Claude-3.5-sonnet-20240620. The models are prompted to generate quantum computing code with beginner-friendly explanations. The project measures performance metrics such as latency, token usage, and throughput, and presents results in structured JSON and Markdown tables for easy comparison.

**Key Features:**
- Generates quantum computing code with explanations for educational purposes  
- Measures inference performance: latency, total tokens, tokens per second, and prompt vs completion tokens  
- Produces structured JSON output for analysis or visualization  
- Displays benchmarking results in Markdown tables for readability  

**Tech Stack:**
- **Models**: GPT-4o, Claude-3.5-sonnet-20240620  
- **Language**: Python  
- **Libraries**: OpenAI, Anthropic, time, json, IPython.display, tabulate  

**Purpose:**  
Compare LLM performance across models, highlighting latency, token usage, and throughput in a structured benchmark.

---

## 8. Retrieval-Augmented Generation (RAG) Chatbot for Quantum Computing

**Project Overview**
This project builds a Retrieval-Augmented Generation (RAG) system integrating LangChain, ChromaDB, and an OpenAI LLM to create a quantum computing chatbot. It loads a quantum computing dataset from Hugging Face, processes instruction–output pairs into document chunks, and generates embeddings for semantic retrieval and visualization.

**Key Features**
- Uses a quantum computing dataset from Hugging Face (Tonmoy-000/quantumcomputingecosystem_dataset)
- Creates embeddings using Hugging Face Sentence Transformers
- Stores and retrieves vector data with ChromaDB
- Visualizes embeddings with t-SNE and Matplotlib
- Provides an interactive Gradio chat interface connected to an OpenAI model

**Tech Stack**
- **Model:** GPT-4o-mini
- **Frameworks:** LangChain, ChromaDB
- **Libraries:** Hugging Face Datasets, Sentence Transformers, Matplotlib, NumPy, Gradio
- **Language:** Python

**Purpose**
Enable retrieval-augmented question answering on quantum computing topics using vector-based search over instruction–output data.

---

## 9. RAG and Fine-Tuning Comparison on a Linux Commands Dataset

**Project Overview**  
This project compares Retrieval-Augmented Generation (RAG) and Fine-Tuning approaches using the same dataset of Linux terminal commands. The goal is to evaluate how each method handles missing or incomplete information and to understand the trade-offs between retrieval-based and parameter-based learning.  

**Key Features**  
- Implements RAG using LangChain and ChromaDB for semantic retrieval  
- Fine-tunes an OpenAI GPT-4o-mini model on the same dataset  
- Evaluates both methods with identical queries for fairness  
- Demonstrates generalization differences when key data is missing  
- Includes detailed outputs and analysis of both model behaviors  

**Tech Stack**  
- **Models:** GPT-4o-mini, Fine-tuned GPT-4o-mini  
- **Frameworks:** LangChain, ChromaDB  
- **Language:** Python  
- **Libraries:** OpenAI, Datasets, JSON  

**Purpose**  
Highlights how RAG and Fine-Tuning differ when trained on the same dataset.  

---

## 10. RAG Linux Command Assistant + Automated Evaluation

**Project Overview**
This project is a follow-up to Project 9, now leveraging an open-source model (`llama3.2` via Ollama) and three Linux command datasets: 

- `harpomaxx/unix-commands`  
- `shikhardadhich/linux_commands`  
- `mecha-org/linux-command-dataset`  

It builds a Retrieval-Augmented Generation (RAG) chatbot for Linux commands using `PromptTemplate`, which yields more accurate, relevant, and elaborated responses than the previous approach. The project also includes an automated evaluation that scores answers using a flexible keyword-matching algorithm with a configurable threshold, allowing developers to quickly assess RAG performance.

**Key Features**
- Retrieval-based Linux command assistant across multiple datasets  
- Structured `PromptTemplate` for precise command + explanation output  
- Automated evaluation module with pass/fail scoring and customizable keywords  
- More accurate, relevant, and elaborated responses than previous implementation  
- Useful for debugging dataset quality, prompt performance, and retriever behavior  

**Tech Stack**
- **Models:** Ollama, `llama3.2`  
- **Libraries:** LangChain, Chroma, Python
- **Environment:** Jupyter Notebook

**Purpose**
Demonstrates a practical, hands-on approach to improving RAG performance for technical tasks—specifically Linux commands—by combining multiple datasets, prompt engineering, and automated evaluation.  
  
---
