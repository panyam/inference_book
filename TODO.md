# Self-Hosted AI Inference: Book Progress Tracker

## Overview

**Book Title:** Self-Hosted AI Inference: A Systems Engineer's Guide
**Publisher:** Apress
**Structure:** 4 Parts, 18 Chapters, 8 Appendices

---

## Status Legend

- ⬜ Not Started
- 🟡 Outline/Stub Created
- 🔵 Writing In Progress
- 🟢 First Draft Complete
- ✅ Final/Reviewed

---

## Front Matter

| Item | Status | Notes |
|------|--------|-------|
| Dedication | 🟡 | Stub created |
| Foreword | 🟡 | Need to identify industry expert to write |
| Preface | 🟡 | Stub with section structure |
| Acknowledgements | 🟡 | Stub created |
| Acronyms | 🟢 | Comprehensive list created |

---

## Part I: Foundations (7B Models, Control Plane v0.1)

**Model Focus:** 7B parameters (Llama 3.2 7B)
**Control Plane:** v0.1 - Single Backend
**Target Personas:** Curious Developer, Tinkerer
**Goal:** Build foundation with consumer hardware, serving 7B models

---

### Chapter 1: Introduction to Self-Hosted Inference

**Status:** 🟡 Outline Created
**File:** `src/chapters/chapter01.tex`

**Learning Objectives:**
- Understand what inference is vs training
- Why self-host instead of using APIs
- Economic and strategic considerations
- Book roadmap and what you'll build

**Topics:**
- Inference fundamentals
- Training vs inference resource requirements
- Use cases for self-hosting (privacy, cost, latency, customization)
- When NOT to self-host
- Overview of the progressive control plane we'll build

**Persona Callouts:**
- Backend Engineer: Cost analysis of API vs self-hosting
- Full-Stack Developer: Freedom to experiment
- Privacy-Conscious: Data sovereignty
- Hobbyist: Learning without ongoing costs

**Hands-On:**
- Run your first model using Ollama (zero config)
- Make your first inference request
- Understand the response structure

**Deliverable:** Working inference request on your machine

#### Task Breakdown

**1.1 Section: The Rise of Self-Hosted AI**
- [ ] Write opening hook about democratization of AI (2-3 paragraphs)
- [ ] List major open model families (Llama, Mistral, Qwen, Phi, Gemma)
- [ ] Explain the gap between API-based and self-hosted AI
- [ ] Add timeline/milestones of open model releases

**1.2 Section: Training vs. Inference**
- [ ] Write subsection: What Happens During Training
- [ ] Write subsection: What Happens During Inference
- [ ] Create resource comparison table (Training vs Inference)
- [ ] Create diagram: Training vs Inference visual comparison

**1.3 Section: Why Self-Host?**
- [ ] Write Cost Control subsection (API pricing, self-hosted cost modeling, break-even)
- [ ] Write Privacy and Data Sovereignty subsection (GDPR, HIPAA, industries)
- [ ] Write Latency and Performance subsection
- [ ] Write Customization and Control subsection
- [ ] Write When NOT to Self-Host subsection

**1.4 Section: What You'll Build in This Book**
- [ ] Write Progressive Journey subsection (4 parts overview)
- [ ] Write Control Plane Evolution subsection (why Go, interface design)
- [ ] Create visual: 4-part journey diagram

**1.5 Section: Hands-On First Inference**
- [ ] Write Installing Ollama guide (macOS, Linux, Windows)
- [ ] Write Running Your First Model guide
- [ ] Write Making API Requests guide (add prose to existing code)

**1.6 Section: Understanding the Response**
- [ ] Explain tokens and tokenization
- [ ] Explain generation parameters
- [ ] Explain metrics (tokens/second, timing)

**1.7 Section: Summary**
- [ ] Write chapter summary paragraph
- [ ] Review Key Takeaways box (exists)
- [ ] Write "Next chapter preview" paragraph

**1.8 References**
- [ ] Add references to references01.tex

---

### Chapter 2: Hardware Fundamentals

**Status:** 🟡 Outline Created
**File:** `src/chapters/chapter02.tex`

**Learning Objectives:**
- Understand VRAM, RAM, CPU vs GPU tradeoffs
- Calculate resource requirements for models
- Choose hardware for your use case
- Owned vs rented infrastructure decisions

**Topics:**
- How transformers use memory during inference
- VRAM calculation formula: `model_size * precision_bytes * overhead`
- CPU inference (when it works, when it doesn't)
- GPU options (NVIDIA, AMD, Apple Silicon)
- Renting GPUs (RunPod, Vast.ai, Lambda Labs comparison)
- Break-even analysis: owned vs rented

**Model Size Context:**
- 7B model requirements: ~14GB VRAM (FP16), ~4GB (Q4)
- What hardware can run it: RTX 3060 12GB, M1 Pro, good CPU
- Cost analysis for 7B deployment

**Persona Callouts:**
- Backend Engineer: Hardware recommendations for production
- Hobbyist: Budget-friendly options
- Infrastructure Engineer: Multi-GPU considerations
- Startup Founder: Buy vs rent decision framework

**Hands-On:**
- Calculate VRAM needs for different models
- Benchmark your hardware
- Compare rental provider pricing

**Deliverable:** Hardware recommendation matrix, cost calculator spreadsheet

#### Task Breakdown

**2.1 Section: How Transformers Use Memory**
- [ ] Write Model Weights subsection (formula, worked examples)
- [ ] Write KV Cache subsection (what it stores, why it grows)
- [ ] Write Activation/Working Memory subsection
- [ ] Create diagram: Memory layout during inference

**2.2 Section: VRAM Calculation**
- [ ] Write Precision and Memory subsection (FP32, FP16, BF16, INT8, INT4 table)
- [ ] Write Worked Examples subsection (7B, 30B, 70B, 400B)
- [ ] Create VRAM calculator spreadsheet (deliverable)

**2.3 Section: CPU vs GPU Inference**
- [ ] Write CPU Inference subsection
- [ ] Write GPU Inference subsection
- [ ] Write Hybrid Approaches subsection (layer offloading)

**2.4 Section: GPU Hardware Comparison**
- [ ] Write NVIDIA Consumer GPUs subsection (RTX 30xx, 40xx)
- [ ] Write NVIDIA Professional GPUs subsection (A100, H100)
- [ ] Write AMD GPUs subsection (ROCm status, MI series)
- [ ] Write Apple Silicon subsection (M1/M2/M3/M4)
- [ ] Create Hardware Recommendation Matrix table

**2.5 Section: Owned vs Rented Infrastructure**
- [ ] Write Buying Hardware subsection (CapEx, ongoing costs)
- [ ] Write Renting Cloud GPUs subsection (RunPod, Vast.ai, Lambda Labs, pricing table)
- [ ] Write Break-Even Analysis subsection (formula, worked examples)
- [ ] Write Hybrid Strategies subsection
- [ ] Create cost comparison spreadsheet (deliverable)

**2.6 Section: Power and Cooling**
- [ ] Write power consumption by GPU table
- [ ] Cooling requirements overview
- [ ] Home lab considerations
- [ ] Data center placement basics

**2.7 Section: Summary**
- [ ] Write chapter summary
- [ ] Review Key Takeaways box
- [ ] Preview next chapter

**2.8 References**
- [ ] Add references to references02.tex

---

### Chapter 3: Model Selection and Formats

**Status:** 🟡 Outline Created
**File:** `src/chapters/chapter03.tex`

**Learning Objectives:**
- Navigate HuggingFace and model repositories
- Understand model formats (GGUF, safetensors, GPTQ, etc.)
- Choose models for your hardware
- Convert between formats

**Topics:**
- Model architectures overview (decoder-only, encoder-decoder)
- The HuggingFace ecosystem
- Model formats explained: safetensors, GGUF, GPTQ, AWQ, EXL2
- Quantization overview (deep-dive deferred to Chapter 10)
- Model naming conventions and what they mean
- Licensing considerations (Apache 2.0, Llama license, etc.)

**Model Size Context:**
- Finding 7B models (Llama, Mistral, Phi, Qwen families)
- Which formats work best for consumer hardware
- Quality vs size tradeoffs

**Persona Callouts:**
- Backend Engineer: Production-ready model selection
- Privacy-Conscious: Licensing and data usage policies
- Edge Engineer: Smallest viable models

**Hands-On:**
- Download models from HuggingFace
- Convert safetensors to GGUF
- Compare same model in different formats
- Benchmark quality and performance

**Deliverable:** Model comparison matrix, format conversion scripts

#### Task Breakdown

**3.0 Section: Model Architectures Overview**
- [ ] Write decoder-only vs encoder-decoder explanation
- [ ] Explain why decoder-only dominates for LLMs
- [ ] Brief mention of encoder models (BERT-style) for embeddings

**3.1 Section: Understanding Model Formats**
- [ ] Write SafeTensors subsection
- [ ] Write GGUF subsection (GGML history, single-file advantage)
- [ ] Write ONNX subsection
- [ ] Write EXL2 subsection (ExLlama format)
- [ ] Create Format Comparison table

**3.2 Section: Quantization Fundamentals**
- [ ] Write What is Quantization subsection
- [ ] Write Quantization Levels subsection (Q8, Q6, Q5, Q4, Q3/Q2)
- [ ] Create quantization comparison chart (quality vs memory)

**3.3 Section: Quantization Methods**
- [ ] Write GGUF Quantization subsection (Q4_K_M explained)
- [ ] Write GPTQ subsection
- [ ] Write AWQ subsection
- [ ] Write BitsAndBytes subsection
- [ ] Create Method Comparison table

**3.4 Section: Choosing Models**
- [ ] Write Model Families Overview (Llama, Mistral, Qwen, Phi, Gemma)
- [ ] Write Size vs Quality Trade-offs
- [ ] Write Model Naming Conventions guide (what 7B-Q4_K_M means)
- [ ] Write Licensing Considerations (Apache 2.0, Llama license, commercial use)
- [ ] Create Decision Matrix table

**3.5 Section: Where to Find Models**
- [ ] Write Hugging Face Hub guide
- [ ] Write Ollama Library guide
- [ ] Write Model Evaluation guide

**3.6 Section: Building a Model Registry**
- [ ] Write Why a Registry subsection
- [ ] Write Registry Data Model (add prose to existing code)

**3.7 Section: Summary**
- [ ] Write chapter summary
- [ ] Review Key Takeaways
- [ ] Preview next chapter

**3.8 References**
- [ ] Add references to references03.tex

---

### Chapter 4: Inference Engines

**Status:** 🟡 Outline Created
**File:** `src/chapters/chapter04.tex`

**Learning Objectives:**
- Understand different inference engines
- Choose the right engine for your use case
- Set up and configure engines
- Understand performance characteristics

**Topics:**
- **llama.cpp:** CPU/GPU hybrid, GGUF format, easy to use
- **Ollama:** Simplified deployment, good for getting started
- **vLLM:** Production-grade, high throughput, continuous batching
- **ONNX Runtime:** Cross-platform, good for edge
- **TensorRT:** NVIDIA-optimized, maximum performance
- Engine comparison matrix

**Model Size Context:**
- 7B performance across engines
- Resource usage comparison
- Throughput benchmarks

**Persona Callouts:**
- Backend Engineer: vLLM for production
- Full-Stack Developer: Ollama for simplicity
- Edge Engineer: ONNX Runtime
- Infrastructure Engineer: Performance tuning

**Hands-On:**
- Set up each major engine
- Run same model on different engines
- Benchmark tokens/second
- Compare resource usage

**Deliverable:** Performance benchmark results, engine selection guide

#### Task Breakdown

**4.1 Section: The Role of Inference Engines**
- [ ] Write overview (load models, process requests, batching, streaming)

**4.2 Section: Ollama**
- [ ] Write Architecture subsection (llama.cpp based, modelfile)
- [ ] Write Installation and Configuration
- [ ] Write API Reference (expand existing code)
- [ ] Write When to Use Ollama

**4.3 Section: llama.cpp**
- [ ] Write Architecture subsection (C/C++, multi-backend)
- [ ] Write Building and Installation (CPU, CUDA, Metal)
- [ ] Write Server Mode (add prose to existing code)
- [ ] Write Advanced Configuration (layer offload, threads)
- [ ] Write When to Use llama.cpp

**4.4 Section: vLLM**
- [ ] Write Paged Attention subsection
- [ ] Write Continuous Batching subsection
- [ ] Write Installation and Setup (add prose to existing code)
- [ ] Write API and OpenAI Compatibility
- [ ] Write Performance Tuning
- [ ] Write When to Use vLLM

**4.5 Section: Other Engines**
- [ ] Write TensorRT-LLM brief overview
- [ ] Write TGI brief overview
- [ ] Write SGLang brief overview

**4.6 Section: Engine Comparison**
- [ ] Create comprehensive Comparison Matrix table
- [ ] Write Decision Framework guide

**4.7 Section: Hands-On**
- [ ] Write exercise: Run all three engines
- [ ] Create benchmark comparison script

**4.8 Section: Summary**
- [ ] Write chapter summary
- [ ] Review Key Takeaways
- [ ] Preview next chapter

**4.9 References**
- [ ] Add references to references04.tex

---

### Chapter 5: Building Control Plane v0.1

**Status:** 🟡 Outline Created
**File:** `src/chapters/chapter05.tex`

**Learning Objectives:**
- Design a simple API gateway for inference
- Implement health checks and metrics
- Structure Go code for growth
- Deploy your first production-ready service

**Topics:**
- Architecture overview (HTTP server, backend abstraction, metrics, health)
- Go implementation (interface design, handlers, context propagation, error handling, graceful shutdown)
- Observability from day one (structured logging, Prometheus metrics, request tracing)
- Deployment (Docker, docker-compose, configuration)
- Queue depth hints (foundation for priority scheduling in Chapter 12)

**Control Plane Interfaces:**
- `InferenceBackend` - abstract inference engine
- `RequestRouter` - route to backend
- `MetricsCollector` - observability

**Persona Callouts:**
- Backend Engineer: Production patterns from the start
- Infrastructure Engineer: Observability best practices
- Full-Stack Developer: API design principles

**Hands-On:**
- Implement control plane v0.1
- Connect to Ollama/llama.cpp backend
- Set up Prometheus + Grafana
- Deploy with docker-compose
- Load test with k6 (measure requests/sec, latency percentiles)

**Deliverable:** Working control plane codebase, Docker deployment, Grafana dashboard, k6 load test results

#### Task Breakdown

**5.1 Section: Control Plane Architecture**
- [ ] Write What is a Control Plane subsection
- [ ] Write v0.1 Capabilities subsection
- [ ] Create System Architecture Diagram

**5.2 Section: Project Setup**
- [ ] Write Go Module Initialization (add context to existing code)
- [ ] Write Dependencies explanation (chi, prometheus, slog)

**5.3 Section: Core Interfaces**
- [ ] Write Backend Interface explanation (add prose to existing code)
- [ ] Write Health Interface explanation (add prose to existing code)
- [ ] Explain design decisions

**5.4 Section: Implementing Ollama Backend**
- [ ] Add prose explaining the implementation (code exists)
- [ ] Explain error handling approach
- [ ] Explain response conversion
- [ ] Explain context propagation patterns
- [ ] Explain graceful shutdown implementation

**5.5 Section: API Server**
- [ ] Write Router Setup (endpoints explanation)
- [ ] Write Request Handling middleware explanation
- [ ] Write request tracing setup (trace IDs, correlation)

**5.6 Section: Metrics with Prometheus**
- [ ] Write Key Metrics to Track
- [ ] Add prose explaining metrics code (exists)

**5.7 Section: Docker Deployment**
- [ ] Write Dockerfile explanation (multi-stage build)
- [ ] Write Docker Compose Stack prose (code exists)

**5.8 Section: Grafana Dashboard**
- [ ] Write dashboard creation guide
- [ ] Create Grafana dashboard JSON (deliverable)

**5.9 Section: Testing the Control Plane**
- [ ] Write integration testing guide
- [ ] Write k6 load testing setup and script
- [ ] Document expected performance baselines
- [ ] Add queue depth hints (foundation for Chapter 12 priority scheduling)

**5.10 Section: Configuration Management**
- [ ] Add prose explaining config code (exists)

**5.11 Section: Summary**
- [ ] Write chapter summary
- [ ] Review Key Takeaways
- [ ] Preview Part II

**5.12 References**
- [ ] Add references to references05.tex

**End of Part I (without Chapter 5.5):** Working inference service with observability, serving a 7B model

---

### Chapter 5.5: Browser AI & Hybrid Architecture (Optional)

**Status:** 🟡 Outline Created
**File:** `src/chapters/chapter05_5.tex`

> **Note:** This chapter is optional. It can be skipped if focusing on server-side inference only.

**Learning Objectives:**
- Understand browser-based inference with WebGPU/WebGL
- Deploy models that run in users' browsers
- Build hybrid systems (browser + server)
- Integrate browser inference with control plane
- Know when browser inference makes economic sense

**Topics:**
- Why Browser Inference (zero cost, latency, privacy, tradeoffs)
- Browser Technologies (WebGPU, WebGL 2.0, WASM, WebNN)
- Frameworks (Transformers.js, WebLLM, ONNX Runtime Web)
- Model Selection for Browser (size constraints, recommended models)
- Capability Detection
- Hybrid Client Architecture
- Control Plane Integration
- Performance & Economics

**Model Size Context:**
- Browser models: 1-7B parameters only
- Server still needed for complex tasks
- Hybrid approach optimizes for both

**Persona Callouts:**
- Web/Mobile Developer: Client-side AI without servers
- Pragmatic Engineer: 60-80% cost reduction
- Privacy-Conscious Founder: Data never leaves device
- Hobbyist: Run AI without cloud costs

**Hands-On:**
- Detect device capabilities
- Implement hybrid inference client
- Add routing to control plane
- Deploy two models (browser: Phi-3-mini, server: Llama 33B)
- Measure cost savings

**Deliverable:** Browser + server integration, capability detection, browser model registry, demo web app

#### Task Breakdown

**5.5.1 Why Browser Inference**
- [ ] Write zero-cost deployment explanation
- [ ] Write ultra-low latency benefits
- [ ] Write privacy advantages
- [ ] Write trade-offs vs server

**5.5.2 Browser Technologies**
- [ ] Write WebGPU section
- [ ] Write WebGL 2.0 fallback
- [ ] Write WebAssembly for CPU
- [ ] Write WebNN (future)

**5.5.3 Frameworks**
- [ ] Write Transformers.js overview
- [ ] Write WebLLM (recommended) guide
- [ ] Write ONNX Runtime Web
- [ ] Create framework comparison

**5.5.4 Model Selection for Browser**
- [ ] Write size constraints by device
- [ ] Write recommended models
- [ ] Write format requirements

**5.5.5 Capability Detection**
- [ ] Write detection code and explanation
- [ ] Write GPU tier classification
- [ ] Write browser compatibility

**5.5.6 Hybrid Client Architecture**
- [ ] Write HybridInferenceClient design
- [ ] Write routing decisions
- [ ] Write fallback mechanisms

**5.5.7 Control Plane Integration**
- [ ] Write HybridRouter interface
- [ ] Write BrowserModelRegistry
- [ ] Write API endpoints
- [ ] Write browser-specific analytics tracking
- [ ] Write cost tracking for browser vs server requests

**5.5.8 Performance & Economics**
- [ ] Write performance expectations by device
- [ ] Write cost analysis (80% browser = 78% cost reduction)
- [ ] Write break-even analysis
- [ ] Create cost comparison methodology

**5.5.9 Best Practices**
- [ ] Write progressive enhancement patterns
- [ ] Write battery awareness guidelines
- [ ] Write browser caching strategies (model weights, KV cache)
- [ ] Write user control principles (let users choose browser vs server)
- [ ] Write error handling and fallback patterns

**5.5.10 Limitations**
- [ ] Document model size constraints
- [ ] Document performance variability across devices
- [ ] Document battery drain considerations
- [ ] Document browser compatibility matrix
- [ ] Document security considerations

**5.5.11 Summary**
- [ ] Write chapter summary
- [ ] Recap when to use browser inference

**5.5.12 References**
- [ ] Add references to references05_5.tex

**End of Part I (with Chapter 5.5):** Complete inference system supporting both server and browser deployment

---

## Part II: Production Deployment (30B Models, Control Plane v0.2)

**Model Focus:** 30B parameters (Llama 3.1 33B)
**Control Plane:** v0.2 - Production Grade
**Target Personas:** Pragmatic Engineer, Infrastructure Engineer
**Goal:** Add production features - auth, rate limiting, caching, queuing

| Chapter | Title | Status | Priority | Notes |
|---------|-------|--------|----------|-------|
| 6 | Authentication and API Keys | 🟡 | HIGH | JWT, API key management |
| 7 | Rate Limiting and Quotas | 🟡 | HIGH | Token bucket, per-tier limits |
| 8 | Response Caching | 🟡 | MEDIUM | Exact + semantic caching |
| 9 | Request Queue and Priority | 🟡 | MEDIUM | Priority queue, load shedding |
| 10 | 30B Model Optimization | 🟡 | HIGH | KV cache, vLLM tuning |

### Part II TODO Items
- [ ] Design complete auth flow diagrams
- [ ] Implement rate limiting algorithms with benchmarks
- [ ] Research semantic caching approaches
- [ ] Create queue simulation for testing
- [ ] Benchmark 30B model configurations

---

## Part III: Multi-Tenant Platform (70B Models, Control Plane v0.3)

**Model Focus:** 70B parameters
**Control Plane:** v0.3 - Multi-Tenant Platform
**Target Personas:** Infrastructure Engineer, Startup Founder
**Goal:** Multi-tenant with billing, distributed inference

| Chapter | Title | Status | Priority | Notes |
|---------|-------|--------|----------|-------|
| 11 | Multi-Tenant Architecture | 🟡 | HIGH | Isolation models, tenant data |
| 12 | Usage Tracking and Billing | 🟡 | MEDIUM | Metering, Stripe integration |
| 13 | Multi-GPU and Distributed Inference | 🟡 | HIGH | Tensor/pipeline parallelism |
| 14 | Model Routing and Selection | 🟡 | MEDIUM | Cost-aware routing, A/B testing |
| 15 | 70B Deployment | 🟡 | HIGH | Complete deployment guide |

### Part III TODO Items
- [ ] Design tenant isolation architecture
- [ ] Create billing data model and Stripe integration guide
- [ ] Document NVLink requirements for multi-GPU
- [ ] Build model routing decision framework
- [ ] Create 70B deployment checklist

---

## Part IV: The Inference Lab (400B Models, Control Plane v1.0)

**Model Focus:** 400B parameters (Qwen 2.5 Coder 400B)
**Control Plane:** v1.0 - Complete Infrastructure
**Target Personas:** All personas - capstone project
**Goal:** Enterprise-scale with 400B models, CodeLab capstone

| Chapter | Title | Status | Priority | Notes |
|---------|-------|--------|----------|-------|
| 16 | 400B Deployment and H100 Optimization | 🟡 | HIGH | H100 deep dive, economics |
| 17 | Building CodeLab | 🟡 | HIGH | Capstone: AI coding assistant |
| 18 | Production Operations | 🟡 | HIGH | Final chapter, v1.0 complete |

### Part IV TODO Items
- [ ] Research H100 vs A100 benchmarks
- [ ] Design CodeLab architecture
- [ ] Create VS Code extension skeleton
- [ ] Write operational runbook template
- [ ] Finalize Control Plane v1.0 feature list

---

## Appendices

| Appendix | Title | Status | Priority | Notes |
|----------|-------|--------|----------|-------|
| A | Complete Control Plane Code | 🟡 | LOW | Reference implementation |
| B | Hardware Specifications | 🟡 | MEDIUM | GPU spec sheets |
| C | Model Catalog | 🟡 | MEDIUM | Model recommendations by size |
| D | API Reference | 🟡 | MEDIUM | OpenAPI spec |
| E | Deployment Templates | 🟡 | MEDIUM | Docker, K8s, Terraform |
| F | Troubleshooting Guide | 🟡 | LOW | Common issues |
| G | TPU Inference | 🟡 | LOW | Google Cloud TPU guide |
| H | Cost Calculators | 🟡 | MEDIUM | Spreadsheet templates |

---

## Back Matter

| Item | Status | Notes |
|------|--------|-------|
| Glossary | 🟢 | Comprehensive, ~30 terms |
| Solutions | 🟡 | Partial solutions, rest in repo |
| Index | ⬜ | Auto-generated at build |

---

## Part I Deliverables Checklist

- [ ] VRAM calculator spreadsheet (Chapter 2)
- [ ] Cost comparison spreadsheet (Chapter 2)
- [ ] Model comparison matrix (Chapter 3)
- [ ] Format conversion scripts (Chapter 3)
- [ ] Engine benchmark scripts (Chapter 4)
- [ ] Control plane v0.1 Go code (Chapter 5)
- [ ] Docker deployment files (Chapter 5)
- [ ] Grafana dashboard JSON (Chapter 5)
- [ ] Browser hybrid client (Chapter 5.5, optional)
- [ ] Demo web application (Chapter 5.5, optional)

---

## Part I Figures and Diagrams

- [ ] Training vs Inference comparison diagram (Ch 1)
- [ ] 4-part journey diagram (Ch 1)
- [ ] Memory layout during inference (Ch 2)
- [ ] GPU comparison chart (Ch 2)
- [ ] Quantization quality vs memory chart (Ch 3)
- [ ] Engine comparison table (Ch 4)
- [ ] Control Plane v0.1 architecture diagram (Ch 5)
- [ ] Hybrid flow diagram (Ch 5.5, optional)

---

## Code Repository

| Item | Status | Notes |
|------|--------|-------|
| Repository setup | ⬜ | GitHub repo needed |
| v0.1 implementation | ⬜ | Chapters 1-5 |
| v0.2 implementation | ⬜ | Chapters 6-10 |
| v0.3 implementation | ⬜ | Chapters 11-15 |
| v1.0 implementation | ⬜ | Chapters 16-18 |
| CodeLab project | ⬜ | Chapter 17 capstone |
| Docker templates | ⬜ | Appendix E |
| Test suite | ⬜ | All chapters |

---

## Decisions Made

- Go for control plane (not Python) - performance + deployment
- Progressive complexity (7B → 30B → 70B → 400B)
- OpenAI-compatible API throughout
- vLLM as primary production engine
- Chapter 5.5 (Browser AI) is optional

---

## Open Questions

1. What code models to feature in CodeLab? (DeepSeek Coder vs CodeLlama)
2. Include Kubernetes deployment or Docker-only?
3. Target H100 or also cover MI300X (AMD)?

---

## Quick Commands

```bash
# Build full book
./build.sh book

# Build single chapter for review
./build.sh chapter01

# Build all chapters individually
./build.sh all-chapters

# Clean build artifacts
./build.sh clean
```

---

## Writing Order (Recommended)

1. **Chapter 1** - Sets the stage, motivates the reader
2. **Chapter 2** - Hardware decisions needed before model selection
3. **Chapter 3** - Model understanding before engine selection
4. **Chapter 4** - Engine understanding before building control plane
5. **Chapter 5** - Capstone of Part I, brings it all together
6. **Chapter 5.5** - Optional, can be deferred or skipped

---

*Last Updated: December 2024*
