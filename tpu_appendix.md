# Appendix G: TPU Inference on Google Cloud

## Overview

This appendix covers inference on Google Cloud TPUs (Tensor Processing Units), an alternative to GPU-based inference covered in the main chapters.

**When to read this:**
- You're deploying on Google Cloud Platform
- Considering TPUs for cost optimization at scale
- Working with JAX/Flax models
- Need very high throughput (100+ concurrent requests)

**When to skip this:**
- Need on-premise hosting (TPUs are GCP-only)
- Using PyTorch models (conversion overhead)
- Small scale (<50 requests/sec)
- Prefer ecosystem maturity (GPUs have more tooling)

---

## G.1: TPU Architecture Fundamentals

### What Are TPUs?

**Tensor Processing Unit (TPU):**
- Custom ASIC designed by Google for ML workloads
- Optimized for matrix multiplication
- High bandwidth memory (HBM)
- Systolic array architecture
- Integrated into Google Cloud infrastructure

**Key Differences from GPUs:**

| Aspect | GPU | TPU |
|--------|-----|-----|
| Architecture | General purpose parallel | ML-specific ASIC |
| Memory | GDDR6/HBM2 | HBM2/HBM2e |
| Programming | CUDA, OpenCL | XLA, JAX |
| Flexibility | High | Medium |
| ML Performance | Excellent | Excellent |
| Cost (GCP) | $$$ | $$ |
| Availability | Buy or rent | Rent only (GCP) |

### TPU Generations

**TPU v4 (Current production):**
- 8 chips per pod
- 275 TFLOPS (BF16)
- 32GB HBM per chip
- Best for: Production inference

**TPU v5e (Cost-optimized):**
- 4 or 8 chips per pod
- 197 TFLOPS (BF16)
- 16GB HBM per chip
- Best for: Budget-conscious deployments

**TPU v5p (Latest, highest performance):**
- Up to 256 chips per pod
- 459 TFLOPS (BF16)
- 95GB HBM per chip
- Best for: Very large models, training

### Pod Architecture

TPUs are organized in "pods":
```
TPU Pod Topology:
├── v4-8: 8 chips, single-host
├── v4-16: 16 chips, 2 hosts
├── v4-32: 32 chips, 4 hosts
└── v4-128+: Multi-rack configurations

Interconnect:
└── Inter-Chip Interconnect (ICI)
    ├── High bandwidth (100+ Gbps per link)
    └── Low latency (<10 μs)
```

---

## G.2: When to Use TPUs for Inference

### Economic Analysis

**Cost Comparison (GCP, as of 2024):**

```
70B Model Inference:

GPU Option (4x A100 40GB):
- On-demand: ~$12/hour
- 1-year commit: ~$8/hour
- 3-year commit: ~$6/hour

TPU Option (v4-8 pod):
- On-demand: ~$4.40/hour
- 1-year commit: ~$3.30/hour
- 3-year commit: ~$2.60/hour

TPU Option (v5e-8 pod):
- On-demand: ~$2.90/hour
- 1-year commit: ~$2.20/hour

Cost Advantage: 50-70% cheaper for sustained workloads
```

**Break-Even Analysis:**

```
When TPUs Make Economic Sense:

Sustained Workload (24/7):
├── v4-8 TPU: $3,168/month (1-year commit)
├── 4x A100: $5,760/month (1-year commit)
└── Savings: $2,592/month (45%)

Burst Workload (8 hours/day):
├── v4-8 TPU: $1,056/month
├── 4x A100: $1,920/month
└── Savings: $864/month (45%)

Small Scale (<100 requests/hour):
├── May not justify setup complexity
└── GPU ecosystem simplicity wins
```

### Technical Fit

**Good Fit for TPUs:**

1. **Already on GCP**
   - Infrastructure investment exists
   - Familiar with GCP tools
   - Integration with other GCP services

2. **JAX/Flax-Native Models**
   - Gemma (Google's model)
   - PaLM-based models
   - T5 family
   - Custom JAX models

3. **High Throughput Required**
   - >100 concurrent requests
   - Batch processing workloads
   - Background inference jobs

4. **Cost-Sensitive at Scale**
   - Large deployment (24/7)
   - Predictable workload
   - Multi-year horizon

**Bad Fit for TPUs:**

1. **PyTorch Models**
   - Requires conversion to JAX (complex)
   - Ecosystem mismatch
   - Ongoing maintenance burden

2. **On-Premise Requirement**
   - TPUs only available on GCP
   - (TPU dev boards exist but limited)

3. **Small Scale**
   - Setup complexity not justified
   - GPU ecosystem simpler

4. **Rapid Experimentation**
   - GPU tooling more mature
   - Faster iteration cycle
   - Better debugging tools

---

## G.3: Inference Frameworks for TPU

### JAX + Pallas (Low-Level)

**What it is:**
- JAX: NumPy + autograd + XLA
- Pallas: Kernel language for TPU
- Low-level control
- High performance

**Example:**
```python
import jax
import jax.numpy as jnp
from jax import jit

# Define inference function
@jit
def generate_token(params, input_ids):
    # Forward pass
    logits = model_forward(params, input_ids)
    return jnp.argmax(logits, axis=-1)

# Compile for TPU
compiled_fn = jax.jit(generate_token, backend='tpu')

# Run inference
output = compiled_fn(params, input_ids)
```

**Pros:**
- Maximum control
- Best performance
- Fine-grained optimization

**Cons:**
- Steep learning curve
- Manual model implementation
- Low-level debugging

### MaxText (Production LLMs)

**What it is:**
- High-performance LLM framework
- Built on JAX
- Supports large models
- Google-maintained

**Example:**
```python
from maxtext import inference

# Load model
model = inference.load_model(
    model_name="llama-70b",
    checkpoint_path="gs://bucket/checkpoints",
    mesh_shape=(4, 2)  # 8 TPU chips
)

# Generate
output = model.generate(
    prompt="Explain quantum computing",
    max_length=200,
    temperature=0.7
)
```

**Pros:**
- Production-ready
- Good performance
- Multi-host support
- Maintained by Google

**Cons:**
- Limited model support
- JAX/Flax models only
- Less flexible than custom code

### JetStream (New, Recommended)

**What it is:**
- Optimized inference engine
- Built for Gemma and PaLM
- Production-grade
- Continuous batching support

**Example:**
```python
from jetstream.engine import create_engine

# Create inference engine
engine = create_engine(
    model_name="gemma-7b",
    tokenizer="gemma-tokenizer",
    devices=8,  # TPU chips
    batch_size=32,
    max_length=2048
)

# Serve requests
async def handle_request(prompt):
    return await engine.generate(
        prompt=prompt,
        max_tokens=200,
        temperature=0.7
    )
```

**Pros:**
- Modern architecture
- Continuous batching
- Good documentation
- Active development

**Cons:**
- New (less battle-tested)
- Limited model support
- Requires JAX models

### TensorFlow Serving (Legacy)

**What it is:**
- Traditional TF inference
- Mature and stable
- Good for older models

**When to use:**
- Legacy TensorFlow models
- Existing TF infrastructure
- Non-transformer models

**Status:** Not recommended for new LLM deployments

---

## G.4: Model Preparation for TPU

### Converting PyTorch to JAX

**Challenge:** Most models are PyTorch, TPUs prefer JAX

**Option 1: Manual Conversion**
```python
# PyTorch model
import torch
import torch.nn as nn

class PyTorchModel(nn.Module):
    def forward(self, x):
        return self.layers(x)

# JAX equivalent
import jax.numpy as jnp
from flax import linen as nn

class JAXModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        return self.layers(x)

# Convert weights
def convert_weights(pt_state_dict):
    # Manual mapping of parameters
    jax_params = {}
    for key, value in pt_state_dict.items():
        # Convert torch tensor to jax array
        jax_params[key] = jnp.array(value.cpu().numpy())
    return jax_params
```

**Pros:** Full control
**Cons:** Extremely time-consuming, error-prone

**Option 2: Use Pre-Converted Models**
```python
# Gemma models (JAX-native)
from jetstream.engine import create_engine

engine = create_engine(
    model_name="gemma-7b",
    # Already in JAX format
)
```

**Pros:** Works immediately
**Cons:** Limited model selection

**Option 3: ONNX as Intermediate**
```python
# PyTorch → ONNX → JAX (experimental)
# Not production-ready for large models
```

### Quantization for TPU

**Supported Quantization:**

1. **BFloat16 (Native)**
   ```python
   # JAX models use BF16 by default on TPU
   @jax.jit
   def inference(params, x):
       # Automatically uses BF16
       return model(params, x)
   ```

2. **INT8 Quantization**
   ```python
   from jax.experimental import jax2tf
   from tensorflow import quantization
   
   # Quantize for TPU
   quantized_model = quantization.quantize_model(
       model,
       quantization_mode='int8_per_channel'
   )
   ```

3. **Custom Quantization**
   ```python
   # Manual quantization with JAX
   def quantize_weights(weights, bits=8):
       scale = jnp.max(jnp.abs(weights))
       quantized = jnp.round(weights / scale * (2**(bits-1) - 1))
       return quantized.astype(jnp.int8), scale
   ```

**Quantization Impact:**
```
70B Model on v4-8 TPU:

BF16 (Native):
├── Memory: 140GB (fits in 8x32GB = 256GB)
├── Speed: 45 tokens/sec
└── Quality: Full precision

INT8:
├── Memory: 70GB (more headroom)
├── Speed: 60 tokens/sec
└── Quality: Minimal degradation (~1%)
```

### XLA Compilation

**What is XLA?**
- Accelerated Linear Algebra
- Compiles JAX to TPU instructions
- Automatic optimization
- Required for TPU execution

**Compilation Process:**
```python
import jax

# Define function
def inference(params, input_ids):
    return model(params, input_ids)

# Compile for TPU
compiled_inference = jax.jit(
    inference,
    backend='tpu',
    static_argnums=(0,)  # params are static
)

# First call: compiles (slow, ~1-2 minutes)
output = compiled_inference(params, input_ids)

# Subsequent calls: fast
output = compiled_inference(params, different_input_ids)
```

**Compilation Tips:**
1. **Compile once** - Avoid recompilation
2. **Static shapes** - XLA loves static shapes
3. **Batch consistently** - Same batch size per compilation
4. **Profile** - Use `jax.profiler` to find bottlenecks

---

## G.5: Deployment on GCP

### Setting Up TPU VM

**Step 1: Create TPU VM**
```bash
gcloud compute tpus tpu-vm create inference-tpu \
  --zone=us-central2-b \
  --accelerator-type=v4-8 \
  --version=tpu-ubuntu2204-base \
  --metadata=startup-script='#!/bin/bash
    # Install dependencies
    pip3 install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    pip3 install jetstream-pt
  '
```

**Step 2: SSH and Setup**
```bash
# SSH into TPU VM
gcloud compute tpus tpu-vm ssh inference-tpu \
  --zone=us-central2-b

# Verify TPUs
python3 -c "import jax; print(jax.devices())"
# Output: [TpuDevice(id=0), TpuDevice(id=1), ..., TpuDevice(id=7)]
```

**Step 3: Deploy Model**
```python
# deploy.py
from jetstream.engine import create_engine
import asyncio

async def main():
    # Create engine
    engine = create_engine(
        model_name="gemma-7b",
        devices=8,
        batch_size=32,
        max_length=2048
    )
    
    print("Engine ready")
    
    # Serve requests
    # (integrate with your control plane)

if __name__ == "__main__":
    asyncio.run(main())
```

### Multi-Host Setup (Large Models)

For models >70B, use multi-host TPU pods:

```bash
# Create v4-32 pod (32 chips, 4 hosts)
gcloud compute tpus tpu-vm create inference-tpu-large \
  --zone=us-central2-b \
  --accelerator-type=v4-32 \
  --version=tpu-ubuntu2204-base
```

**Multi-host coordination:**
```python
import jax
from jax.experimental import mesh_utils

# Define mesh for 4 hosts, 8 TPUs each
devices = mesh_utils.create_device_mesh((4, 8))

# Shard model across hosts
from jax.sharding import Mesh, PartitionSpec

mesh = Mesh(devices, axis_names=('data', 'model'))

# Shard parameters
def shard_params(params):
    return jax.tree_map(
        lambda x: jax.device_put(
            x,
            jax.sharding.NamedSharding(
                mesh,
                PartitionSpec('model',)
            )
        ),
        params
    )

sharded_params = shard_params(params)
```

### Load Balancing

**Architecture:**
```
Internet
    ↓
GCP Load Balancer
    ↓
┌─────────┬─────────┬─────────┐
│ TPU VM 1│ TPU VM 2│ TPU VM 3│
│ (v4-8)  │ (v4-8)  │ (v4-8)  │
└─────────┴─────────┴─────────┘
```

**Setup:**
```bash
# Create instance group
gcloud compute instance-groups managed create tpu-inference-group \
  --base-instance-name=tpu-inference \
  --size=3 \
  --zone=us-central2-b

# Create load balancer
gcloud compute backend-services create tpu-inference-backend \
  --protocol=HTTP \
  --port-name=http \
  --health-checks=http-health-check \
  --global

# Add TPU instances to backend
gcloud compute backend-services add-backend tpu-inference-backend \
  --instance-group=tpu-inference-group \
  --instance-group-zone=us-central2-b \
  --global
```

---

## G.6: Control Plane Integration

### TPU Backend Implementation

```go
// internal/backend/tpu_backend.go

type TPUBackend struct {
    endpoint    string // TPU VM endpoint
    projectID   string
    zone        string
    podSize     int
    client      *http.Client
}

func NewTPUBackend(config TPUConfig) (*TPUBackend, error) {
    return &TPUBackend{
        endpoint: config.Endpoint,
        projectID: config.ProjectID,
        zone:      config.Zone,
        podSize:   config.PodSize,
        client:    &http.Client{Timeout: 30 * time.Second},
    }, nil
}

func (b *TPUBackend) Generate(
    ctx context.Context,
    req GenerateRequest,
) (GenerateResponse, error) {
    // Forward to TPU VM's inference server
    payload, err := json.Marshal(map[string]interface{}{
        "prompt":      req.Prompt,
        "max_tokens":  req.MaxTokens,
        "temperature": req.Temperature,
    })
    if err != nil {
        return GenerateResponse{}, err
    }
    
    httpReq, err := http.NewRequestWithContext(
        ctx,
        "POST",
        b.endpoint+"/generate",
        bytes.NewReader(payload),
    )
    if err != nil {
        return GenerateResponse{}, err
    }
    
    httpReq.Header.Set("Content-Type", "application/json")
    
    resp, err := b.client.Do(httpReq)
    if err != nil {
        return GenerateResponse{}, err
    }
    defer resp.Body.Close()
    
    var result GenerateResponse
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return GenerateResponse{}, err
    }
    
    return result, nil
}

func (b *TPUBackend) Health(ctx context.Context) HealthStatus {
    // Check TPU VM health
    resp, err := b.client.Get(b.endpoint + "/health")
    if err != nil {
        return HealthStatus{
            Healthy: false,
            Message: err.Error(),
        }
    }
    defer resp.Body.Close()
    
    return HealthStatus{
        Healthy: resp.StatusCode == 200,
        Message: "TPU backend healthy",
    }
}

func (b *TPUBackend) Metrics(ctx context.Context) BackendMetrics {
    // Fetch metrics from TPU VM
    // Implementation similar to Health()
    return BackendMetrics{
        // ... metrics ...
    }
}
```

### Configuration

```yaml
# config.yaml
backends:
  - name: tpu-gemma-7b
    type: tpu
    endpoint: http://tpu-vm-internal-ip:8080
    project_id: my-gcp-project
    zone: us-central2-b
    pod_size: 8
    timeout: 30s
```

### Resource Tracking

```go
// TPU-specific resource tracking
type TPUResources struct {
    Chips       int     // Number of TPU chips
    MemoryPerChipGB float64
    Utilization float64 // 0.0 - 1.0
}

func (b *TPUBackend) GetResources() TPUResources {
    return TPUResources{
        Chips:       b.podSize,
        MemoryPerChipGB: 32, // v4 = 32GB per chip
        Utilization: b.getCurrentUtilization(),
    }
}
```

---

## G.7: Performance Optimization

### Batching for TPUs

TPUs excel at large batch sizes:

```python
# Optimal batch sizes
OPTIMAL_BATCH_SIZES = {
    'v4-8': 32,   # 32 concurrent requests
    'v5e-8': 24,  # 24 concurrent requests
    'v5p-8': 48,  # 48 concurrent requests
}

# Continuous batching
async def batch_requests(engine, request_queue):
    batch = []
    while len(batch) < OPTIMAL_BATCH_SIZES['v4-8']:
        try:
            req = await asyncio.wait_for(
                request_queue.get(),
                timeout=0.1  # Wait 100ms for batch to fill
            )
            batch.append(req)
        except asyncio.TimeoutError:
            break
    
    if batch:
        # Process batch on TPU
        results = await engine.generate_batch(
            [req.prompt for req in batch]
        )
        return results
```

### Compilation Caching

```python
import jax
from functools import lru_cache

@lru_cache(maxsize=10)
def get_compiled_fn(batch_size, max_length):
    """Cache compiled functions by shape"""
    
    @jax.jit
    def inference(params, input_ids):
        # Inference logic
        return model(params, input_ids)
    
    return inference

# Use cached compilation
compiled_fn = get_compiled_fn(
    batch_size=32,
    max_length=2048
)
output = compiled_fn(params, input_ids)
```

### Memory Optimization

```python
# Gradient checkpointing (saves memory)
from jax.experimental.checkpoint import checkpoint

def inference_with_checkpointing(params, input_ids):
    # Checkpoint every N layers
    for i in range(0, num_layers, checkpoint_interval):
        layer_fn = checkpoint(layers[i])
        hidden = layer_fn(params[i], hidden)
    return hidden
```

---

## G.8: Monitoring and Debugging

### TPU Metrics

```python
from jax.profiler import start_trace, stop_trace

# Profile TPU execution
start_trace("gs://my-bucket/traces")

# Run inference
output = compiled_fn(params, input_ids)

# Stop profiling
stop_trace()

# View trace in TensorBoard:
# tensorboard --logdir=gs://my-bucket/traces
```

### Key Metrics to Track

1. **Chip Utilization**
   ```python
   # Monitor via Cloud Monitoring
   from google.cloud import monitoring_v3
   
   client = monitoring_v3.MetricServiceClient()
   
   # Query TPU utilization
   project_name = f"projects/{project_id}"
   interval = monitoring_v3.TimeInterval(...)
   
   results = client.list_time_series(
       request={
           "name": project_name,
           "filter": 'metric.type="tpu.googleapis.com/chip/utilization"',
           "interval": interval,
       }
   )
   ```

2. **Memory Usage**
3. **Request Latency**
4. **Compilation Time**
5. **Batch Efficiency**

### Debugging Tips

1. **XLA Dumps**
   ```bash
   export XLA_FLAGS="--xla_dump_to=/tmp/xla_dumps"
   python inference.py
   # Inspect /tmp/xla_dumps for compilation artifacts
   ```

2. **JAX Debugging**
   ```python
   # Enable debug mode
   jax.config.update('jax_debug_nans', True)
   jax.config.update('jax_debug_infs', True)
   ```

3. **TPU Utilization**
   ```bash
   # Watch TPU metrics
   gcloud compute tpus tpu-vm describe inference-tpu \
     --zone=us-central2-b \
     --format="value(health)"
   ```

---

## G.9: Cost Optimization

### Spot/Preemptible TPUs

```bash
# Create preemptible TPU (60-90% discount)
gcloud compute tpus tpu-vm create inference-tpu-spot \
  --zone=us-central2-b \
  --accelerator-type=v4-8 \
  --version=tpu-ubuntu2204-base \
  --preemptible
```

**Handle preemption:**
```python
import signal
import sys

def handle_preemption(signum, frame):
    print("Preemption signal received, shutting down gracefully")
    # Save state
    # Notify load balancer
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_preemption)
```

### Right-Sizing

```
Model Size → TPU Pod Size

7B model:
└── v5e-4 (4 chips) - $1.45/hour

30B model:
└── v4-8 (8 chips) - $4.40/hour

70B model:
└── v4-8 or v5e-8 (8 chips) - $4.40/hour or $2.90/hour

400B model:
└── v5p-32 (32 chips) - ~$17/hour
```

### Committed Use Discounts

```
1-Year Commitment:
├── v4-8: $3.30/hour (25% discount)
└── Saves: $950/month

3-Year Commitment:
├── v4-8: $2.60/hour (41% discount)
└── Saves: $1,580/month
```

---

## G.10: Comparison Matrix

### TPU vs GPU: Side-by-Side

| Feature | TPU (v4-8) | GPU (4x A100) |
|---------|------------|---------------|
| **Cost (On-Demand)** | $4.40/hr | $12/hr |
| **Cost (1-year)** | $3.30/hr | $8/hr |
| **Availability** | GCP only | GCP, AWS, Azure, On-prem |
| **Setup Complexity** | Medium | Low |
| **Ecosystem** | JAX, limited | PyTorch, extensive |
| **Model Support** | JAX models | All models |
| **Performance (70B)** | 40-50 tok/s | 45-55 tok/s |
| **Batch Efficiency** | Excellent | Excellent |
| **Flexibility** | Medium | High |
| **Community Support** | Growing | Mature |

### When Each Wins

**TPUs Win:**
- ✅ Cost-sensitive, sustained workload
- ✅ Already on GCP
- ✅ JAX/Flax models
- ✅ Very high batch sizes
- ✅ Long-term deployment

**GPUs Win:**
- ✅ PyTorch models
- ✅ On-premise required
- ✅ Ecosystem maturity
- ✅ Rapid experimentation
- ✅ Portable across clouds

---

## G.11: Migration Guide

### Moving from GPU to TPU

**Step 1: Assess Compatibility**
```python
# Check if your model can run on TPU
compatibility_checklist = {
    "model_framework": "pytorch",  # ❌ Need JAX
    "model_size": "70B",           # ✅ Fits in v4-8
    "deployment": "gcp",           # ✅ TPUs available
    "workload": "sustained",       # ✅ Cost effective
}

if compatibility_checklist["model_framework"] != "jax":
    print("⚠️  Need to convert model to JAX")
```

**Step 2: Convert Model**
- Option A: Find pre-converted JAX version
- Option B: Manual conversion (weeks of work)
- Option C: Use ONNX as intermediate (experimental)

**Step 3: Update Control Plane**
```go
// Add TPU backend alongside GPU
backends := []InferenceBackend{
    NewGPUBackend(gpuConfig),  // Keep GPU
    NewTPUBackend(tpuConfig),  // Add TPU
}

// Route based on model
func (r *Router) Route(req Request) InferenceBackend {
    if req.ModelID == "gemma-7b" {
        return r.tpuBackend  // JAX model → TPU
    }
    return r.gpuBackend      // PyTorch model → GPU
}
```

**Step 4: A/B Test**
- Run both GPU and TPU
- Compare performance, cost, reliability
- Gradually shift traffic

**Step 5: Optimize**
- Tune batch sizes
- Optimize compilation
- Monitor costs

---

## G.12: Summary

### Key Takeaways

1. **TPUs are cost-effective** (50-70% cheaper) for sustained workloads on GCP
2. **JAX ecosystem required** - PyTorch models need conversion
3. **Setup complexity** - Steeper learning curve than GPUs
4. **GCP lock-in** - Can't self-host or use other clouds
5. **Good for scale** - Economics improve with larger deployments

### Decision Framework

```
Should I use TPUs?

YES if:
├── On Google Cloud Platform
├── JAX/Flax models OR willing to convert
├── Sustained workload (24/7)
└── Cost optimization critical

NO if:
├── Need on-premise hosting
├── PyTorch models (and can't convert)
├── Small scale (<50 req/hour)
└── Prefer ecosystem maturity
```

### Integration with Main Book

- **Chapter 2**: Reference TPU hardware options
- **Chapter 4**: Note JetStream as TPU inference engine
- **Chapter 16**: Include TPU in cost comparisons
- **Control Plane**: TPUBackend implements same interface

---

## Resources

- **Google Cloud TPU Documentation**: https://cloud.google.com/tpu/docs
- **JAX Documentation**: https://jax.readthedocs.io/
- **JetStream**: https://github.com/google/jetstream
- **MaxText**: https://github.com/google/maxtext
- **TPU Research Cloud**: https://sites.research.google/trc/

---

## Exercises

1. **Cost Comparison**
   - Calculate TPU vs GPU costs for your workload
   - Find break-even point
   - Consider commitment discounts

2. **Model Conversion**
   - Attempt converting a small PyTorch model to JAX
   - Document challenges encountered
   - Estimate effort for production model

3. **Performance Benchmark**
   - Deploy same model on GPU and TPU
   - Measure throughput, latency, cost
   - Optimize each platform
   - Compare results

4. **Control Plane Integration**
   - Implement TPUBackend
   - Add routing logic
   - Test failover GPU ↔ TPU
   - Monitor metrics