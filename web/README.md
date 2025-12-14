# Inference Book - Interactive Calculators

Interactive web-based calculators for the formulas in "Self-Hosted AI Inference" book.

## Features

- **KaTeX formula rendering** - Beautiful math display
- **URL query parameters** - Pre-populate values from book examples
- **Quick presets** - Common configurations for different model sizes
- **Copy link** - Share calculator state with others
- **Responsive design** - Works on mobile and desktop

## Calculators

### Chapter 3: Hardware Fundamentals

| Calculator | URL | Description |
|------------|-----|-------------|
| Weight Memory | `/c/3/vram/` | Calculate GPU memory for model weights |
| KV Cache | `/c/3/kv/` | Calculate KV cache memory requirements |
| Total VRAM | `/c/3/total/` | Estimate total GPU memory needed |
| Break-Even | `/c/3/breakeven/` | Buy vs rent analysis |

## Running Locally

```bash
cd web
go mod tidy
go run main.go
# Open http://localhost:8088
```

## URL Structure

```
/                    # Calculator index
/c/3/                # Chapter 3 calculators
/c/3/vram/           # Weight Memory calculator
/c/3/vram/?p=70&q=0.5  # With preset values (70B, INT4)
```

## Query Parameters

### Weight Memory (`/c/3/vram/`)
- `p` - Parameters in billions (default: 7)
- `q` - Bytes per parameter (4=FP32, 2=FP16, 1=INT8, 0.5=INT4)

### KV Cache (`/c/3/kv/`)
- `l` - Layers
- `h` - KV heads
- `d` - Head dimension
- `c` - Context length
- `b` - Bytes per value
- `n` - Batch size

### Total VRAM (`/c/3/total/`)
- `p` - Parameters (billions)
- `q` - Quantization (bytes)
- `c` - Context length
- `n` - Batch size
- `overhead` - Overhead multiplier (1.10, 1.15, 1.20)

### Break-Even (`/c/3/breakeven/`)
- `hw` - Hardware cost ($)
- `setup` - Setup costs ($)
- `rent` - Rental rate ($/hour)
- `hours` - Usage hours/month
- `power` - Power cost ($/kWh)
- `watts` - GPU wattage

## Tech Stack

- **s3gen** - Static site generator (Go)
- **KaTeX** - Math rendering
- **Vanilla JS** - No framework dependencies

## Book Integration

Add calculator links to LaTeX chapters:

```latex
\calclink{/c/3/vram?p=7&q=0.5}{vram-7b-q4}
```
