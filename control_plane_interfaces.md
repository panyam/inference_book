# Control Plane Interface Specifications

## Design Philosophy

- **Single codebase** that evolves across all four phases
- **Interface-driven** design for testability and swappability
- **Feature flags** enable/disable capabilities without code changes
- **No rewrites** - each phase adds to previous work
- **Backward compatible** - new features don't break old APIs

---

## Phase 1: Foundation (7B Models, Chapters 1-5)

### Control Plane v0.1: Single Backend

**Purpose:** Basic inference serving with observability

### Core Interfaces

```go
// InferenceBackend abstracts the actual inference engine
// Implementations: OllamaBackend, LlamaCppBackend, vLLMBackend
type InferenceBackend interface {
    // Generate produces inference output for a request
    Generate(ctx context.Context, req GenerateRequest) (GenerateResponse, error)
    
    // Health returns current backend health status
    Health(ctx context.Context) HealthStatus
    
    // Metrics returns current performance metrics
    Metrics(ctx context.Context) BackendMetrics
    
    // Close gracefully shuts down the backend
    Close() error
}

type GenerateRequest struct {
    Prompt      string
    MaxTokens   int
    Temperature float64
    TopP        float64
    Stop        []string
    Stream      bool
    Metadata    map[string]string // For tracking/logging
}

type GenerateResponse struct {
    Text         string
    TokensUsed   TokenUsage
    FinishReason string
    Latency      time.Duration
}

type TokenUsage struct {
    PromptTokens     int
    CompletionTokens int
    TotalTokens      int
}

type HealthStatus struct {
    Healthy   bool
    Message   string
    Timestamp time.Time
}

type BackendMetrics struct {
    RequestsPerSecond float64
    AverageLatency    time.Duration
    P95Latency        time.Duration
    P99Latency        time.Duration
    ActiveRequests    int
    QueueDepth        int
    ErrorRate         float64
}
```

```go
// RequestRouter determines which backend handles a request
// Phase 1: Simple - always returns the single backend
// Phase 2+: Load balancing, model routing
type RequestRouter interface {
    Route(ctx context.Context, req InferenceRequest) (InferenceBackend, error)
}

type InferenceRequest struct {
    RequestID    string
    UserID       string // Empty in Phase 1
    TenantID     string // Empty until Phase 3
    ModelID      string // Empty until Phase 4
    GenerateReq  GenerateRequest
    Priority     int    // 0 in Phase 1
    SubmittedAt  time.Time
}
```

```go
// MetricsCollector handles observability
// Implementation: PrometheusCollector
type MetricsCollector interface {
    RecordRequest(ctx context.Context, metadata RequestMetadata)
    RecordLatency(ctx context.Context, duration time.Duration)
    RecordTokens(ctx context.Context, inputTokens, outputTokens int)
    RecordError(ctx context.Context, errorType string)
}

type RequestMetadata struct {
    RequestID string
    UserID    string
    ModelID   string
    Timestamp time.Time
}
```

### Configuration

```go
type ControlPlaneConfig struct {
    // Phase 1: Core settings
    ServerPort int
    Backends   []BackendConfig
    Metrics    MetricsConfig
    
    // Phase 2+: Optional features (disabled in Phase 1)
    EnableAuth       bool
    EnableRateLimit  bool
    EnableCaching    bool
    EnableQueuing    bool
    
    // Phase 3+
    EnableMultiTenant bool
    EnablePriority    bool
    EnableCostTracking bool
    
    // Phase 4+
    EnableBilling    bool
    EnablePayments   bool
    EnableMultiModel bool
    EnableDistributed bool
}

type BackendConfig struct {
    Name     string
    Type     string // "ollama", "llamacpp", "vllm"
    Endpoint string
    Timeout  time.Duration
}

type MetricsConfig struct {
    Enabled bool
    Port    int
    Path    string
}
```

### Request Flow (Phase 1)

```
HTTP Request → API Handler → Router → Single Backend → Response → Metrics
```

---

## Phase 2: Production Ready (30B Models, Chapters 6-10)

### Control Plane v0.2: Production Grade

**Purpose:** Multi-user, production-ready serving with reliability features

### New Interfaces

```go
// UserManager handles authentication and user data
// Implementation: JWTUserManager, APIKeyUserManager
type UserManager interface {
    Authenticate(ctx context.Context, token string) (User, error)
    GetUser(ctx context.Context, userID string) (User, error)
    CreateUser(ctx context.Context, user User) error
    UpdateQuota(ctx context.Context, userID string, quota Quota) error
    GetQuota(ctx context.Context, userID string) (Quota, error)
}

type User struct {
    ID        string
    Email     string
    APIKey    string
    CreatedAt time.Time
    Quota     Quota
}

type Quota struct {
    RequestsPerMinute int
    RequestsPerDay    int
    MaxTokensPerRequest int
    RemainingToday    int
}
```

```go
// RateLimiter enforces usage limits per user
// Implementation: TokenBucketLimiter (Redis-backed)
type RateLimiter interface {
    Allow(ctx context.Context, userID string) (bool, error)
    Remaining(ctx context.Context, userID string) (int, error)
    Reset(ctx context.Context, userID string) error
}
```

```go
// QueueManager handles request queuing when backends are busy
// Implementation: RedisQueueManager, InMemoryQueueManager
type QueueManager interface {
    Enqueue(ctx context.Context, req InferenceRequest) (QueuedRequest, error)
    Dequeue(ctx context.Context) (InferenceRequest, error)
    Position(ctx context.Context, requestID string) (int, error)
    Remove(ctx context.Context, requestID string) error
    Length(ctx context.Context) (int, error)
}

type QueuedRequest struct {
    Request     InferenceRequest
    QueuedAt    time.Time
    Position    int
    EstimatedWait time.Duration
}
```

```go
// CacheManager handles response caching
// Implementation: RedisCacheManager
type CacheManager interface {
    Get(ctx context.Context, cacheKey string) (CachedResponse, bool, error)
    Set(ctx context.Context, cacheKey string, response CachedResponse, ttl time.Duration) error
    Delete(ctx context.Context, cacheKey string) error
    Clear(ctx context.Context) error
}

type CachedResponse struct {
    Response  GenerateResponse
    CachedAt  time.Time
    HitCount  int
}
```

```go
// BackendPool manages multiple backend instances
// Provides load balancing and health checking
type BackendPool interface {
    AddBackend(backend InferenceBackend) error
    RemoveBackend(name string) error
    GetBackend(ctx context.Context) (InferenceBackend, error) // Returns least loaded
    HealthCheck(ctx context.Context) map[string]HealthStatus
    Backends() []InferenceBackend
}
```

### Enhanced Configuration

```go
type Phase2Config struct {
    // Auth settings
    Auth AuthConfig
    
    // Rate limiting
    RateLimit RateLimitConfig
    
    // Caching
    Cache CacheConfig
    
    // Queue settings
    Queue QueueConfig
    
    // Backend pool
    BackendPool BackendPoolConfig
}

type AuthConfig struct {
    Enabled bool
    Type    string // "jwt", "apikey"
    Secret  string
}

type RateLimitConfig struct {
    Enabled           bool
    DefaultRPM        int // Requests per minute
    DefaultDailyLimit int
}

type CacheConfig struct {
    Enabled    bool
    Backend    string // "redis", "memory"
    DefaultTTL time.Duration
}

type QueueConfig struct {
    Enabled    bool
    MaxSize    int
    Timeout    time.Duration
}

type BackendPoolConfig struct {
    HealthCheckInterval time.Duration
    MaxRetries          int
    LoadBalancingStrategy string // "round-robin", "least-loaded"
}
```

### Request Flow (Phase 2)

```
HTTP Request → Auth → Rate Limit → Cache Check → 
Queue (if needed) → Backend Pool → Backend → 
Response → Cache Set → Metrics
```

---

## Phase 3: Multi-Tenant Platform (70B Models, Chapters 11-15)

### Control Plane v0.3: Multi-Tenant Platform

**Purpose:** Serve multiple customers with different service tiers and priorities

### New Interfaces

```go
// TenantManager handles multi-tenancy
type TenantManager interface {
    GetTenant(ctx context.Context, tenantID string) (Tenant, error)
    CreateTenant(ctx context.Context, tenant Tenant) error
    UpdateTenant(ctx context.Context, tenantID string, updates TenantUpdate) error
    DeleteTenant(ctx context.Context, tenantID string) error
    ListTenants(ctx context.Context, filters TenantFilters) ([]Tenant, error)
    GetTier(ctx context.Context, tenantID string) (Tier, error)
}

type Tenant struct {
    ID          string
    Name        string
    TierID      string
    CreatedAt   time.Time
    Settings    TenantSettings
    Users       []string // User IDs belonging to this tenant
}

type TenantSettings struct {
    AllowedModels    []string
    MaxConcurrency   int
    CustomRateLimit  *RateLimitConfig
    BillingEnabled   bool
}

type Tier struct {
    ID             string
    Name           string // "Free", "Pro", "Enterprise"
    MaxConcurrency int
    MaxQueueDepth  int
    Priority       int    // Higher = more priority
    RateLimit      RateLimitConfig
    Features       []Feature
    PricePerToken  float64 // For Phase 4 billing
}

type Feature string

const (
    FeatureAPIAccess      Feature = "api_access"
    FeaturePriorityQueue  Feature = "priority_queue"
    FeatureAdvancedModels Feature = "advanced_models"
    FeatureCustomModels   Feature = "custom_models"
    FeatureDedicated      Feature = "dedicated_capacity"
)
```

```go
// SchedulingPolicy determines request ordering and prioritization
// Implementation: WeightedFairScheduler, PriorityScheduler
type SchedulingPolicy interface {
    Schedule(ctx context.Context, requests []QueuedRequest) ([]QueuedRequest, error)
    Priority(ctx context.Context, req QueuedRequest) (int, error)
    ShouldPreempt(ctx context.Context, running, waiting QueuedRequest) (bool, error)
}
```

```go
// ResourceAllocator manages resource reservations per tenant
type ResourceAllocator interface {
    Allocate(ctx context.Context, req InferenceRequest) (Allocation, error)
    Release(ctx context.Context, allocation Allocation) error
    Available(ctx context.Context, tenantID string) (Resources, error)
    Reserve(ctx context.Context, tenantID string, resources Resources) error
}

type Allocation struct {
    ID          string
    TenantID    string
    Resources   Resources
    AllocatedAt time.Time
    ExpiresAt   time.Time
}

type Resources struct {
    GPUMemoryMB  int
    CPUCores     float64
    Concurrency  int
}
```

```go
// CostTracker records usage for billing purposes
type CostTracker interface {
    RecordUsage(ctx context.Context, usage Usage) error
    GetUsage(ctx context.Context, tenantID string, period TimePeriod) ([]Usage, error)
    GetCost(ctx context.Context, tenantID string, period TimePeriod) (Cost, error)
    AggregateUsage(ctx context.Context, period TimePeriod) (map[string]Cost, error)
}

type Usage struct {
    ID              string
    TenantID        string
    UserID          string
    RequestID       string
    ModelID         string
    PromptTokens    int
    CompletionTokens int
    TotalTokens     int
    Latency         time.Duration
    Timestamp       time.Time
    Cost            float64
}

type Cost struct {
    TenantID      string
    Period        TimePeriod
    TotalTokens   int
    TotalRequests int
    Amount        float64
    Currency      string
}

type TimePeriod struct {
    Start time.Time
    End   time.Time
}
```

### Enhanced Configuration

```go
type Phase3Config struct {
    // Multi-tenancy
    MultiTenant MultiTenantConfig
    
    // Scheduling
    Scheduler SchedulerConfig
    
    // Resource management
    Resources ResourceConfig
    
    // Cost tracking
    CostTracking CostTrackingConfig
}

type MultiTenantConfig struct {
    Enabled     bool
    DefaultTier string
}

type SchedulerConfig struct {
    Policy           string // "weighted-fair", "priority", "fifo"
    PreemptionEnabled bool
}

type ResourceConfig struct {
    EnableReservations bool
    TotalGPUMemoryMB   int
    TotalCPUCores      float64
}

type CostTrackingConfig struct {
    Enabled         bool
    StorageBackend  string // "postgres", "clickhouse"
    AggregationInterval time.Duration
}
```

### Request Flow (Phase 3)

```
HTTP Request → Auth → Tenant Resolution → Tier Check → 
Priority Assignment → Resource Allocation → 
Scheduled Queue → Backend → Cost Tracking → Response
```

---

## Phase 4: Inference Lab (400B Models, Chapters 16-18)

### Control Plane v1.0: Complete Infrastructure

**Purpose:** Full commercial platform with billing, multi-model support, and distributed coordination

### New Interfaces

```go
// BillingEngine calculates charges and generates invoices
type BillingEngine interface {
    CalculateCharges(ctx context.Context, usage []Usage) ([]Charge, error)
    CreateInvoice(ctx context.Context, tenantID string, period TimePeriod) (Invoice, error)
    GetInvoice(ctx context.Context, invoiceID string) (Invoice, error)
    ListInvoices(ctx context.Context, tenantID string, filters InvoiceFilters) ([]Invoice, error)
    MarkPaid(ctx context.Context, invoiceID string, payment PaymentInfo) error
}

type Charge struct {
    ID          string
    TenantID    string
    Description string
    Quantity    float64
    UnitPrice   float64
    Amount      float64
    Timestamp   time.Time
}

type Invoice struct {
    ID         string
    TenantID   string
    Period     TimePeriod
    Charges    []Charge
    Subtotal   float64
    Tax        float64
    Total      float64
    Status     InvoiceStatus
    DueDate    time.Time
    PaidAt     *time.Time
}

type InvoiceStatus string

const (
    InvoiceStatusDraft   InvoiceStatus = "draft"
    InvoiceStatusIssued  InvoiceStatus = "issued"
    InvoiceStatusPaid    InvoiceStatus = "paid"
    InvoiceStatusOverdue InvoiceStatus = "overdue"
    InvoiceStatusVoid    InvoiceStatus = "void"
)
```

```go
// PaymentProcessor handles payment transactions
// Implementation: StripeProcessor, PayPalProcessor
type PaymentProcessor interface {
    ProcessPayment(ctx context.Context, invoice Invoice) (PaymentResult, error)
    GetPaymentMethods(ctx context.Context, tenantID string) ([]PaymentMethod, error)
    AddPaymentMethod(ctx context.Context, tenantID string, method PaymentMethod) error
    RemovePaymentMethod(ctx context.Context, tenantID, methodID string) error
    RefundPayment(ctx context.Context, paymentID string, amount float64) error
}

type PaymentResult struct {
    TransactionID string
    Status        PaymentStatus
    Amount        float64
    ProcessedAt   time.Time
    Error         string
}

type PaymentStatus string

const (
    PaymentStatusSuccess PaymentStatus = "success"
    PaymentStatusFailed  PaymentStatus = "failed"
    PaymentStatusPending PaymentStatus = "pending"
)

type PaymentMethod struct {
    ID          string
    Type        string // "card", "bank_account"
    Last4       string
    ExpiryMonth int
    ExpiryYear  int
    IsDefault   bool
}
```

```go
// ModelRegistry manages multiple models and their deployments
type ModelRegistry interface {
    ListModels(ctx context.Context) ([]ModelInfo, error)
    GetModel(ctx context.Context, modelID string) (ModelInfo, error)
    RegisterModel(ctx context.Context, model ModelInfo) error
    DeployModel(ctx context.Context, deployment ModelDeployment) error
    UndeployModel(ctx context.Context, deploymentID string) error
    UpdateModel(ctx context.Context, modelID string, update ModelUpdate) error
    GetDeployment(ctx context.Context, deploymentID string) (ModelDeployment, error)
}

type ModelInfo struct {
    ID              string
    Name            string
    Version         string
    Size            int64 // Parameters
    QuantizationType string
    Capabilities    []string // "chat", "code", "vision"
    ContextWindow   int
    RequiredVRAM    int // MB
    PricePerToken   float64
    Metadata        map[string]string
}

type ModelDeployment struct {
    ID          string
    ModelID     string
    NodeIDs     []string // Distributed across these nodes
    Status      DeploymentStatus
    Replicas    int
    CreatedAt   time.Time
    UpdatedAt   time.Time
    Config      DeploymentConfig
}

type DeploymentStatus string

const (
    DeploymentStatusPending  DeploymentStatus = "pending"
    DeploymentStatusDeploying DeploymentStatus = "deploying"
    DeploymentStatusReady    DeploymentStatus = "ready"
    DeploymentStatusFailed   DeploymentStatus = "failed"
    DeploymentStatusScaling  DeploymentStatus = "scaling"
)

type DeploymentConfig struct {
    TensorParallelism int
    PipelineParallelism int
    MaxBatchSize      int
    Quantization      string
}
```

```go
// DistributedCoordinator manages multi-node, multi-GPU coordination
type DistributedCoordinator interface {
    RegisterNode(ctx context.Context, node Node) error
    UnregisterNode(ctx context.Context, nodeID string) error
    GetNode(ctx context.Context, nodeID string) (Node, error)
    ListNodes(ctx context.Context) ([]Node, error)
    AssignWorkload(ctx context.Context, workload Workload) ([]NodeAssignment, error)
    MonitorHealth(ctx context.Context) (ClusterHealth, error)
    Rebalance(ctx context.Context) error
}

type Node struct {
    ID          string
    Hostname    string
    GPUs        []GPUInfo
    CPUCores    int
    MemoryGB    int
    Status      NodeStatus
    Workloads   []string // Deployment IDs
    LastSeen    time.Time
}

type GPUInfo struct {
    Index       int
    Model       string
    MemoryMB    int
    UsedMemoryMB int
    Utilization float64
}

type NodeStatus string

const (
    NodeStatusHealthy   NodeStatus = "healthy"
    NodeStatusDegraded  NodeStatus = "degraded"
    NodeStatusUnhealthy NodeStatus = "unhealthy"
    NodeStatusOffline   NodeStatus = "offline"
)

type Workload struct {
    DeploymentID string
    ModelID      string
    Resources    Resources
}

type NodeAssignment struct {
    NodeID    string
    GPUIndices []int
    Allocation Resources
}

type ClusterHealth struct {
    TotalNodes    int
    HealthyNodes  int
    TotalGPUs     int
    AvailableGPUs int
    TotalMemoryMB int
    UsedMemoryMB  int
    Timestamp     time.Time
}
```

```go
// CapacityPlanner predicts future needs and recommends scaling
type CapacityPlanner interface {
    PredictCapacity(ctx context.Context, horizon time.Duration) (CapacityForecast, error)
    RecommendScaling(ctx context.Context) (ScalingRecommendation, error)
    OptimizeAllocation(ctx context.Context) (AllocationPlan, error)
}

type CapacityForecast struct {
    Period            TimePeriod
    PredictedRequests int
    PredictedTokens   int64
    RequiredGPUs      int
    Confidence        float64
}

type ScalingRecommendation struct {
    Action      ScalingAction
    TargetNodes int
    Reason      string
    Urgency     UrgencyLevel
}

type ScalingAction string

const (
    ScalingActionScaleUp   ScalingAction = "scale_up"
    ScalingActionScaleDown ScalingAction = "scale_down"
    ScalingActionMaintain  ScalingAction = "maintain"
)

type UrgencyLevel string

const (
    UrgencyLow      UrgencyLevel = "low"
    UrgencyMedium   UrgencyLevel = "medium"
    UrgencyHigh     UrgencyLevel = "high"
    UrgencyCritical UrgencyLevel = "critical"
)
```

```go
// AnomalyDetector identifies unusual patterns in usage or performance
type AnomalyDetector interface {
    DetectAnomalies(ctx context.Context, metrics []Metric) ([]Anomaly, error)
    Alert(ctx context.Context, anomaly Anomaly) error
    GetAnomalies(ctx context.Context, period TimePeriod) ([]Anomaly, error)
}

type Metric struct {
    Name      string
    Value     float64
    Timestamp time.Time
    Labels    map[string]string
}

type Anomaly struct {
    ID          string
    Type        AnomalyType
    Severity    Severity
    Description string
    DetectedAt  time.Time
    Metrics     []Metric
    Suggestion  string
}

type AnomalyType string

const (
    AnomalyTypeLatencySpike   AnomalyType = "latency_spike"
    AnomalyTypeErrorRateHigh  AnomalyType = "error_rate_high"
    AnomalyTypeCapacityLow    AnomalyType = "capacity_low"
    AnomalyTypeUnusualUsage   AnomalyType = "unusual_usage"
    AnomalyTypeQualityDrop    AnomalyType = "quality_drop"
)

type Severity string

const (
    SeverityInfo     Severity = "info"
    SeverityWarning  Severity = "warning"
    SeverityError    Severity = "error"
    SeverityCritical Severity = "critical"
)
```

### Complete Configuration

```go
type Phase4Config struct {
    // All previous phases included
    
    // Billing
    Billing BillingConfig
    
    // Payments
    Payments PaymentConfig
    
    // Model registry
    Models ModelRegistryConfig
    
    // Distributed coordination
    Distributed DistributedConfig
    
    // Capacity planning
    CapacityPlanning CapacityPlanningConfig
    
    // Anomaly detection
    AnomalyDetection AnomalyDetectionConfig
}

type BillingConfig struct {
    Enabled          bool
    BillingCycle     string // "monthly", "usage-based"
    InvoiceDay       int    // Day of month for invoice generation
    GracePeriodDays  int
}

type PaymentConfig struct {
    Provider       string // "stripe", "paypal"
    WebhookSecret  string
    AutoCharge     bool
}

type ModelRegistryConfig struct {
    StorageBackend string // "postgres", "etcd"
    AutoDeployment bool
}

type DistributedConfig struct {
    Enabled            bool
    CoordinatorType    string // "etcd", "consul", "kubernetes"
    HeartbeatInterval  time.Duration
    RebalanceInterval  time.Duration
}

type CapacityPlanningConfig struct {
    Enabled              bool
    ForecastHorizon      time.Duration
    PredictionModel      string // "linear", "arima", "ml"
    AutoScalingEnabled   bool
}

type AnomalyDetectionConfig struct {
    Enabled           bool
    SensitivityLevel  float64
    AlertChannels     []string // "email", "slack", "pagerduty"
}
```

### Request Flow (Phase 4)

```
HTTP Request → Auth → Tenant/Billing Check → Model Selection → 
Distributed Scheduler → Node Selection → GPU Allocation →
Inference → Billing Update → Analytics → Response

Parallel Systems:
- Billing processor (periodic invoice generation)
- Capacity planner (auto-scaling recommendations)
- Anomaly detector (quality/performance monitoring)
- Model updater (A/B testing, canary deployments)
```

---

## Integration Points

### External Systems

```go
// Observability
type ObservabilityProvider interface {
    // Metrics (Prometheus)
    RecordMetric(name string, value float64, labels map[string]string)
    
    // Logging (structured logging)
    Log(level LogLevel, message string, fields map[string]interface{})
    
    // Tracing (OpenTelemetry)
    StartSpan(ctx context.Context, name string) (context.Context, Span)
}

// Storage
type StorageProvider interface {
    // Persistent storage for configuration, state
    Get(ctx context.Context, key string) ([]byte, error)
    Set(ctx context.Context, key string, value []byte) error
    Delete(ctx context.Context, key string) error
    List(ctx context.Context, prefix string) ([]string, error)
}

// Notification
type NotificationProvider interface {
    SendEmail(ctx context.Context, to, subject, body string) error
    SendSlack(ctx context.Context, channel, message string) error
    SendWebhook(ctx context.Context, url string, payload interface{}) error
}
```

---

## Testing Strategy

### Interface Mocking

Each interface should have:
1. Production implementation
2. Mock implementation for testing
3. In-memory implementation for development

Example:
```go
// Production
type PostgresUserManager struct { ... }

// Mock for testing
type MockUserManager struct {
    Users map[string]User
    // ... controllable test behavior
}

// In-memory for development
type InMemoryUserManager struct {
    users sync.Map
}
```

### Integration Tests

Test flows across interfaces:
```go
func TestCompleteRequestFlow(t *testing.T) {
    // Setup control plane with all interfaces
    cp := NewControlPlane(config)
    
    // Test request goes through all stages
    response, err := cp.HandleRequest(ctx, request)
    
    // Verify each stage was called correctly
    assert.NoError(t, err)
    assert.NotNil(t, response)
    // ... more assertions
}
```

---

## Migration Path

### Phase 1 → Phase 2
```yaml
# Add to config.yaml
enable_auth: true
enable_rate_limit: true
enable_caching: true
enable_queuing: true

auth:
  type: jwt
  secret: ${JWT_SECRET}

rate_limit:
  default_rpm: 60
```

### Phase 2 → Phase 3
```yaml
enable_multi_tenant: true
enable_priority: true
enable_cost_tracking: true

multi_tenant:
  default_tier: free

tiers:
  - id: free
    max_concurrency: 1
    priority: 1
  - id: pro
    max_concurrency: 5
    priority: 2
```

### Phase 3 → Phase 4
```yaml
enable_billing: true
enable_payments: true
enable_multi_model: true
enable_distributed: true

billing:
  cycle: monthly
  invoice_day: 1

payments:
  provider: stripe
  api_key: ${STRIPE_API_KEY}

models:
  - id: qwen-400b
    deployment:
      tensor_parallelism: 8
      nodes: [node1, node2]
```

Each phase is backward compatible - older configs work with newer control planes.