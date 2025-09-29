## Design Spec

The Chain Odometer is a token usage tracking system that provides real-time monitoring and historical analytics for LLM token consumption across the Chain framework. It operates at three levels: conversation-level tracking within MessageStore instances, session-level tracking across all Model queries, and persistent storage for long-term analytics.
## Core Requirements

### Functional Requirements

- **Automatic Token Tracking**: Transparently track input/output tokens for all successful Model queries
- **Multi-Level Monitoring**: Support conversation, session, and persistent tracking simultaneously
- **Real-Time Context Monitoring**: Display current token usage relative to model context windows
- **Historical Analytics**: Store and visualize token usage patterns over time
- **Provider/Model Breakdown**: Track usage by provider (OpenAI, Anthropic, etc.) and specific models
### Non-Functional Requirements

- **Default Enabled**: Tracking enabled by default but suppressible for other users
- **Minimal Performance Impact**: In-memory tracking with database writes only on process exit
- **Graceful Degradation**: Continue functioning if database is unavailable
- **Simple Integration**: No changes required to existing Chain usage patterns

## Architecture Design

### Token Emission Strategy

Token data is emitted from `Response.__init__()` when successful queries complete. This ensures:

- Only successful queries are tracked
- Leverages existing Usage data from provider APIs
- Natural integration point without modifying query logic

### Three-Tier Tracking System

- **Scope**: Individual MessageStore instances
- **Lifecycle**: Created/destroyed with MessageStore
- **Purpose**: Real-time context window monitoring
- **Storage**: In-memory only
- **Features**:
    - Current conversation token count
    - Context window utilization percentage
    - Warnings when approaching context limits

#### 2. Session Odometer

- **Scope**: Entire Python process
- **Lifecycle**: Model class singleton, always active when Model._odometer exists
- **Purpose**: Track all queries in current session
- **Storage**: In-memory until process exit
- **Features**: Aggregate session totals by provider/model

#### 3. Persistent Odometer

- **Scope**: Historical data across all sessions
- **Lifecycle**: Always active when enabled
- **Purpose**: Long-term usage analytics and reporting
- **Storage**: Database (SQLite fallback, PostgreSQL preferred)
- **Features**: Time-series data for dashboard visualization

## Data Models

### Token Event Structure

```python
@dataclass
class TokenEvent:
    provider: str           # "openai", "anthropic", etc.
    model: str             # "gpt-4o", "claude-3-5-sonnet", etc.
    input_tokens: int      # Input token count
    output_tokens: int     # Output token count  
    timestamp: float       # Unix epoch timestamp
    host: str              # Simple host detection for multi-machine tracking
```

### Database Schema (PostgreSQL/SQLite)

```sql
CREATE TABLE token_usage (
    id SERIAL PRIMARY KEY,
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    timestamp BIGINT NOT NULL,
    host VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_token_usage_timestamp ON token_usage(timestamp);
CREATE INDEX idx_token_usage_provider_model ON token_usage(provider, model);
```

## Integration Points

### Model Class Integration

```python
class Model:
    _odometer: Optional[SessionOdometer] = SessionOdometer()  # Default enabled
    
    # Existing query method remains unchanged
    # Token emission happens in Response.__init__()
```

### MessageStore Integration

```python
class MessageStore:
    def __init__(self, ..., odometer: Optional[ConversationOdometer] = None):
        self.odometer = odometer or ConversationOdometer()
```

### Response Class Integration

```python
class Response:
    def __init__(self, ...):
        # Existing initialization
        
        # Emit token event to available odometers
        if Model._odometer:  # Session tracking
            Model._odometer.record_usage(self.input_tokens, self.output_tokens, ...)
            
        # Conversation tracking handled via MessageStore binding
```

## Configuration System

### Database Configuration

```python
ODOMETER_CONFIG = {
    "enabled": True,
    "database_url": "postgresql://localhost:5432/chain_tokens",  
    "fallback_database": "sqlite:///chain_tokens.db",
    "table_name": "token_usage",
    "batch_size": 1000,  # For future batched writes
}
```

### Environment Variable Support

- `CHAIN_ODOMETER_ENABLED`: Enable/disable tracking
- `CHAIN_ODOMETER_DB_URL`: Database connection string
- `CHAIN_ODOMETER_FALLBACK`: Fallback to SQLite if PostgreSQL unavailable

## Dashboard System (TokenDash)

### CLI Interface Using Rich

```python
# Usage examples:
chain-odometer --monthly 2024-12    # Monthly breakdown
chain-odometer --provider openai     # Provider-specific stats  
chain-odometer --model gpt-4o        # Model-specific stats
chain-odometer --session             # Current session stats
```

### Dashboard Views

1. **Aggregate by Provider**: Total tokens per provider with cost estimates
2. **Provider Drill-down**: Model breakdown within each provider
3. **Time Series**: Monthly/daily usage trends
4. **Session Summary**: Current session statistics
5. **Context Window**: Real-time conversation usage (when in MessageStore context)

## Technical Challenges & Solutions

### Challenge: Circular Import Prevention

**Problem**: Response needs to notify Model odometer without circular imports **Solution**: Use dependency injection pattern where Response receives odometer reference during initialization, or implement a global event bus that odometers subscribe to

### Challenge: Database Availability

**Problem**: PostgreSQL may be unavailable in some environments  
**Solution**: Automatic fallback to SQLite with optional manual migration tools

### Challenge: Multi-Machine Coordination

**Problem**: Multiple machines writing to shared PostgreSQL **Solution**: Simple host detection in token events, let database handle concurrency

### Challenge: Token Attribution Consistency

**Problem**: Ensuring same tokens counted across conversation/session odometers **Solution**: Single emission point (Response.**init**) with fan-out to registered odometers

## Out of Scope (MVP)

- Streaming response token tracking
- Automatic message pruning on context limits
- Billing rate integration and cost calculation
- User-defined tags or categories
- Multi-user/multi-project support
- Individual query metadata beyond token counts
- Real-time database writes (exit-only for performance)

## Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1)

1. Create TokenEvent data structure
2. Implement SessionOdometer as Model singleton
3. Add token emission to Response.**init**()
4. Create basic SQLite database integration
5. Add configuration system

### Phase 2: MessageStore Integration (Week 2)

1. Implement ConversationOdometer class
2. Integrate with MessageStore initialization
3. Add context window monitoring methods
4. Implement real-time usage display

### Phase 3: Persistent Storage (Week 3)

1. Implement PersistentOdometer with SQLAlchemy
2. Add PostgreSQL support with SQLite fallback
3. Create database schema and migration tools
4. Implement process exit hooks for data persistence

### Phase 4: Dashboard & Analytics (Week 4)

1. Create Rich-based CLI dashboard (TokenDash)
2. Implement provider/model breakdown views
3. Add time-series visualization
4. Create session summary displays

### Phase 5: Polish & Integration (Week 5)

1. Add comprehensive error handling and graceful degradation
2. Implement host detection for multi-machine scenarios
3. Add configuration validation and defaults
4. Update documentation and examples
5. Integration testing across all Chain components

### Future Enhancements (Post-MVP)

- Streaming response support
- Cost calculation with manual rate configuration
- Performance optimizations (batched writes, connection pooling)
- Advanced analytics (usage patterns, efficiency metrics)
- Web-based dashboard interface
- Export/import functionality for data portability
