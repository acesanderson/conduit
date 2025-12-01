### conversation state management
- upgraded MessageStore functionality to support:
 - ephemerality
 - branching
 - storing / retrieving
 - returning to previous conversations / states
 - navigation

### Persistence framework
- two data primitives
 - Message (with immutability + expanded metadata, hash id)
 - Conversation (a list of references to Messages, and related metadata like token usage, timestamps, parent ids for branching)

### Opportunities
- semantic layer
 - embeddings, vector db
 - graph (NER / relationships)
 - conversation summaries (also embedded)
- cross-conversation memory
