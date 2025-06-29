                                            ┌─────────────────────────────┐
                                            │      USER INPUT (Prompt)    │
                                            └────────────┬────────────────┘
                                                         │
                                              ┌──────────▼───────────┐
                                              │ NATURAL LANGUAGE     │
                                              │ UNDERSTANDING (NLP)  │
                                              └──────────┬───────────┘
                                                         │
                                      ┌──────────────────▼──────────────────┐
                                      │ CONTEXT ENGINE                      │
                                      │ (Chat History, Open Files,         │
                                      │ Terminal Output, Commands, etc.)   │
                                      └──────────┬────────────┬────────────┘
                                                 │            │
                              ┌──────────────────▼──┐     ┌───▼───────────────────┐
                              │ CODE MEMORY / CACHE │     │ SMART CONTEXT ROUTER  │
                              └────────┬────────────┘     └────────────┬──────────┘
                                       │                               │
                 ┌────────────────────▼────────────┐      ┌────────────▼────────────────┐
                 │ INTENT ANALYZER / TASK DECIDER  │      │      FILE TRACKER           │
                 └────────────┬────────────────────┘      │ (Monitors file changes)     │
                              │                            └────────────┬───────────────┘
              ┌──────────────▼──────────────┐                    ┌──────▼────────────┐
              │ IS IT FILE-RELATED?         │─────────Yes──────►│ FILE MANAGER      │
              │ (Else code/gen/debug/term)  │                   │ Create, Edit,     │
              └─────────────────────────────┘                   │ Delete, Move etc. │
                         │                                        └──────┬────────────┘
                        No
                         │
        ┌────────────────▼────────────────────┐
        │ TASK HANDLER MODULE (Multithreaded) │
        └────────────────┬────────────────────┘
                         │
        ┌────────────────▼─────────────┐
        │ TASK TYPES:                  │
        │ • Terminal Execution         │
        │ • Code Generation            │
        │ • Bug Detection & Fix        │
        │ • Search Web                 │
        │ • Cross-language Translation │
        └──────────┬───────────────────┘
                   │
        ┌──────────▼──────────────┐
        │ MULTITHREAD TASK QUEUE │  ◄─────┐
        │ (Parallel Workers)     │        │
        └─────┬──────────▲───────┘        │
              │          │                │
      ┌───────▼──┐  ┌────▼────────────┐   │
      │Terminal  │  │Code Generator   │   │
      │Command   │  │(LLM based)      │   │
      │Runner    │  └────┬────────────┘   │
      └────┬─────┘       │                │
           │      ┌──────▼─────────────┐  │
           │      │ Error Handler /    │  │
           │      │ Autonomous Debugger│──┘
           │      └──────┬─────────────┘
           │             │
   ┌───────▼─────────────▼───────┐
   │ SELF-EVAL + CODE IMPROVER  │
   │ Chain-of-Thought + RAG     │
   └────────────┬───────────────┘
                │
     ┌──────────▼──────────────┐
     │ SMART SUGGESTIONS ENGINE│
     │ (Next lines, functions, │
     │ structure completions)  │
     └──────────┬──────────────┘
                │
         ┌──────▼────────────┐
         │ RESULT LOGGER     │
         │ (Knows what was   │
         │ done so far)      │
         └────────┬──────────┘
                  │
       ┌──────────▼────────────┐
       │ OUTPUT TO USER        │
       │ (via interactive CLI) │
       └───────────────────────┘
