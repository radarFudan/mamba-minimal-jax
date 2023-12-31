1. Vanilla on 16-core CPU (Xeon W-3335). 
2. Vanilla on RTX3090. (see `model.py`) (This is the v0)
3. Associative scan on RTX3090. (see `model_speedup_v1.py`)
4. 

(I'll record the minimal runtime only)

| Method            | Time    |
|-------------------|---------|
| cpu               | 31min   |
| vanilla           | 52min+  |
| associative_scan  | 9min    |