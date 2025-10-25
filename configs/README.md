# LLMCluConfig — YAML Options & Defaults

This document lists the available options for `LLMCluConfig`

---

## Top Level: `LLMCluConfig`

| Key | Type | Default |
|---|---|---|
| `model_name` | `String` | **None (required)** |
| `nats_uri` | `String` | `nats://127.0.0.1:4222` |
| `api_server` | `APIServerConfig` | **None (required)** |
| `controller` | `ControllerConfig` | **None (required)** |
| `llm_servers` | `Vec<LLMSrvConfig>` | **None (required)** |

---

## `APIServerConfig`

| Key | Type | Default |
|---|---|---|
| `api_address` | `String` | `127.0.0.1:8000` |

---

## `LLMSrvConfig`

| Key | Type | Default | Notes |
|---|---|---|---|
| `kind` | `types::EngineKind` | **None (required)** | Engine type |
| `address` | `String` | **None (required)** | LLM server address (`host:port`) |
| `engine` | `EngineConfig` | **See per-field defaults below** | See below |

---

## `EngineConfig`

| Key | Type | Default | Notes |
|---|---|---|---|
| `block_size` | `usize` | `16` | Allocation block size |
| `gpu_memory_fraction` | `f32` | `0.9` | Fraction of GPU memory to use |
| `host_kv_cache_size` | `usize` | `16` | Host DRAM KV-cache size |
| `disk_kv_cache_size` | `usize` | `0` | Disk KV-cache size |
| `disk_kv_cache_path` | `String` | `/tmp` | KV-cache path on disk |
| `enable_reorder` | `bool` | `false` | Enable request reordering |
| `max_batch_size` | `usize` | `256` | Maximum batch size |
| `max_seq_len` | `usize` | `16384` | Maximum sequence length |
| `max_num_batched_tokens` | `usize` | `2048` | Max tokens per batch |
| `tp_size` | `u8` | `1` | Tensor parallel size |

---

## `ControllerConfig`

| Key | Type | Default | Notes |
|---|---|---|---|
| `route_policy` | `string` | `round_robinn` | `round_robin` or `load_balance` |

---

## Minimal YAML Example

```yaml
model_name: "Qwen/Qwen2.5-7B"
nats_uri: "nats://127.0.0.1:4222" # default if omitted

api_server:
  api_address: "127.0.0.1:8000" # default if omitted

controller:
  route_policy: "round_robin"  # default if omitted

llm_servers:
  - kind: "all"
    address: "127.0.0.1:7000"
    # If 'engine' is omitted, the following defaults apply:
    engine:
      block_size: 16
      gpu_memory_fraction: 0.9
      host_kv_cache_size: 16
      disk_kv_cache_size: 0
      disk_kv_cache_path: "/tmp"
      enable_reorder: false
      max_batch_size: 256
      max_seq_len: 16384
      max_num_batched_tokens: 2048
      tp_size: 1
```
