# LMServe

**LMServe** is a lightweight and fast LLM serving framework for academic purposes. It includes the following features.

* **KV block management with PagedAttention** for memory efficiency
* **Prefix KV sharing** to cut redundant compute and memory pressure across multiple requests 
* **Multi-level KV caching** across GPU, host DRAM, and SSD to avoid recomputation for historical KVs
* **Request reordering** to mitigate head-of-line blocking and improve tail latency
* **Chunked prefill** to reduce decode delays from long prefills
* **Disaggregated inference** to isolate prefill and decode phases

Note that LMServe includes key ideas proposed in our ASPLOS 2025 paper, `Accelerating LLM Serving for Multi-turn Dialogues with Efficient Resource Management` [[Paper](https://dl.acm.org/doi/pdf/10.1145/3676641.3716245)].


## Requirements
* CUDA
* OpenSSL
* Protobuf compiler

To install the openssl and the protobuf compiler on Ubuntu, run:
```bash
$ apt install -y pkg-config libssl-dev protobuf-compiler
```

## Supported Models
* **Llama**: 2, 3, 3.1, 3.2
* **Qwen**: 2, 2.5

## Get Started
You can easily build this project by running:
```bash
$ make
```

### Inital Setup
Before running the server, you must set the `LMSERVE_HOME` environment variable to the root directory of the project:
```bash
$ export LMSERVE_HOME=/path/to/LMServe
```

Additionally, LMServe has a monitoring daemon configured with a pub/sub architecture to track the status of each node (e.g., number of running or pending requests, etc.).
Before launching LMServe, we must prepare the `nats-server`. You can simply run it using Docker:

```bash
$ docker network create nats
$ docker run -d --name nats --network nats --rm -p 4222:4222 -p 8222:8222 nats --http_port 8222
```

### Launching LMServe
Then, launch the server with:
```bash
$ bin/launcher --config configs/default.yaml
```

### Running Benchmark

Once the server is running, you can measure its performance using the following benchmark scripts:

| Single-turn benchmark
```bash
$ python3 benchmarks/benchmark_server.py --dataset sharegpt
```

| Multi-turn benchmark
```bash
$ python3 benchmark/benchmark_server_chat.py --dataset sharegpt_chat --num-clients 50
```
