# LLMServe

## Requirements
* CUDA
* Protobuf compiler

To install the protobuf compiler on Ubuntu, run:
```bash
$ apt install -y protobuf-compiler
```

## Get Started
You can easily build this project by running:
```bash
$ make
```

### Inital Setup
Before running the server, you must set the `LLMSERVE_HOME` environment variable to the root directory of the project:
```bash
$ export LLMSERVE_HOME=/path/to/LLMServe
```

Additionally, LLMServe has a monitoring daemon configured with a pub/sub architecture to track the status of each node (e.g., number of running or pending requests, etc.).
Before launching LLMServe, we must prepare the `nats-server`. You can simply run it using Docker:

```bash
$ docker network create nats
$ docker run -d --name nats --network nats --rm -p 4222:4222 -p 8222:8222 nats --http_port 8222
```
Additionally, LLMServe configures pub/s

### Launching LLMServe
Then, launch the server with:
```bash
$ bin/llm_clu --config configs/default.yaml
```

### Running Benchmark
Once the server is running, you can measure its performance using the following benchmark script:
```bash
$ python3 benchmarks/benchmark_server.py --dataset sharegpt
```
