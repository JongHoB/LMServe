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

### Launching Server

Before running the server, you must set the `LLMSERVE_HOME` environment variable to the root directory of the project:
```bash
$ export LLMSERVE_HOME=/path/to/LLMServe
```

Then, launch the server with:
```bash
$ bin/launcher --config configs/default.yaml
```

### Running Benchmark
Once the server is running, you can measure its performance using the following benchmark script:
```bash
$ python3 benchmarks/benchmark_server.py
```
