PY_PROTO_OUT_DIR = llmserve_worker/pb
PROTO_FILE = ../proto/worker.proto
PROTO_DIR = ../proto

.PHONY: install clean

install: install-pb
	@echo "Installing llmserve-worker and Python dependencies..."
	pip install -e .

install-pb: check-grpc-tools
	@echo "Compiling proto for Python..."
	mkdir -p $(PY_PROTO_OUT_DIR)
	python -m grpc_tools.protoc \
		--proto_path=$(PROTO_DIR) \
		--python_out=$(PY_PROTO_OUT_DIR) \
		--grpc_python_out=$(PY_PROTO_OUT_DIR) \
		$(PROTO_FILE)
	find $(PY_PROTO_OUT_DIR) -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
	touch $(PY_PROTO_OUT_DIR)/__init__.py

check-grpc-tools:
	@python3 -c "import grpc_tools" 2>/dev/null || \
		( echo "Installing grpcio-tools..."; pip install grpcio-tools )

clean:
	rm -f $(PY_PROTO_OUT_DIR)/*.py
	rmdir $(PY_PROTO_OUT_DIR)
