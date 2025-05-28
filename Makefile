.PHONY: install build clean

install-rust: install-llmserve install-api-server

install: install-llmserve install-api-server install-worker

build: install-worker
	cargo build

install-llmserve:
	cargo install --path llmserve

install-api-server:
	cargo install --path api_server

install-worker:
	cd worker && make install

clean:
	cargo clean
	cd worker && make clean
