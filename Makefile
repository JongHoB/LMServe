.PHONY: install build clean

install: install-llmserve install-worker

build: install-worker
	cargo build

install-llmserve:
	cargo install --path llmserve

install-worker:
	cd worker && make install

clean:
	cargo clean
	cd worker && make clean
