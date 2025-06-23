BIN_DIR    := bin
TARGET_DIR := target/release
BINARIES   := llmserve api_server launcher

.PHONY: all build copy-bins clean

all: develop

develop: $(addprefix build-,$(BINARIES)) build-worker copy-bins

install: $(addprefix install-,$(BINARIES)) install-worker copy-bins

build-worker:
	@$(MAKE) -C worker build-dev

install-worker:
	@$(MAKE) -C worker install

build-%: %
	@echo "Building $*..."
	cargo build -p $< --release --locked

copy-bins:
	@mkdir -p $(BIN_DIR)
	@for bin in $(BINARIES); do \
		echo "Copying $$bin to $(BIN_DIR)/"; \
		cp $(TARGET_DIR)/$$bin $(BIN_DIR)/; \
	done

clean:
	cargo clean
	@rm -rf $(BIN_DIR)
	@$(MAKE) -C worker clean
