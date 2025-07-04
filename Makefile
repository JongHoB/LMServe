BIN_DIR    := bin
TARGET_DIR := target/release
PACKAGES   := runtime clis
BINARIES   := llm_srv api_server llm_clu

.PHONY: all build copy-bins clean

all: develop

develop: $(addprefix build-,$(PACKAGES)) build-worker copy-bins

install: $(addprefix install-,$(PACKAGES)) install-worker copy-bins

build-worker:
	@$(MAKE) -C runtime/worker build-dev

install-worker:
	@$(MAKE) -C runtime/worker install

build-%:
	@echo "Building $*..."
	cargo build -p $* --release --locked

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
