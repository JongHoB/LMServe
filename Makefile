BIN_DIR    := bin
PACKAGES   := runtime clis
BINARIES   := llm_srv api_server launcher
MAKEFLAGS += --no-print-directory

# Default setting
TARGET_DIR := target/debug
MODE := debug

.PHONY: all release build clean link-bins build-% build-worker install-worker

all: develop

develop: MODE := debug
develop: TARGET_DIR := target/debug
develop: $(addprefix build-,$(PACKAGES)) build-worker link-bins

release: MODE := release
release: TARGET_DIR := target/release
release: $(addprefix build-,$(PACKAGES)) install-worker link-bins

build-worker:
	@$(MAKE) -C runtime/worker build-dev

install-worker:
	@$(MAKE) -C runtime/worker install

build-%:
	@echo "Building $* in $(MODE) mode..."
	@if [ "$(MODE)" = "release" ]; then \
		cargo build -p $* --release --frozen; \
	else \
		cargo build -p $* --locked; \
	fi

link-bins:
	@mkdir -p $(BIN_DIR)
	@for bin in $(BINARIES); do \
		echo "Linking $(TARGET_DIR)/$$bin -> $(BIN_DIR)/$$bin"; \
		ln -sf $(CURDIR)/$(TARGET_DIR)/$$bin $(BIN_DIR)/$$bin; \
	done

clean:
	cargo clean
	@rm -rf $(BIN_DIR)
	@$(MAKE) -C runtime/worker clean
