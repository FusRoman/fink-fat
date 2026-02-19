WORKSPACE_ROOT := $(shell pwd)

.PHONY: doc
doc:
	RUSTDOCFLAGS="--html-in-header $(WORKSPACE_ROOT)/katex-header.html" cargo doc --open
