. PHONY: help

help:
	@echo "Available targets are pulled from the makefiles/ directory"

include makefiles/train.mk
include makesfiles/inference.mk