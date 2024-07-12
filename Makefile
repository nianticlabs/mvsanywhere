SHELL = /bin/bash

SYSTEM_NAME := $(shell uname)
SYSTEM_ARCHITECTURE := $(shell uname -m)
MAMBA_INSTALL_SCRIPT := Mambaforge-$(SYSTEM_NAME)-$(SYSTEM_ARCHITECTURE).sh

MAMBA_ENV_NAME := doubletake
PACKAGE_FOLDER := src/doubletake

# HELP: install-mamba: Install Mamba
.PHONY: install-mamba
install-mamba:
	@echo "Installing Mamba..."
	@curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/$(MAMBA_INSTALL_SCRIPT)"
	@chmod +x "$(MAMBA_INSTALL_SCRIPT)"
	@./$(MAMBA_INSTALL_SCRIPT)
	@rm "$(MAMBA_INSTALL_SCRIPT)"

# HELP: create-mamba-env: Create a new Mamba environment 
.PHONY: create-mamba-env
create-mamba-env:
	@mamba env create -f environment.yml -n "$(MAMBA_ENV_NAME)"
	@echo -e " Mamba env created!"
	@echo "Installing pip dependencies..."
	@echo -e "🎉🎉 Your new $(MAMBA_ENV_NAME) mamba environment is ready to be used 🎉🎉"


# HELP: black: Run Black
.PHONY: black
black:
	@echo "Running Black..."
	black --check --diff --config pyproject.toml .

# HELP: isort: Run isort
.PHONY: isort
isort:
	@echo "Running isort..."
	isort ${PACKAGE_FOLDER} tests -c --settings-path pyproject.toml

# HELP: format-code: Format code using
.PHONY: format-code
format-code:
	@echo "Formatting code..."
	isort ${PACKAGE_FOLDER} tests --settings-path pyproject.toml
	black --config pyproject.toml .
	@echo -e "✅✅ Code formatted ✅✅"