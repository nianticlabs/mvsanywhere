RED='\033[1;31m'
GREEN='\033[1;32m'
CYAN='\033[1;36m'
NOCOLOR='\033[0m'

# Use Bash if it's available and fallback to regular Shell (needed to use Docker in GitLab CI Pipeline)
ifeq (, $(shell which bash))
SHELL = /bin/sh
else
SHELL = /bin/bash
endif

SYSTEM_NAME := $(shell uname)
SYSTEM_ARCHITECTURE := $(shell uname -m)
MAMBA_INSTALL_SCRIPT := Mambaforge-$(SYSTEM_NAME)-$(SYSTEM_ARCHITECTURE).sh

GCP_PROJECT := niantic-masala
GCP_REPOSITORY_NAME := research-docker
DOCKER_IMG_PATH := europe-west4-docker.pkg.dev/$(GCP_PROJECT)/$(GCP_REPOSITORY_NAME)/ar-planes

MAMBA_ENV_NAME := geometryhints
PACKAGE_FOLDER := src/geometryhints


#
# Mamba environment
# 

# HELP: install-mamba: [Mamba] Install Mamba on a fresh environment
.PHONY: install-mamba
install-mamba:
	@echo "🏗  Installing Mamba..."
	@curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/$(MAMBA_INSTALL_SCRIPT)"
	@chmod +x "$(MAMBA_INSTALL_SCRIPT)"
	@./$(MAMBA_INSTALL_SCRIPT)
	@rm "$(MAMBA_INSTALL_SCRIPT)"

# HELP: create-mamba-env: [Mamba] Create new Mamba environment 
.PHONY: create-mamba-env
create-mamba-env:
ifeq ($(SYSTEM_NAME), Darwin)
	@echo -e $(RED)"⚠️  WARNING!"$(NOCOLOR)" This Makefile target may fail on macOS. In such case, please run this command manually:"
	@echo ""
	@echo "    $$ mamba env create -f environment.yml -n \"$(MAMBA_ENV_NAME)\""
	@echo ""
endif
	@mamba env create -f environment.yml -n "$(MAMBA_ENV_NAME)"
	@echo -e $(GREEN)"✅ Mamba env created! ✅"$(NOCOLOR)


# HELP: verify-mamba-env: [Mamba] Verify existing Mamba environment
.PHONY: verify-mamba-env
verify-mamba-env:
	@echo "❓ Verifying Mamba environment..."
	@python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda); print('CUDA Device Count:', torch.cuda.device_count())"

# HELP: absolute-imports: [Static Analysis] Check if all imports are absolute
.PHONY: absolute-imports
absolute-imports:
	@echo "🚀 Checking if all imports are absolute..."
	@# Look for any code lines starting with "from ." in any Python file (recursively in current directory)
	@if grep -r -E "^from \." --include \*.py .; then \
		echo -e $(RED)"❌ Found relative imports!"$(NOCOLOR); \
		exit 1; \
	else \
		echo -e $(GREEN)"Looks good! 🎉"$(NOCOLOR); \
	fi

# HELP: darglint: [Static Analysis] Run darglint
.PHONY: darglint
darglint:
	@echo "🚀 Running darglint..."
	darglint .

# HELP: black: [Static Analysis] Run Black
.PHONY: black
black:
	@echo "🚀 Running Black..."
	black --check --diff --config pyproject.toml .
	@echo -e $(GREEN)"Looks good! 🎉"$(NOCOLOR)

# HELP: isort: [Static Analysis] Run isort
.PHONY: isort
isort:
	@echo "🚀 Running isort..."
	isort ${PACKAGE_FOLDER} tests -c --settings-path pyproject.toml
	@echo -e $(GREEN)"Looks good! 🎉"$(NOCOLOR)

# HELP: mypy-all: [Static Analysis] Run MyPy covering all available files
.PHONY: mypy-all
mypy-all:
	@echo "🚀 Running MyPy for the whole project..."
	mypy ${PACKAGE_FOLDER} tests --config-file pyproject.toml
	@echo -e $(GREEN)"Looks good! 🎉"$(NOCOLOR)

# HELP: format-code: [Static Analysis] Format code using available formatting tools
.PHONY: format-code
format-code:
	@echo "🚀 Formatting code..."
	isort ${PACKAGE_FOLDER} tests --settings-path pyproject.toml
	black --config pyproject.toml .
	@echo -e $(GREEN)"Done! ✅"$(NOCOLOR)

# HELP: radon-all: [Static Analysis] Run Radon on MDP on all folders, including tests
.PHONY: radon-all
radon-all:
	@echo "🚀 Running code analysis with Radon on all folders (including tests)..."
	@mkdir -p ./radon
	@# generate radon json report
	radon cc src tools tests static_analysis -j -O radon_cc_report.json
	@# get current complexity biggest offenders
	python -m static_analysis.radon_complexity top-most-complex radon_cc_report.json -n 10
	@echo -e $(GREEN)"Looks good! 🎉"$(NOCOLOR)

# HELP: radon: [Static Analysis] Run Radon on MDP (skip tests)
.PHONY: radon
radon:
	@echo "🚀 Running code analysis with Radon (skip tests)..."
	@mkdir -p ./radon
	@# generate radon json report
	radon cc src tools static_analysis -j -O radon_cc_report.json
	@# get current complexity biggest offenders
	python -m static_analysis.radon_complexity top-most-complex radon_cc_report.json -n 10
	@echo -e $(GREEN)"Looks good! 🎉"$(NOCOLOR)

# HELP: radon-compare-to-main: [Static Analysis] Run Radon and compare complexity of your branch to main
.PHONY: radon-compare-to-main
radon-compare-to-main: radon
	@echo "🚀 Comparing your branch to main..."
	git fetch
	python -m static_analysis.radon_complexity compare-complexity radon_cc_report.json --save-to-csv radon/radon_cc_report.csv
	@echo -e $(GREEN)"Looks good! 🎉"$(NOCOLOR)

#
# Tests
#

# HELP: tests: [Tests] Run all tests
.PHONY: tests
tests: unit-tests newline.1 integration-tests newline.2 e2e-tests newline.3 performance-tests newline.4 data-tests newline.5 tooling-tests

# HELP: unit-tests: [Tests] Run unit tests
.PHONY: unit-tests
unit-tests:
	@echo "🚀 Running unit tests..."
	python -m pytest --verbose \
		--fail-slow=50s \
		--timeout=600 \
		--durations=50 \
		--durations-min=5.0 \
		--strict-markers \
		--junit-xml test-reports/unit-tests/results.xml \
		--cov=$(PACKAGE_FOLDER) \
		tests/unit

# HELP: integration-tests: [Tests] Run integration tests
.PHONY: integration-tests
integration-tests:
	@echo "🚀 Running integration tests..."
	python -m pytest --verbose \
		--timeout=600 \
		--durations=10 \
		--durations-min=5.0 \
		--strict-markers \
		--junit-xml test-reports/unit-tests/results.xml \
		--cov=$(PACKAGE_FOLDER) \
		tests/integration

# HELP: e2e-tests: [Tests] Run end-to-end tests
.PHONY: e2e-tests
e2e-tests:
	@echo "🚀 Running e2e tests..."
	python -m pytest --verbose \
		--timeout=600 \
		--durations=10 \
		--durations-min=5.0 \
		--strict-markers \
		--junit-xml test-reports/unit-tests/results.xml \
		--cov=$(PACKAGE_FOLDER) \
		tests/e2e

#
# Tests
#



# HELP: docker-build: [Docker] Build Docker image
.PHONY: docker-build
docker-build: require-gitlab-envs docker-build-pip-conf
	@echo "🏗  Building Docker image"
	docker buildx build \
		--tag $(DOCKER_IMG_PATH):$(CI_COMMIT_SHA) \
		--tag $(DOCKER_IMG_PATH):$(CI_COMMIT_REF_SLUG) \
		--target ar-planes \
		--build-arg CI_COMMIT_SHA=$(CI_COMMIT_SHA) \
		--secret id=pip_conf,src=pip.conf \
		--cache-to type=inline \
		--cache-from $(DOCKER_IMG_PATH):$(CI_COMMIT_SHA) \
		--cache-from $(DOCKER_IMG_PATH):$(CI_COMMIT_REF_SLUG) \
		--cache-from $(DOCKER_IMG_PATH):latest \
		.

# HELP: docker-push: [Docker] Push Docker images to GCP Artifact Registry
.PHONY: docker-push
docker-push: require-gitlab-envs
	@echo "🏗  Pushing images to GCP Artifact Registry..."
	@# Push images with tags based on commit hash (CI_COMMIT_SHA) and branch name (CI_COMMIT_REF_SLUG)
	docker push $(DOCKER_IMG_PATH):$(CI_COMMIT_SHA)
	docker push $(DOCKER_IMG_PATH):$(CI_COMMIT_REF_SLUG)

# HELP: docker-pull-latest: [Docker] Pull the latest available Docker images from GCP Artifact Registry
.PHONY: docker-pull-latest
docker-pull-latest:
	@echo "🏗  Pulling the latest images from GCP Artifact Registry..."
	docker pull $(DOCKER_IMG_PATH):latest

# HELP: help: [Other] Help
.PHONY: help
help:
	@echo "🧑‍💻 Available commands:"
	@sed -n 's/^# HELP://p' ${MAKEFILE_LIST} | column -t -s ':' | sed -e 's/^/ - make/'

.PHONY: newline%
newline%:
	@echo ""

.PHONY: require-gitlab-envs
require-gitlab-envs:
ifndef CI_COMMIT_SHA
	$(error CI_COMMIT_SHA is undefined - GitLab env variable with commit hash)
endif
ifndef CI_COMMIT_REF_SLUG
	$(error CI_COMMIT_REF_SLUG is undefined - GitLab env variable with safe branch name)
endif
