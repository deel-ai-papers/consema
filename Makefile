.PHONY: help install \
        clean \
		setup \
        venv \
        dotenv \
        benchmarks \

MYPYTHON ?= /usr/bin/python3.12
PROJECT_DIR=$(PWD)
DATA_DIR ?= $(PROJECT_DIR)/benchmarks
# EXTERNAL_REQUIREMENTS = $(PROJECT_DIR)/external.requirements.txt

# === Use case (1) ===
#  - colorectal polyps data
#  - binary segmentation model (PraNet)
#  - we use data and precomputed predictions as provided in repo of Angelopoulos Gentle Intro
#    where they give a notebook with gdrive files to pranet model and data

# pre-computed predictionsL: pranet and polyp data as distributed by aangelopoulos
POLYPS_PRECOMPUTED = $(DATA_DIR)/polyps_precomputed
REPO_ROOT_URL = https://github.com/aangelopoulos/conformal-prediction
AANGELOPOULOS_REPO_NOTEBOOK_URL = $(REPO_ROOT_URL)/blob/67f506e4880e192ef9fc6a2de73e21b277f8c544/notebooks/tumor-segmentation.ipynb
AANGELOPOULOS_POLYPS_GDRIVE_ID = 1h7S6N_Rx7gdfO3ZunzErZy6H7620EbZK


# === Use case (2) ===
# UniverSeg segmentation models and some of their bechmarks datasets (WBC, OASIS)
# --- repo: pre-trained segmentation models, utils
UNIVERSEG_DIR = $(DATA_DIR)/universeg
UNIVERSEG_REPO_URL = https://github.com/JJGO/UniverSeg
UNIVERSEG_COMMIT_HASH = 833a0c34c65e38d675e21bd48ddec6797cc03259


help:
	@echo "make install [MYPYTHON=python3.XX] [DATA_DIR=/path/to/raw/data/files]"

install: setup clean venv dotenv

# useful to modify pyproject.toml dependencies + reinstall local package
local:
	@.venv/bin/python -m pip install --editable .[dev]

benchmarks: setup tumor_benchmark universeg

data: setup tumor_data universeg_data

venv:
	@echo "creating virtual environment (.venv)"
	@$(MYPYTHON) -m venv .venv
	@.venv/bin/python -m pip install --upgrade pip
	@.venv/bin/python -m pip install --editable .[dev]

dotenv:
	@echo "exporting local paths to .env file"
	@echo PROJECT_DIR=$(PROJECT_DIR) > .env
	@echo EXPERIMENTS_DIR=$(PROJECT_DIR)/experiments >> .env
	@echo DATA_DIR=$(DATA_DIR) >> .env
	@echo POLYPS_NPZ=$(POLYPS_PRECOMPUTED)/data/polyps >> .env
	@echo CONFIGS_EXPERIMENTS=$(PROJECT_DIR)/experiments/configs >> .env
	@echo RESULTS_DIR=$(PROJECT_DIR)/results >> .env

setup:
	@echo " --- preparing data directories"
	@mkdir -p $(DATA_DIR)
	@echo " --- created data directory: $(DATA_DIR)"
	@mkdir -p $(POLYPS_PRECOMPUTED)
	@echo " --- created precomputed data directory: $(POLYPS_PRECOMPUTED)"

# --- benchmark n.1: PraNet + Polyps data
# download data and pre-computed predictions via existing repo by aangelopoulos
tumor_benchmark: setup tumor_notebook tumor_data

tumor_notebook:
	@echo " --- downloading tumor-segmentation notebook (via aangelopoulos)"
	@echo " --- $(POLYPS_PRECOMPUTED)/tumor-segmentation.ipynb"
	wget $(AANGELOPOULOS_REPO_NOTEBOOK_URL)?raw=true -O $(POLYPS_PRECOMPUTED)/tumor-segmentation.ipynb

tumor_data:
	@echo " --- downloading polyps data (Angelopoulos)"
	.venv/bin/python -m gdown $(AANGELOPOULOS_POLYPS_GDRIVE_ID) -O $(POLYPS_PRECOMPUTED)/data_polyps.tar.xz
	@echo " --- unzipping polyps data_polyps"
	tar -xf $(POLYPS_PRECOMPUTED)/data_polyps.tar.xz -C $(POLYPS_PRECOMPUTED)
	@echo " --- removing all subdirectories except polyps"
	find $(POLYPS_PRECOMPUTED)/data -mindepth 1 -maxdepth 1 -type d ! -name 'polyps' -exec rm -rf {} +

# --- Benchmark n.2: UniverSeg + OASIS & WBC
universeg: setup
	@echo " === Cloning the UniverSeg repository"
	@if [ -d $(UNIVERSEG_DIR) ]; then \
		echo " --- dir: $(UNIVERSEG_DIR) already exists. Deleting it and re-clone repo."; \
		rm -rf $(UNIVERSEG_DIR); \
	fi
	@git clone $(UNIVERSEG_REPO_URL) $(UNIVERSEG_DIR);
	@echo " --- Checking out commit: $(UNIVERSEG_COMMIT_HASH)";
	@cd $(UNIVERSEG_DIR) && git checkout $(UNIVERSEG_COMMIT_HASH) --quiet;
	@echo " --- delete .git/ files";
	@rm -rf $(UNIVERSEG_DIR)/.git;
	@echo " --- Repository ready at $(UNIVERSEG_DIR)."
	# install repo as an editable package
	@touch $(UNIVERSEG_DIR)/example_data/__init__.py
	.venv/bin/python -m pip install -e benchmarks/universeg

clean:
	@if [ -d .venv ]; then \
		size=$$(du -sh .venv 2>/dev/null | awk '{print $1}'); \
		echo "Deleting: .venv/ ($$size)"; \
		read -p "ARE YOU SURE? [YES/no] " confirm; \
		if [ "$$confirm" = "YES" ]; then \
			echo "Deleting .venv and all its content"; \
			rm -rf .venv; \
		else \
		echo "make clean: canceled."; \
		fi; \
	else \
		echo ".venv does not exist or is already removed."; \
	fi

##
## === Not used in experiments for paper (now just using precomputed predictions):
##
## Polyps + PraNet:
##  - colorectal polyps data
##  - binary segmentation model (PraNet)

# pranet model, polyps data as distributed by pranet repo
POLYPS = $(DATA_DIR)/polyps_pranet

PRANET_DIR = $(POLYPS)/PraNet
# --- reference commit: 2024 October 16
PRANET_REPO_URL = https://github.com/DengPingFan/PraNet
PRANET_COMMIT_HASH = e79bf47c14ba98e6db639dcbbc7f29e259e27bde

# --- dataset: see PraNet README.md for google drive links
POLYPS_GDRIVE_ID_TEST ?= 1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao
POLYPS_GDRIVE_ID_TRAIN ?= 1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb
POLYPS_TEST_DATA=$(POLYPS)/data/polyps_test.zip
# --- Weights: "downloading pretrained weights and move it
# 				into snapshots/PraNet_Res2Net/PraNet-19.pth, 
PRANET_GDRIVE_WEIGHTS ?= 1lJv8XVStsp3oNKZHaSr42tawdMOq6FLP
PRANET_RESNET_WEIGHTS ?= 1FjXh_YG1hLGPPM6j-c8UxHcIWtzGGau5

polyps:
	@if [ -f $(POLYPS)/data ]; then \
		echo " --- downloading polyps TEST data"; \
		read -p "Polyps dataset already exists. Do you want to download and overwrite it? (y/n) " confirm; \
		if [ "$$confirm" != "y" ]; then \
			echo "Skipping download."; \
			exit 0; \
		fi; \
	fi; \
	mkdir -p $(POLYPS)/data; \
	.venv/bin/python -m gdown $(POLYPS_GDRIVE_ID_TEST) -O $(POLYPS_TEST_DATA); \
	echo " --- unzipping polyps data"; \
	unzip -o $(POLYPS_TEST_DATA) -d $(POLYPS)/data; \
	if [ -d $(POLYPS)/data/__MACOSX ]; then \
		rm -rf $(POLYPS)/data/__MACOSX; \
	fi

pranet:
	@echo " === Cloning the PraNet repository"
	@if [ -d $(PRANET_DIR) ]; then \
		echo " --- Repository already exists at $(PRANET_DIR). Skipping clone."; \
	else \
		git clone $(PRANET_REPO_URL) $(PRANET_DIR); \
		echo " --- Checking out commit $(PRANET_COMMIT_HASH)"; \
		cd $(PRANET_DIR) && git checkout $(PRANET_COMMIT_HASH); \
		echo " --- delete .git files"; \
		rm -rf $(PRANET_DIR)/.git; \
	fi
	@echo " --- Repository ready at $(PRANET_DIR)."
	@echo " --- Creating symlink from $(POLYPS)/data to data"
	@ln -sfn $(POLYPS)/data $(PRANET_DIR)/data

	@echo " --- Downloading PraNet pretrained weights"
	@mkdir -p $(PRANET_DIR)/snapshots/PraNet_Res2Net
	@if [ ! -f $(PRANET_DIR)/snapshots/PraNet_Res2Net/PraNet-19.pth ]; then \
		echo " --- PraNet-19.pth does not exist. Downloading"; \
		.venv/bin/python -m gdown $(PRANET_GDRIVE_WEIGHTS) -O $(PRANET_DIR)/snapshots/PraNet_Res2Net/PraNet-19.pth; \
	else \
		echo " --- PraNet-19.pth already exists. Skipping download."; \
	fi
	@echo " --- Downloading ResNet backbone weights"
	@mkdir -p $(PRANET_DIR)/snapshots/PraNet_Res2Net
	@if [ ! -f $(PRANET_DIR)/snapshots/PraNet_Res2Net/res2net50_v1b_26w_4s-3cf99910.pth ]; then \
		echo " --- res2net50_v1b_26w_4s-3cf99910.pth does not exist. Downloading"; \
		.venv/bin/python -m gdown $(PRANET_RESNET_WEIGHTS) -O $(PRANET_DIR)/snapshots/PraNet_Res2Net/res2net50_v1b_26w_4s-3cf99910.pth; \
	else \
		echo " --- res2net50_v1b_26w_4s-3cf99910.pth already exists. Skipping download."; \
	fi
