CONFIG ?= config/alexa.yaml
STAGE ?= all
U := scripts/unraid.sh

.PHONY: image sync test smoke run logs status pull publish test-local

# runtime image; only needed again when the Dockerfile changes
image: sync
	$(U) build

sync:
	$(U) sync

test: sync
	$(U) test

# tiny end-to-end run: proves the pipeline, not the model
smoke: sync
	$(U) run --config config/smoke.yaml all

# one stage (or all) of the real run, e.g. make run STAGE=evaluate
run: sync
	$(U) run --config $(CONFIG) $(STAGE)

logs:
	$(U) logs

status:
	$(U) status

pull:
	$(U) pull

# copy the exported model and its evaluation into models/ for committing
publish: pull
	cp dist/alexa/alexa.json dist/alexa/alexa.tflite models/
	cp dist/alexa/report.md models/alexa-evaluation.md

# only if a local env exists; data and training live on unraid
test-local:
	uv run pytest -q
