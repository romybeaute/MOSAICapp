# Makefile

HF_SPACE_REMOTE ?= hf
HF_SPACE_URL    ?= https://huggingface.co/spaces/romybeaute/MOSAICapp
BRANCH_CLEAN    ?= hf-clean

.PHONY: deploy-hf add-hf-remote ensure-ignore

deploy-hf: add-hf-remote ensure-ignore
	@set -e; \
	echo ">> Creating orphan branch $(BRANCH_CLEAN)"; \
	git rev-parse --abbrev-ref HEAD > .git-previous-branch; \
	git checkout --orphan $(BRANCH_CLEAN); \
	echo ">> Clearing index"; \
	git rm -rf --cached . >/dev/null 2>&1 || true; \
	echo ">> Staging all files"; \
	git add -A; \
	echo ">> Committing clean deploy"; \
	git commit -m "Deploy to HF Space (clean orphan)"; \
	echo ">> Pushing to $(HF_SPACE_REMOTE) main (force)"; \
	git push $(HF_SPACE_REMOTE) $(BRANCH_CLEAN):main --force; \
	echo ">> Returning to previous branch"; \
	git checkout "$$(cat .git-previous-branch)"; \
	git branch -D $(BRANCH_CLEAN) || true; \
	rm -f .git-previous-branch; \
	echo "✅ Deploy completed."

add-hf-remote:
	@set -e; \
	if ! git remote | grep -qx "$(HF_SPACE_REMOTE)"; then \
		echo ">> Adding HF remote '$(HF_SPACE_REMOTE)' -> $(HF_SPACE_URL)"; \
		git remote add $(HF_SPACE_REMOTE) $(HF_SPACE_URL); \
	else \
		echo ">> HF remote '$(HF_SPACE_REMOTE)' already exists"; \
	fi

ensure-ignore:
	@set -e; \
	echo ">> Ensuring .gitignore excludes caches/binaries"; \
	touch .gitignore; \
	grep -qxF 'data/**/preprocessed/cache/' .gitignore || echo 'data/**/preprocessed/cache/' >> .gitignore; \
	grep -qxF 'eval/**/'                        .gitignore || echo 'eval/**/' >> .gitignore; \
	grep -qxF '**/__pycache__/'                 .gitignore || echo '**/__pycache__/' >> .gitignore; \
	grep -qxF '*.npy'                           .gitignore || echo '*.npy' >> .gitignore; \
	grep -qxF '*.npz'                           .gitignore || echo '*.npz' >> .gitignore; \
	grep -qxF '.DS_Store'                       .gitignore || echo '.DS_Store' >> .gitignore; \
	git add .gitignore; \
	git commit -m "Ensure .gitignore excludes caches/binaries" >/dev/null 2>&1 || true
	echo ">> .gitignore is up to date"

# End of Makefile