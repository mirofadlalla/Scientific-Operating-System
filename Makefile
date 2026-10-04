# Makefile — Scientific Operating System
# ─────────────────────────────────────────────────────────────────────────────
# Common targets for local development and EC2 production operations.
# Run `make help` to see all available commands.
#
# Prerequisites:
#   Local dev  : Docker, docker compose plugin, Python 3.11, uv
#   EC2 deploy : Docker, docker compose plugin, .env file, GHCR login
# ─────────────────────────────────────────────────────────────────────────────

# ── Configuration ─────────────────────────────────────────────────────────────
REGISTRY   := ghcr.io
REPO       := mirofadlalla/scientific-operating-system
IMAGE      := $(REGISTRY)/$(REPO)
# Use the short git SHA as the build tag; fall back to "dev" if git isn't available.
GIT_SHA    := $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
TAG        ?= $(GIT_SHA)

COMPOSE_DEV  := docker compose -f docker-compose.yml
COMPOSE_PROD := docker compose -f docker-compose.prod.yml

.DEFAULT_GOAL := help
.PHONY: help \
        dev dev-build dev-down dev-logs dev-ps \
        build push pull \
        deploy deploy-pull rollback \
        test test-backend test-frontend \
        lint \
        ec2-setup \
        clean prune

# ── Help ──────────────────────────────────────────────────────────────────────
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Local development (builds image from source) ──────────────────────────────
dev: dev-build  ## Build and start all services locally (foreground)
	$(COMPOSE_DEV) up

dev-build:  ## Build the local dev image
	$(COMPOSE_DEV) build

dev-up:  ## Start all services in the background
	$(COMPOSE_DEV) up -d

dev-down:  ## Stop and remove dev containers
	$(COMPOSE_DEV) down

dev-logs:  ## Tail logs from all dev containers
	$(COMPOSE_DEV) logs -f

dev-ps:  ## Show running dev containers
	$(COMPOSE_DEV) ps

# ── Image build & push to GHCR ────────────────────────────────────────────────
build:  ## Build the runtime image locally (PRELOAD_MODEL=0)
	docker build \
	  --target runtime \
	  --build-arg PRELOAD_MODEL=0 \
	  -t $(IMAGE):$(TAG) \
	  -t $(IMAGE):latest \
	  .

push: build  ## Build and push the image to GHCR
	@echo "Pushing $(IMAGE):$(TAG) …"
	docker push $(IMAGE):$(TAG)
	docker push $(IMAGE):latest
	@echo "Done. Pull with: docker pull $(IMAGE):$(TAG)"

pull:  ## Pull the latest image from GHCR (useful before deploy)
	docker pull $(IMAGE):latest

# ── EC2 production deployment ─────────────────────────────────────────────────
deploy: pull  ## Pull latest image and restart prod services on EC2
	$(COMPOSE_PROD) up -d --remove-orphans
	@echo ""
	@echo "Deployment complete. Running containers:"
	@$(COMPOSE_PROD) ps

deploy-pull:  ## Pull a specific tag and deploy  (make deploy-pull TAG=sha-abc1234)
	@[ -n "$(TAG)" ] || (echo "Usage: make deploy-pull TAG=<tag>"; exit 1)
	docker pull $(IMAGE):$(TAG)
	IMAGE_TAG=$(TAG) $(COMPOSE_PROD) up -d --remove-orphans

rollback:  ## Rollback to a previous image tag  (make rollback TAG=sha-abc1234)
	@[ -n "$(TAG)" ] || (echo "Usage: make rollback TAG=<tag>"; exit 1)
	@echo "Rolling back to $(IMAGE):$(TAG) …"
	docker pull $(IMAGE):$(TAG)
	sed -i "s|$(IMAGE):.*|$(IMAGE):$(TAG)|g" docker-compose.prod.yml
	$(COMPOSE_PROD) up -d --remove-orphans
	@echo "Rollback complete."

# ── Testing ───────────────────────────────────────────────────────────────────
test: test-backend test-frontend  ## Run all tests

test-backend:  ## Run Python backend tests (requires uv)
	uv run --no-sync pytest tests/ -v --tb=short

test-frontend:  ## Run frontend lint + build (requires Node 22+)
	cd frontend && npm ci && npm run lint && npm run build

# ── Lint ──────────────────────────────────────────────────────────────────────
lint:  ## Lint Python code with ruff
	uv run --no-sync ruff check app/ tests/

# ── EC2 first-time setup helper ───────────────────────────────────────────────
ec2-setup:  ## Print the one-time EC2 setup instructions
	@echo ""
	@echo "══════════════════════════════════════════════════════════"
	@echo "  EC2 first-time setup"
	@echo "══════════════════════════════════════════════════════════"
	@echo ""
	@echo "1. Install Docker:"
	@echo "   sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin"
	@echo "   sudo usermod -aG docker ubuntu && newgrp docker"
	@echo ""
	@echo "2. Log in to GHCR (needs a PAT with packages:read scope):"
	@echo "   echo \$$CR_PAT | docker login ghcr.io -u <github-username> --password-stdin"
	@echo ""
	@echo "3. Clone the repo:"
	@echo "   git clone https://github.com/$(REPO).git && cd Scientific-Operating-System"
	@echo ""
	@echo "4. Configure environment:"
	@echo "   cp .env.example .env && nano .env"
	@echo ""
	@echo "5. Set up TLS certificates (free via Certbot):"
	@echo "   sudo apt install certbot"
	@echo "   sudo certbot certonly --standalone -d api.yourdomain.com"
	@echo "   sudo cp /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem nginx/certs/cert.pem"
	@echo "   sudo cp /etc/letsencrypt/live/api.yourdomain.com/privkey.pem   nginx/certs/key.pem"
	@echo "   sudo chown \$$USER:\$$USER nginx/certs/*.pem && chmod 600 nginx/certs/key.pem"
	@echo ""
	@echo "6. Deploy:"
	@echo "   make deploy"
	@echo ""
	@echo "Subsequent updates are automatic via Watchtower (every 5 min)"
	@echo "or manually with: make deploy"
	@echo ""

# ── Housekeeping ──────────────────────────────────────────────────────────────
clean:  ## Stop containers and remove local images
	$(COMPOSE_DEV) down --rmi local -v 2>/dev/null || true
	docker rmi $(IMAGE):$(TAG) $(IMAGE):latest 2>/dev/null || true

prune:  ## Remove all unused Docker objects (images, containers, volumes, networks)
	docker system prune -af --volumes
