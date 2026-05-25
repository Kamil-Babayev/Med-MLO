.PHONY: up test

up:
	docker compose up db -d && DATABASE_URL=postgresql://postgres:postgres@localhost:5432/medmlops uvicorn api.main:app --reload

test:
	pytest tests/ -v
