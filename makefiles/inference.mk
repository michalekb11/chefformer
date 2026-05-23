start-api:
	uvicorn app.main:app $(ARGS) --port 8000

start-ui:
	streamlit run app/ui.py