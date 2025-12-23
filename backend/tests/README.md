#To run all tests:

## Backend tests
cd backend
pytest -v --cov=. --cov-report=html

## MLOps tests  
cd mlops
pytest -v

## With specific markers
pytest -v -m "not slow"  # Skip slow tests
pytest -v -m "unit"      # Run only unit tests
