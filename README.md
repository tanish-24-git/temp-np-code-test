```
project_root/
├── main.py                  # FastAPI app entrypoint
├── config/
│   └── settings.py          # Expanded config with DB, Celery, auth
├── models/
│   ├── requests.py          # Input models (added tune_hyperparams)
│   ├── responses.py         # Output models (updated TrainResponse)
│   └── db_models.py         # New: Pydantic DB models
├── services/
│   ├── file_service.py      # Updated with user_id, DB integration
│   ├── insight_service.py   # Minor error handling improvements
│   ├── ml_service.py        # Added tuning, versioning, DB saves
│   ├── preprocessing_service.py  # Enhanced errors
│   ├── db_service.py        # New: SQLAlchemy DB operations
│   └── model_registry.py    # New: Model registration/logic
├── utils/
│   ├── exceptions.py        # Enhanced handlers/messages
│   └── validators.py        # More descriptive errors
├── tasks.py                 # New: Celery tasks
├── celery_worker.sh         # New: Script to run Celery
└── tests/
    └── test_ml_service.py   # New: Example pytest file

```