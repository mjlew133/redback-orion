# Backend API Gateway — Project Orion

FastAPI gateway that connects the frontend to the player tracking and crowd monitoring microservices.

## Prerequisites

- Python 3.11+
- PostgreSQL (running locally)

## Setup

**1. Install dependencies**

```bash
cd backend
pip install -r requirements.txt
```

**2. Create the `.env` file**

Create `backend/.env` with the following content:

```env
# Service URLs
PLAYER_SERVICE_URL=http://localhost:8001
CROWD_SERVICE_URL=http://localhost:8002
BACKEND_PORT=8000
UPLOAD_DIR=uploads

# Mock toggles — set to false when the real service is running
USE_MOCK_PLAYER=true
USE_MOCK_CROWD=false

# PostgreSQL
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/orion_db

# Auth
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=7

# Optional
DEBUG=true
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE=524288000
```

**3. Set up the database**

Create the database in PostgreSQL:

```sql
CREATE DATABASE orion_db;
```

Then run the app once — SQLAlchemy will create the tables automatically on first start (or run Alembic migrations if configured).

**4. Create the uploads folder**

```bash
mkdir uploads
```

## Run

```bash
cd backend
uvicorn app.main:app --port 8000 --reload
```

The gateway will be available at:

- API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`

## Mock vs Real Services

The gateway can run with mock or real player/crowd services independently:

| Flag | `true` | `false` |
|---|---|---|
| `USE_MOCK_PLAYER` | Returns hardcoded player metrics | Calls player service on port 8001 |
| `USE_MOCK_CROWD` | Returns hardcoded crowd metrics | Calls crowd service on port 8002 |

Set these in `.env`. No code changes needed to switch.

## API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/auth/register` | No | Create a new user account |
| POST | `/auth/login` | No | Get a JWT token |
| GET | `/auth/me` | JWT | Current user info |
| POST | `/upload` | JWT | Upload video for processing |
| GET | `/status/{job_id}` | JWT | Poll job status |
| GET | `/jobs` | JWT | List past jobs (paginated) |
| GET | `/jobs/{job_id}` | JWT | Full job details |
| POST | `/jobs/{job_id}/retry` | JWT | Retry failed service(s) |
| DELETE | `/jobs/{job_id}` | JWT | Delete a job |
| GET | `/jobs/{job_id}/heatmap` | JWT | Stream heatmap image from crowd service |
| GET | `/health` | Admin | Check gateway + service connectivity |

## Running the Crowd Service

The crowd service must be started separately on port 8002 before the gateway can call it with `USE_MOCK_CROWD=false`.

**1. Install crowd dependencies**

```bash
cd Crowd_Monitoring/2026_T1
pip install -r requirements.txt
```

**2. Start the crowd service**

```bash
cd Crowd_Monitoring/2026_T1
uvicorn shared.services.main:app --port 8002 --reload
```

The crowd service will be available at:

- Swagger docs: `http://localhost:8002/`
- Demo UI: `http://localhost:8002/demo`

Once running, set `USE_MOCK_CROWD=false` in `backend/.env` and restart the gateway.

---

## Running Tests

```bash
cd backend
pytest tests/
```

Tests use mocked services — no real player or crowd service needed.
