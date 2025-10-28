Setup for Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

Run BE:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Test API

```bash
curl -X POST http://192.168.1.91:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Bão là gì?"}'
```
