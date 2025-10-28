Run BE:

```bash
 uvicorn main:app --reload
```

Test API

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Bão là gì?"}'
```
