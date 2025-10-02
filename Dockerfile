# 1. কোন Python ব্যবহার করবে (3.10 fixed)
FROM python:3.10-slim

# 2. কাজ করার ডিরেক্টরি সেট
WORKDIR /app

# 3. requirements install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. পুরো কোড কপি
COPY . .

# 5. Run command (gunicorn দিয়ে app চালাবে)
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:$PORT"]
