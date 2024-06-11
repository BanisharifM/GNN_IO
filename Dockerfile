FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /usr/src/app
COPY . .

#CMD ["python", "src/gnn_v1.py"]
CMD ["python", "src/csv_to_json.py"]
