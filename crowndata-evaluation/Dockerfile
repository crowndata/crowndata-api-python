# Use the official Python 3.11-slim image as the base
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /crowndata-evaluation

# Copy the contents of the current directory on your host to the container
COPY . /crowndata-evaluation

RUN pip install --upgrade pip

# Install Poetry
RUN pip install poetry

# Install dependencies without dev dependencies
RUN poetry install --no-dev

# Command to run the application
CMD ["poetry", "run", "uvicorn", "crowndata_evaluation.main:app", "--host", "0.0.0.0", "--port", "8000"]