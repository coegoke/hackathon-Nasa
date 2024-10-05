# Use a specific version of Python 3.11.7
FROM python:3.11.7

# Set environment variables
ENV PYTHONBUFFERED True
ENV APP_HOME /app

# Set the working directory
WORKDIR $APP_HOME

# Copy the current directory contents into the container at /app
COPY . ./

RUN pip install --upgrade pip
# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Run the application using gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
