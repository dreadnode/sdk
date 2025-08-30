FROM python:3.10-slim


# Update package lists and install git
RUN apt-get update && \
    apt-get install -y git

RUN useradd -m adsim
USER adsim
# Set the HOME environment variable to the non-root user's home directory
ENV HOME /home/adsim

RUN mkdir -p $HOME/.adsimulator && \
    git clone https://github.com/nicolas-carolo/adsimulator.git $HOME/.adsimulator/adsimulator

RUN cp -r $HOME/.adsimulator/adsimulator/data $HOME/.adsimulator
RUN ls -la && echo "Listing complete"

RUN pip install -r $HOME/.adsimulator/adsimulator/requirements.txt --user

WORKDIR $HOME/.adsimulator/adsimulator
RUN python setup.py install --user

RUN pip install neo4j
RUN pip install fire

COPY generate.py .

# CMD ["python", "generate.py"]