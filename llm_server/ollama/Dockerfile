FROM ollama/ollama:0.13.5

RUN apt-get update && \
    apt-get install -y --no-install-recommends jq && \
    rm -rf /var/lib/apt/lists/*

COPY llm-models.json /etc/ollama/llm-models.json

COPY init-models.sh /usr/local/bin/init-models.sh
RUN chmod +x /usr/local/bin/init-models.sh

ENTRYPOINT ["/usr/local/bin/init-models.sh"]
