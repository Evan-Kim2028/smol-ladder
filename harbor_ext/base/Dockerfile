# Shared image for every smol-ladder task: the solver's Python stack, the grader's deps,
# and Node + Command Code so trials don't reinstall them.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates bash tmux \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pandas numpy scipy scikit-learn statsmodels matplotlib seaborn \
    tabulate openpyxl pyarrow math-verify
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs \
    && npm i -g command-code@1.66.0 && rm -rf /var/lib/apt/lists/*
ENV MPLBACKEND=Agg
WORKDIR /app
