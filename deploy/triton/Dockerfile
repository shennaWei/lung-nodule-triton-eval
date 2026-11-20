# Triton server image with both ONNX graphs baked in.
#
# The .onnx files are intentionally not committed (see .gitignore). Export them
# first with `make export`, or download a release artefact into
# deploy/triton/model_repository/ before building:
#
#   docker build -f deploy/triton/Dockerfile -t lung-nodule-triton:local .

FROM nvcr.io/nvidia/tritonserver:24.08-py3

ENV MODEL_REPOSITORY=/models

COPY deploy/triton/model_repository /models

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=10 \
    CMD curl -f http://localhost:8000/v2/health/ready || exit 1

CMD ["tritonserver", \
     "--model-repository=/models", \
     "--strict-model-config=true", \
     "--http-port=8000", \
     "--grpc-port=8001", \
     "--metrics-port=8002"]
