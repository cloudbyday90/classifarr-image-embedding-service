from fastapi import FastAPI, HTTPException

from .config import Settings
from .embedder import ImageEmbedder
from . import __version__
from .models import EmbedImageRequest, EmbedImageResponse, HealthResponse, ModelInfo


def create_app(embedder: ImageEmbedder | None = None) -> FastAPI:
    settings = Settings()
    app = FastAPI(title="Classifarr Image Embedding Service", version=__version__)

    app.state.embedder = embedder or ImageEmbedder(settings=settings)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            provider="local",
            default_model=settings.default_model
        )

    @app.get("/models", response_model=list[ModelInfo])
    def list_models():
        embedder_instance: ImageEmbedder = app.state.embedder
        return [
            ModelInfo(
                id=spec.name,
                name=spec.name,
                dims=spec.dims,
                image_size=spec.image_size
            )
            for spec in embedder_instance.list_models()
        ]

    @app.post("/embed-image", response_model=EmbedImageResponse)
    def embed_image(payload: EmbedImageRequest):
        embedder_instance: ImageEmbedder = app.state.embedder
        if not payload.image_url and not payload.image_base64:
            raise HTTPException(
                status_code=400,
                detail="image_url or image_base64 is required"
            )
        try:
            embedding, dims, provider, model_name, image_size = embedder_instance.embed(
                image_url=payload.image_url,
                image_base64=payload.image_base64,
                model=payload.model,
                normalize=payload.normalize,
                image_size=payload.image_size
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return EmbedImageResponse(
            embedding=embedding,
            dims=dims,
            provider=provider,
            model=model_name,
            image_size=image_size
        )

    return app


app = create_app()
