from dataclasses import dataclass


@dataclass
class DisplayState:
    inline_index: int = 0
    crossline_index: int = 0
    z_index: int = 0
    input_clip: float = 0.5
    output_clip: float = 0.5
    overlay_threshold: float = 0.5
    overlay_alpha: float = 0.6
    similarity_mode: str = "cosine"
    output_loaded: bool = False
    overlay_preview_mean: float = 0.0
    overlay_preview_std: float = 0.0

    def update_from_snapshot(self, snapshot: dict) -> None:
        self.inline_index = int(snapshot.get("inline_index", self.inline_index))
        self.crossline_index = int(snapshot.get("crossline_index", self.crossline_index))
        self.z_index = int(snapshot.get("z_index", self.z_index))
        self.input_clip = float(snapshot.get("input_clip", self.input_clip))
        self.output_clip = float(snapshot.get("output_clip", self.output_clip))
        self.overlay_threshold = float(snapshot.get("overlay_threshold", self.overlay_threshold))
        self.overlay_alpha = float(snapshot.get("overlay_alpha", self.overlay_alpha))
        self.similarity_mode = str(snapshot.get("similarity_mode", self.similarity_mode))
        self.output_loaded = bool(snapshot.get("output_loaded", self.output_loaded))
        self.overlay_preview_mean = float(
            snapshot.get("overlay_preview_mean", self.overlay_preview_mean)
        )
        self.overlay_preview_std = float(
            snapshot.get("overlay_preview_std", self.overlay_preview_std)
        )
