"""
Unified Policy Wrapper for the ALTERA Hybrid Cognitive Scheduler.
Pairs:
  1. CognitiveDoctrine (Mode state machine + Candidate Action Mask)
  2. RuleBasedDwellController (Mode-to-Dwell Mapper)
  3. HybridLSTMScorer (Masked Neural Band Scorer)

Provides a standard reset() / predict(obs) interface compatible with AlterraEnv,
PolicyRunner, and benchmark runners.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from model.hybrid.doctrine import CognitiveDoctrine, DoctrineMode
from model.hybrid.dwell_controller import RuleBasedDwellController
from model.hybrid.scorers import HybridScorer


class HybridPolicy:
    """
    End-to-End Hybrid Scheduler Policy.
    Executes:
      Observation -> Doctrine Mode & Action Mask -> Neural Band Utility -> Dwell Control -> Environment Action
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        backbone: str = "lstm",
        num_bands: int = 128,
        action_mode: str | None = None,
        hopping_mode: str = "global",
        device: str | torch.device | None = None,
    ):
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.num_bands = num_bands
        self.hopping_mode = str(hopping_mode).lower()
        if action_mode is None:
            try:
                import yaml
                cfg_path = Path(__file__).resolve().parents[2] / "configs" / "default_config.yaml"
                if cfg_path.exists():
                    with open(cfg_path, "r") as f:
                        raw_c = yaml.safe_load(f)
                    action_mode = raw_c.get("environment", {}).get("action_mode", "relative")
            except Exception:
                action_mode = "relative"
        self.action_mode = action_mode or "relative"
        self.backbone = backbone.lower()

        self.doctrine = CognitiveDoctrine(num_bands=num_bands)
        self.dwell_controller = RuleBasedDwellController()
        self.scorer = HybridScorer(backbone=self.backbone, num_bands=num_bands).to(self.device)

        self.last_target_band = num_bands // 2
        self.last_mode = DoctrineMode.EXPLORE
        self.last_mask = np.ones(num_bands, dtype=bool)

        if checkpoint_path is not None:
            self.load(checkpoint_path)

    def reset(self) -> None:
        """Reset state at episode start."""
        self.doctrine.reset()
        self.last_target_band = self.num_bands // 2
        self.last_mode = DoctrineMode.EXPLORE
        self.last_mask = np.ones(self.num_bands, dtype=bool)

    def load(self, checkpoint_path: str | Path) -> None:
        """Loads trained weights into the LSTM scorer."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        if path.suffix.lower() == ".zip":
            import zipfile
            import io
            with zipfile.ZipFile(path, "r") as zf:
                pt_names = [n for n in zf.namelist() if n.endswith(".pt")]
                if not pt_names:
                    raise FileNotFoundError(f"No .pt weights found inside zip package: {path}")
                with zf.open(pt_names[0]) as f:
                    state_dict = torch.load(io.BytesIO(f.read()), map_location=self.device)
        else:
            state_dict = torch.load(path, map_location=self.device)

        # Handle backwards compatibility for keys like lstm.* -> seq_encoder.*
        adapted_dict = {}
        for k, v in state_dict.items():
            if k.startswith("lstm."):
                adapted_dict["seq_encoder." + k[len("lstm."):]] = v
            else:
                adapted_dict[k] = v
        self.scorer.load_state_dict(adapted_dict)
        self.scorer.eval()

    def save(self, checkpoint_path: str | Path) -> None:
        """Saves current weights of the LSTM scorer."""
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.scorer.state_dict(), path)

    def update_feedback(self, last_info: dict) -> None:
        """Incorporate step outcome feedback into doctrine state."""
        hit = bool(last_info.get("hit", False) or last_info.get("any_hit", False))
        band = int(last_info.get("band", self.doctrine.current_band))
        power_norm = float(last_info.get("mean_power_norm", 0.0))
        consec_hits = int(last_info.get("consecutive_hits", 0))
        t = int(last_info.get("t", 0))

        self.doctrine.update_state(band, hit, power_norm, consec_hits, t)

    def predict(
        self,
        obs: dict[str, np.ndarray],
        last_info: dict | None = None,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Executes hybrid decision step:
          1. Updates doctrine state machine
          2. Generates candidate action mask
          3. LSTM scores valid candidate bands
          4. Dwell controller selects duration
          5. Returns environment action [dir_idx, dwell_idx] or [band, dwell_idx]
        """
        receiver = obs.get("receiver")
        if receiver is not None and len(receiver) >= 1:
            self.doctrine.current_band = int(round(float(receiver[0]) * (self.num_bands - 1)))

        if last_info is not None:
            self.update_feedback(last_info)
        elif receiver is not None and len(receiver) >= 7:
            hit = bool(receiver[2] > 0.5)
            consec_hits = int(round(float(receiver[3]) * 5.0))
            power_norm = float(receiver[6])
            self.doctrine.update_state(
                self.doctrine.current_band, hit, power_norm, consec_hits, self.doctrine.t + 1
            )

        mode = self.doctrine.mode
        self.last_mode = mode

        band_tracks = obs.get("band_tracks")
        mask = self.doctrine.generate_action_mask(band_tracks, receiver)
        self.last_mask = mask

        if self.action_mode == "absolute" and mode == DoctrineMode.TRACK and self.doctrine.tracked_band is not None and self.doctrine.last_hit:
            # In TRACK with active emitter: lock dwell on target band (0 retune cost, maximum PRI capture)
            target_band = self.doctrine.tracked_band
        elif self.action_mode == "absolute" and mode == DoctrineMode.INVESTIGATE:
            # In INVESTIGATE: re-dwell on current candidate band
            target_band = self.doctrine.current_band
        else:
            target_band, _, _ = self.scorer.predict_action(
                obs=obs,
                mode=mode,
                mask=mask,
                deterministic=deterministic,
                device=self.device,
            )
            # In absolute mode during wideband search (EXPLORE / RELOCATE):
            # Fuse neural sequence prediction with EW information-gain utility (staleness + unscanned)
            if self.action_mode == "absolute" and band_tracks is not None and mode in (DoctrineMode.EXPLORE, DoctrineMode.RELOCATE):
                staleness = band_tracks[:, 4]
                unscanned = band_tracks[:, 6]
                threat = band_tracks[:, 0]
                with torch.no_grad():
                    seq_t = torch.tensor(obs["hit_miss_seq"], dtype=torch.float32, device=self.device).unsqueeze(0)
                    tracks_t = torch.tensor(band_tracks, dtype=torch.float32, device=self.device).unsqueeze(0)
                    rec_t = torch.tensor(receiver, dtype=torch.float32, device=self.device).unsqueeze(0)
                    mode_t = self.scorer.encode_doctrine_mode(mode, batch_size=1, device=self.device)
                    logits, _ = self.scorer.forward(seq_t, tracks_t, rec_t, mode_t)
                    raw_logits = logits[0].cpu().numpy()

                utility = raw_logits + 2.0 * staleness + 4.0 * unscanned + 1.2 * threat
                utility[~mask] = -1e9
                target_band = int(np.argmax(utility))

        self.last_target_band = target_band

        # Dwell duration selection
        conf = float(receiver[2]) if receiver is not None and len(receiver) > 2 else 0.0
        pwr = float(receiver[6]) if receiver is not None and len(receiver) > 6 else 0.0
        dwell_slots, dwell_idx = self.dwell_controller.select_dwell(
            mode=mode,
            confidence=conf,
            measured_power_norm=pwr,
            consecutive_hits=self.doctrine.consecutive_hits,
        )

        dir_idx = self.doctrine.select_relative_action(target_band)

        if self.hopping_mode == "local":
            # Non-hopping receiver (local sequential crawling +/- 1)
            if self.action_mode == "relative":
                return np.array([dir_idx, dwell_idx], dtype=np.int64)
            else:
                delta = [-1, 0, 1][dir_idx]
                crawl_band = int(max(0, min(self.num_bands - 1, self.doctrine.current_band + delta)))
                return np.array([crawl_band, dwell_idx], dtype=np.int64)
        else:
            # Instantaneous global hopping receiver
            if self.action_mode == "relative":
                return np.array([dir_idx, dwell_idx], dtype=np.int64)
            else:
                return np.array([target_band, dwell_idx], dtype=np.int64)
