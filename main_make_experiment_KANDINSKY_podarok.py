from dataclasses import dataclass
from datetime import datetime
import os
import pickle
import sys
from typing import List, Optional, Sequence, Tuple

import mne
import nest_asyncio
import numpy as np
import pylsl
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QFont, QKeyEvent, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import AsyncKandinsky as kandinsky
from lsl_inlet import LSLInlet

VIS_EPOCH_TMIN = -0.2
VIS_EPOCH_TMAX = 1.0
METRIC_WINDOW_TMIN = 0.1
METRIC_WINDOW_TMAX = 0.7


def trca_fit(epochs_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit TRCA spatial filters for epochs with shape (n_trials, n_channels, n_times)."""
    if epochs_data.ndim != 3:
        raise ValueError("epochs_data must have shape (n_trials, n_channels, n_times)")

    n_trials, n_chan, _ = epochs_data.shape
    if n_trials < 2:
        raise ValueError("TRCA needs at least two trials per class")

    x_centered = epochs_data - epochs_data.mean(axis=2, keepdims=True)

    s_matrix = np.zeros((n_chan, n_chan), dtype=np.float64)
    for i in range(n_trials):
        xi = x_centered[i]
        for j in range(i + 1, n_trials):
            xj = x_centered[j]
            s_matrix += xi @ xj.T + xj @ xi.T

    q_matrix = np.zeros((n_chan, n_chan), dtype=np.float64)
    for k in range(n_trials):
        xk = x_centered[k]
        q_matrix += xk @ xk.T

    reg = 1e-6 * np.trace(q_matrix) / max(n_chan, 1)
    q_reg = q_matrix + reg * np.eye(n_chan)

    m_matrix = np.linalg.pinv(q_reg) @ s_matrix
    eigenvals, eigenvecs = np.linalg.eig(m_matrix)

    idx = np.argsort(np.real(eigenvals))[::-1]
    eigenvals = np.real(eigenvals[idx])
    eigenvecs = np.real(eigenvecs[:, idx])

    return eigenvecs, eigenvals, s_matrix, q_reg


def _cross_cov_power(trials_a: np.ndarray, trials_b: Optional[np.ndarray] = None) -> float:
    """Average absolute cross-covariance power across trial pairs.

    `trials_a` and `trials_b` are expected as (n_trials, n_times) and are centered per trial.
    """
    if trials_a.size == 0:
        return 0.0

    a = trials_a - trials_a.mean(axis=1, keepdims=True)

    if trials_b is None:
        if len(a) < 2:
            return 0.0
        values = []
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                values.append(abs(float(np.dot(a[i], a[j]))))
        return float(np.mean(values)) if values else 0.0

    b = trials_b - trials_b.mean(axis=1, keepdims=True)
    if b.size == 0:
        return 0.0

    values = []
    for i in range(len(a)):
        for j in range(len(b)):
            values.append(abs(float(np.dot(a[i], b[j]))))
    return float(np.mean(values)) if values else 0.0


@dataclass
class ProtocolBlock:
    name: str
    duration: float
    message: str
    code: int


@dataclass
class StimEvent:
    class_idx: int
    onset_lsl: float


class ClosableLabel(QtWidgets.QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setAlignment(QtCore.Qt.AlignCenter)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == QtCore.Qt.Key_Space:
            self.close()
        else:
            super().keyPressEvent(event)


class ShapeLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.shape_params = None

    def setImage(self, kind: str, stim: int = 0):
        self.shape_params = (kind, stim)
        if kind != "clear":
            self.setPixmap(QPixmap(kind))
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.shape_params is None:
            return
        kind, _ = self.shape_params
        if kind == "clear":
            painter = QPainter(self)
            painter.eraseRect(self.rect())


class ProtocolEditor(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.protocol_blocks: List[ProtocolBlock] = []
        self.streams = []
        self.inlet_info = None
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.path_to_pics = "elements"
        self.all_stages_scenes = [["candy", "bear", "book", "guitar", "skis"]]
        self.colors = ["blue", "orange", "green", "violet", "red", "lightblue", "yellow", "navy"]

        self.stim_mode = "comp"
        self.stim_duration = 0.05
        self.trial_duration = 0.4
        self.trials_per_class = 20

        self.lsl_button = QPushButton("UPD lsl streams")
        self.lsl_button.clicked.connect(self.upd_lsl_streams)

        self.lsl_combobox = QComboBox(self)
        self.lsl_combobox.currentIndexChanged.connect(self.choose_lsl)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.onStartButtonClicked)

        nest_asyncio.apply()
        self.model = kandinsky.FusionBrainApi(kandinsky.ApiWeb("np_fedosov@list.ru", "2086schA1"))

    def upd_lsl_streams(self):
        self.lsl_combobox.clear()
        self.streams = pylsl.resolve_streams()
        for stream in self.streams:
            self.lsl_combobox.addItem(stream.name())
        if self.streams:
            self.inlet_info = self.streams[0]

    def choose_lsl(self, idx: int):
        if 0 <= idx < len(self.streams):
            self.inlet_info = self.streams[idx]

    def _show_instruction(self, text: str):
        label = ClosableLabel(text)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont("Gost", 26))
        label.showMaximized()
        while label.isVisible():
            QCoreApplication.processEvents()

    def _collect_stimulus_paths(self, scenes: Sequence[str]) -> Tuple[List[List[str]], List[List[str]]]:
        regular, gift = [], []
        for scene in scenes:
            src = os.path.join(self.path_to_pics, scene)
            src_gift = os.path.join(self.path_to_pics, f"{scene}_gift")
            regular.append([os.path.join(src, f) for f in os.listdir(src)])
            gift.append([os.path.join(src_gift, f) for f in os.listdir(src_gift)])
        return regular, gift

    @staticmethod
    def _choose_channel_groups(channel_names: Sequence[str]) -> Tuple[List[int], List[int]]:
        photokeywords = ("photo", "phot", "stim", "trigger", "trig", "diode")
        photo_indices = [
            idx for idx, name in enumerate(channel_names) if any(key in name.lower() for key in photokeywords)
        ]
        eeg_indices = [idx for idx in range(len(channel_names)) if idx not in photo_indices]
        if not eeg_indices:
            eeg_indices = list(range(len(channel_names)))
            photo_indices = []
        return eeg_indices, photo_indices

    def _build_protocol(self, scenes: Sequence[str]) -> List[ProtocolBlock]:
        blocks = [ProtocolBlock("non_pause", 3.0, "+", -2)]
        for _ in range(self.trials_per_class):
            order = np.random.permutation(len(scenes))
            for cls in order:
                blocks.append(ProtocolBlock(scenes[cls], self.trial_duration, "", int(cls)))
        blocks.append(ProtocolBlock("PAUSE", 1.0, "+", -1))
        return blocks

    def _run_protocol(
        self,
        inlet: LSLInlet,
        protocol_blocks: Sequence[ProtocolBlock],
        image_paths: Sequence[Sequence[str]],
    ) -> Tuple[np.ndarray, np.ndarray, List[StimEvent], List[Tuple[float, float]]]:
        data_chunks: List[np.ndarray] = []
        ts_chunks: List[np.ndarray] = []
        stim_events: List[StimEvent] = []
        gaps: List[Tuple[float, float]] = []

        self.stim_label = ShapeLabel("+")
        self.stim_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stim_label.setAlignment(Qt.AlignCenter)
        self.stim_label.setFont(QFont("Gost", 50))
        self.stim_label.showMaximized()

        prev_last_ts = None
        nominal_step = 1.0 / float(inlet.srate)

        for block in protocol_blocks:
            start_time = pylsl.local_clock()
            is_stim = block.code >= 0
            img_path = None

            if is_stim:
                img_path = np.random.choice(image_paths[block.code])
                self.stim_label.clear()
                self.stim_label.setImage(str(img_path), stim=1)
                QCoreApplication.processEvents()
                stim_events.append(StimEvent(class_idx=block.code, onset_lsl=pylsl.local_clock()))
            else:
                self.stim_label.clear()
                self.stim_label.setImage("clear", stim=0)
                self.stim_label.setText(block.message)
                QCoreApplication.processEvents()

            while True:
                now = pylsl.local_clock()
                elapsed = now - start_time

                chunk, t_stamp = inlet.get_next_chunk()
                if chunk is not None and len(chunk):
                    chunk_np = np.asarray(chunk)
                    ts_np = np.asarray(t_stamp)
                    data_chunks.append(chunk_np)
                    ts_chunks.append(ts_np)
                    if prev_last_ts is not None and len(ts_np):
                        dt = ts_np[0] - prev_last_ts
                        if dt > 1.5 * nominal_step:
                            gaps.append((float(prev_last_ts), float(ts_np[0])))
                    if len(ts_np):
                        prev_last_ts = float(ts_np[-1])

                QCoreApplication.processEvents()
                if elapsed >= block.duration:
                    break

        if not data_chunks:
            raise RuntimeError("No LSL samples were received during the protocol.")

        data = np.vstack(data_chunks)
        timestamps = np.concatenate(ts_chunks)
        return data, timestamps, stim_events, gaps

    def _classify_with_trca_ratio(
        self,
        raw: mne.io.RawArray,
        events_by_sample: np.ndarray,
        list_of_scenes: Sequence[str],
        event_id: dict,
    ) -> Tuple[int, np.ndarray]:
        epochs = mne.Epochs(
            raw,
            events=events_by_sample,
            event_id=event_id,
            tmin=VIS_EPOCH_TMIN,
            tmax=VIS_EPOCH_TMAX,
            baseline=None,
            preload=True,
            verbose=False,
        )
        metric_time_mask = (epochs.times >= METRIC_WINDOW_TMIN) & (epochs.times <= METRIC_WINDOW_TMAX)
        if not np.any(metric_time_mask):
            raise RuntimeError("Metric window mask is empty. Check epoch and metric window bounds.")

        class_epochs = []
        for scene in list_of_scenes:
            scene_data = epochs[scene].get_data()[:, :, metric_time_mask]
            if len(scene_data) < 2:
                raise RuntimeError(
                    f"Not enough epochs for class '{scene}' for TRCA (need >=2, got {len(scene_data)})."
                )
            class_epochs.append(scene_data)

        scores = np.zeros(len(list_of_scenes), dtype=np.float64)
        for class_idx, cls_data in enumerate(class_epochs):
            w, _, _, _ = trca_fit(cls_data)
            filter_vec = w[:, 0]

            cls_proj = np.einsum("tcn,c->tn", cls_data, filter_vec)
            other_classes = [class_epochs[i] for i in range(len(class_epochs)) if i != class_idx]
            other_data = np.concatenate(other_classes, axis=0)
            other_proj = np.einsum("tcn,c->tn", other_data, filter_vec)

            p_target = _cross_cov_power(cls_proj)
            p_other = _cross_cov_power(cls_proj, other_proj)
            scores[class_idx] = p_target / (p_other + 1e-12)

        winner_idx = int(np.argmax(scores))
        return winner_idx, scores

    def _build_unique_events(
        self,
        sample_timestamps: np.ndarray,
        stim_events: Sequence[StimEvent],
        max_samples: int,
    ) -> np.ndarray:
        """Map LSL event timestamps to unique sample indices for MNE Epochs."""
        event_samples = []
        last_sample = -1
        dropped_events = 0

        for ev in stim_events:
            sample_idx = int(np.searchsorted(sample_timestamps, ev.onset_lsl, side="left"))
            if sample_idx >= max_samples:
                sample_idx = max_samples - 1

            if sample_idx <= last_sample:
                sample_idx = last_sample + 1

            if sample_idx >= max_samples:
                dropped_events += 1
                continue

            event_samples.append([sample_idx, 0, ev.class_idx + 1])
            last_sample = sample_idx

        if not event_samples:
            raise RuntimeError("No valid stimulus events remained after aligning events to EEG samples.")

        if dropped_events:
            print(f"Warning: dropped {dropped_events} stimulus events because they fell beyond the EEG sample range.")

        return np.asarray(event_samples, dtype=int)

    def onStartButtonClicked(self):
        if self.inlet_info is None:
            raise RuntimeError("Select an LSL stream first.")

        instruction = (
            "Какой подарок вы бы хотели получить на Новый Год?\n"
            "Загадайте что-то из списка:\n\n\n"
            "Книга\t\tПлюшевый медведь\t\tКоробка конфет\t\tГитара\t\tКоньки\n\n\n"
            "Считайте про себя каждый раз, когда видите загаданный предмет.\n"
            "Нажмите ПРОБЕЛ, когда будете готовы начать."
        )
        self._show_instruction(instruction)

        for stage_scenes in self.all_stages_scenes:
            list_of_names, list_of_names_gift = self._collect_stimulus_paths(stage_scenes)
            protocol_blocks = self._build_protocol(stage_scenes)

            timestamp_str = datetime.strftime(datetime.now(), "%m-%d_%H-%M-%S")
            results_path = f"results/baseline_experiment_{timestamp_str}/"

            exp_settings = {
                "exp_name": "Baseline",
                "lsl_stream_name": self.inlet_info.name(),
                "max_buflen": 5,
                "max_chunklen": 1,
                "results_path": results_path,
                "stim_duration": self.stim_duration,
                "trial_duration": self.trial_duration,
                "trials_per_class": self.trials_per_class,
            }

            inlet = LSLInlet(exp_settings)
            inlet.srate = inlet.get_frequency()
            xml_info = inlet.info_as_xml()
            channel_names = inlet.get_channels_labels()
            exp_settings["channel_names"] = channel_names
            exp_settings["srate"] = int(round(inlet.srate))

            data, sample_timestamps, stim_events, detected_gaps = self._run_protocol(
                inlet=inlet,
                protocol_blocks=protocol_blocks,
                image_paths=list_of_names,
            )
            inlet.disconnect()

            eeg_indices, photo_indices = self._choose_channel_groups(channel_names)
            eeg_names = [channel_names[i] for i in eeg_indices]
            eeg_data = data[:, eeg_indices]

            info = mne.create_info(eeg_names, exp_settings["srate"], ch_types="eeg")
            raw = mne.io.RawArray(eeg_data.T, info, verbose=False)
            raw.notch_filter([50.0, 100.0], verbose=False)
            raw.filter(0.3, 20.0, verbose=False)

            event_ids = {scene: idx + 1 for idx, scene in enumerate(stage_scenes)}
            events_array = self._build_unique_events(
                sample_timestamps=sample_timestamps,
                stim_events=stim_events,
                max_samples=len(sample_timestamps),
            )
            winner_idx, ratio_scores = self._classify_with_trca_ratio(
                raw=raw,
                events_by_sample=events_array,
                list_of_scenes=stage_scenes,
                event_id=event_ids,
            )

            os.makedirs(results_path, exist_ok=True)
            with open(os.path.join(results_path, "data.pickle"), "wb") as file:
                pickle.dump(
                    {
                        "eeg": data,
                        "timestamps": sample_timestamps,
                        "stim_events": [ev.__dict__ for ev in stim_events],
                        "detected_gaps": detected_gaps,
                        "channel_names": channel_names,
                        "eeg_indices": eeg_indices,
                        "photo_indices": photo_indices,
                        "xml_info": xml_info,
                        "exp_settings": exp_settings,
                        "trca_ratio_scores": ratio_scores,
                        "winner_idx": winner_idx,
                        "winner_label": stage_scenes[winner_idx],
                    },
                    file=file,
                )

            self._show_instruction("Нажмите пробел, чтобы получить подарок")
            gift_class = winner_idx
            gift_pic = np.random.choice(list_of_names_gift[gift_class])
            gift_label = ClosableLabel("")
            gift_label.setPixmap(QPixmap(str(gift_pic)))
            gift_label.showMaximized()
            while gift_label.isVisible():
                QCoreApplication.processEvents()

        self.stim_label.close()
        print("Finished")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setFixedSize(520, 285)
        self.protocol_editor = ProtocolEditor()

        container = QWidget()
        protocol_layout = QVBoxLayout()
        protocol_layout.addWidget(self.protocol_editor)
        protocol_layout.addWidget(self.protocol_editor.start_button)
        protocol_layout.addWidget(self.protocol_editor.lsl_button)
        protocol_layout.addWidget(self.protocol_editor.lsl_combobox)
        protocol_layout.addStretch(1)
        container.setLayout(protocol_layout)
        self.setCentralWidget(container)


app = QApplication([])
app.setWindowIcon(QtGui.QIcon("mind.png"))
app.setApplicationName("mindReader")
w = MainWindow()
w.show()

app.exec_()
