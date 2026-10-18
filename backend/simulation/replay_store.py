"""Bounded, dependency-free replay cache; MongoDB remains the durable store."""
from collections import OrderedDict, deque
from threading import RLock


class ReplayStore:
    def __init__(self, max_runs=3, max_events=2000):
        self.max_runs = max_runs
        self.max_events = max_events
        self.lock = RLock()
        self.runs = OrderedDict()

    def begin(self, recording_id, metadata):
        with self.lock:
            self.runs[recording_id] = {
                "metadata": {**metadata, "recording_id": recording_id},
                "events": deque(maxlen=self.max_events),
            }
            while len(self.runs) > self.max_runs:
                self.runs.popitem(last=False)

    def append(self, recording_id, event):
        with self.lock:
            run = self.runs.get(recording_id)
            if run is not None:
                run["events"].append(event)

    def finish(self, recording_id, error=None):
        with self.lock:
            run = self.runs.get(recording_id)
            if run is not None:
                run["metadata"].update(status="error" if error else "stopped")

    def list_runs(self):
        with self.lock:
            return [{**run["metadata"], "retained_events": len(run["events"])}
                    for run in reversed(list(self.runs.values()))]

    def events(self, recording_id):
        with self.lock:
            run = self.runs.get(recording_id)
            return list(run["events"]) if run else []
