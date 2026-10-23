"""Classify vLLM 0.14.1 scheduler work; no torch/vLLM import is needed here."""
from dataclasses import dataclass


@dataclass(frozen=True)
class StepPhase:
    phase: str
    requests: int
    scheduled_tokens: int
    prefill_tokens: int
    decode_tokens: int
    unknown_requests: int
    request_ids: tuple

    def fields(self):
        return dict(phase=self.phase, requests=self.requests,
                    scheduled_tokens=self.scheduled_tokens,
                    prefill_tokens=self.prefill_tokens, decode_tokens=self.decode_tokens,
                    unknown_requests=self.unknown_requests, request_ids=self.request_ids)


class StepClassifier:
    def __init__(self):
        self.prompt_lengths = {}

    def classify(self, output):
        for request_id in getattr(output, "finished_req_ids", ()):
            self.prompt_lengths.pop(request_id, None)
        positions = {}
        for request in output.scheduled_new_reqs:
            ids = request.prompt_token_ids
            embeds = getattr(request, "prompt_embeds", None)
            length = len(ids) if ids is not None else (int(embeds.shape[0]) if embeds is not None else None)
            if length is not None:
                self.prompt_lengths[request.req_id] = length
            positions[request.req_id] = request.num_computed_tokens
        cached = output.scheduled_cached_reqs
        positions.update(zip(cached.req_ids, cached.num_computed_tokens))
        scheduled = {key: value for key, value in output.num_scheduled_tokens.items() if value > 0}
        prefill = decode = unknown = 0
        for request_id, count in scheduled.items():
            length, position = self.prompt_lengths.get(request_id), positions.get(request_id)
            if length is None or position is None or position < 0:
                unknown += 1
                continue
            prompt_part = min(count, max(0, length-position))
            prefill += prompt_part
            decode += count-prompt_part
        phase = ("unknown" if unknown else "mixed" if prefill and decode else
                 "prefill" if prefill else "decode" if decode else "idle")
        return StepPhase(phase, len(scheduled), sum(scheduled.values()), prefill,
                         decode, unknown, tuple(scheduled))
