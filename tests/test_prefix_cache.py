"""Cache boundaries with fakes; optional real CPU Qwen3 greedy equivalence."""

from contextlib import nullcontext
import importlib.util
from time import monotonic
from types import SimpleNamespace
import unittest

from src.agent.local_model import LocalPlanner, _GENERATING, _PrefixCache, _prefix_input_key


class Tensor(list):
    def __init__(self, rows, *, device="cpu", dtype="int64"):
        super().__init__(rows)
        self.device, self.dtype = device, dtype

    @property
    def shape(self):
        return (len(self), len(self[0]))

    def to(self, device):
        return Tensor(self, device=str(device), dtype=self.dtype)


class Batch(dict):
    def to(self, device):
        return Batch({name: value.to(device) for name, value in self.items()})


class KV:
    def __init__(self, length, *, device="cpu", dtype="float32", heads=2):
        self.shape = (1, heads, length, 8)
        self.device, self.dtype = device, dtype


class Cache:
    is_croppable = True

    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.layers = [SimpleNamespace(keys=KV(len(tokens)), values=KV(len(tokens)), is_croppable=True)]
        self.crops = []

    def get_seq_length(self):
        return len(self.tokens)

    def crop(self, remove):
        # 5.20.2 accepts negative removal counts, never positive absolute lengths.
        assert remove < 0
        self.crops.append(remove)
        self.tokens = self.tokens[:remove]
        self.layers[0].keys.shape = self.layers[0].values.shape = (1, 2, len(self.tokens), 8)


class Tokenizer:
    all_special_tokens = []

    def __init__(self, torch=None):
        self.torch = torch

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        ids = [1] + [ord(char) % 53 + 3 for char in messages[0]["content"]]
        ids += [2] + [ord(char) % 53 + 3 for char in messages[1]["content"]] + [2]
        if self.torch is not None:
            return Batch(input_ids=self.torch.tensor([ids]), attention_mask=self.torch.ones((1, len(ids)), dtype=self.torch.long))
        return Batch(input_ids=Tensor([ids]), attention_mask=Tensor([[1] * len(ids)]))

    def decode(self, tokens, **kwargs):
        return ",".join(str(int(token)) for token in tokens)


class Model:
    device, dtype = "cpu", "float32"

    def __init__(self):
        self.calls = []
        self.fail = False

    def generate(self, input_ids, attention_mask, **options):
        past = options.get("past_key_values")
        size = past.get_seq_length() if past is not None else 0
        self.calls.append({"prefill": len(input_ids[0]) - size, "past": size, **options})
        if self.fail:
            raise RuntimeError("forward failed")
        tokens = (past.tokens if past is not None else []) + list(input_ids[0][size:])
        generated = []
        for _ in range(options["max_new_tokens"]):
            next_token = sum((index + 1) * token for index, token in enumerate(tokens)) % 61 + 3
            generated.append(next_token)
            tokens.append(next_token)
        sequences = [list(input_ids[0]) + generated]
        if options["return_dict_in_generate"]:
            return SimpleNamespace(sequences=sequences, past_key_values=Cache(tokens[:-1]))
        return sequences


def planner(reuse=True, *, model=None, torch=None):
    instance = LocalPlanner.__new__(LocalPlanner)
    instance._torch = torch or SimpleNamespace(inference_mode=nullcontext)
    instance._device = torch.device("cpu") if torch else SimpleNamespace(type="cpu")
    instance._tokenizer = Tokenizer(torch)
    instance._model = model or Model()
    instance._context_tokens = 4096
    instance._reuse_prefix_cache = reuse
    instance._prefix_cache = _PrefixCache()
    return instance


def generate(instance, user, *, max_tokens=6, deadline=None):
    return instance._generate("stable tools and context", user, max_tokens=max_tokens, deadline=deadline)


class PrefixTests(unittest.TestCase):
    def test_greedy_equivalence_and_suffix_prefill(self):
        cached, fresh = planner(), planner(False)
        for user in ("request observations=[]", "request observations=[lookup]", "request observations=[lookup,write]",
                     "request observations=[lookup,write]", "request", "other"):
            self.assertEqual(generate(cached, user), generate(fresh, user))
        calls = cached._model.calls
        self.assertGreater(calls[1]["past"], 0)
        self.assertLess(calls[1]["prefill"], fresh._model.calls[1]["prefill"])
        self.assertTrue(all(call["do_sample"] is False and call["num_beams"] == 1 for call in calls))
        self.assertTrue(all(call["past"] == 0 for call in fresh._model.calls))
        self.assertIsNone(fresh._prefix_cache.past_key_values)

    def test_identical_prompt_recomputes_one_token(self):
        instance = planner()
        generate(instance, "same")
        generate(instance, "same")
        self.assertEqual(instance._model.calls[-1]["prefill"], 1)

    def test_limit_violation_clears_snapshot_and_never_reuses_it(self):
        instance = planner()
        expected = generate(instance, "valid")
        self.assertIsNotNone(instance._prefix_cache.past_key_values)
        instance._context_tokens = 30
        with self.assertRaisesRegex(ValueError, "context limit"):
            generate(instance, "too long " * 50)
        self.assertIsNone(instance._prefix_cache.past_key_values)
        self.assertEqual(instance._prefix_cache.token_ids, ())
        instance._context_tokens = 4096
        self.assertEqual(generate(instance, "valid"), expected)
        self.assertEqual(instance._model.calls[-1]["past"], 0)

    def test_cached_tokens_still_count_toward_output_budget(self):
        instance = planner()
        generate(instance, "same")
        count = len(instance._prefix_cache.token_ids)
        instance._context_tokens = count + 2
        generate(instance, "same", max_tokens=100)
        self.assertEqual(instance._model.calls[-1]["max_new_tokens"], 2)
        self.assertEqual(instance._model.calls[-1]["prefill"], 1)

    def test_failed_forward_consumes_snapshot(self):
        instance = planner()
        generate(instance, "valid")
        instance._model.fail = True
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            generate(instance, "valid more")
        self.assertIsNone(instance._prefix_cache.past_key_values)
        instance._model.fail = False
        generate(instance, "valid")
        self.assertEqual(instance._model.calls[-1]["past"], 0)

    def test_cache_shapes_devices_dtypes_and_model_identity_must_match(self):
        for change in ("device", "dtype", "shape", "model", "tokenizer"):
            with self.subTest(change=change):
                instance = planner()
                generate(instance, "same")
                cache = instance._prefix_cache.past_key_values
                if change in ("device", "dtype"):
                    for value in (cache.layers[0].keys, cache.layers[0].values):
                        setattr(value, change, "cuda:0" if change == "device" else "float16")
                elif change == "shape":
                    cache.layers[0].keys.shape = cache.layers[0].values.shape = (1, 4, cache.get_seq_length(), 8)
                elif change == "model":
                    instance._model = Model()
                else:
                    instance._tokenizer = Tokenizer()
                generate(instance, "same")
                self.assertEqual(instance._model.calls[-1]["past"], 0)

    def test_recurrent_sliding_and_broken_crop_fall_back(self):
        for change in ("recurrent", "sliding", "crop"):
            with self.subTest(change=change):
                instance = planner()
                generate(instance, "same")
                cache = instance._prefix_cache.past_key_values
                if change == "recurrent":
                    cache.is_croppable = False
                elif change == "sliding":
                    cache.layers[0].is_sliding = True
                else:
                    def unsupported(remove):
                        raise NotImplementedError("cannot roll back")
                    cache.crop = unsupported
                generate(instance, "same")
                self.assertEqual(instance._model.calls[-1]["past"], 0)

    def test_nontext_batches_and_padding_are_not_reused(self):
        instance = planner()
        def key(batch):
            return _prefix_input_key(instance._model, instance._tokenizer, batch, instance._device)
        self.assertIsNone(key(Batch(input_ids=Tensor([[1], [2]]))))
        self.assertIsNone(key(Batch(input_ids=Tensor([[1, 2]]), attention_mask=Tensor([[0, 1]]))))
        self.assertIsNone(key(Batch(input_ids=Tensor([[1]]), pixel_values=Tensor([[2]]))))
        first = key(Batch(input_ids=Tensor([[1]], device="cpu")))
        second = key(Batch(input_ids=Tensor([[1]], device="cuda:0")))
        self.assertNotEqual(first, second)

    def test_post_forward_sync_failure_discards_cache(self):
        instance = planner()
        instance._device = SimpleNamespace(type="cuda")
        def failed_sync(device):
            raise RuntimeError("sync failed")
        instance._torch.cuda = SimpleNamespace(synchronize=failed_sync)
        with self.assertRaisesRegex(RuntimeError, "sync failed"):
            generate(instance, "user")
        self.assertIsNone(instance._prefix_cache.past_key_values)
        self.assertFalse(_GENERATING.locked())

    def test_deadline_and_lock_are_retained(self):
        # No transformers import is needed for the initial expired-deadline check.
        instance = planner()
        with self.assertRaises(TimeoutError):
            generate(instance, "user", deadline=monotonic() - 1)
        with _GENERATING:
            with self.assertRaises(TimeoutError):
                generate(instance, "user", deadline=monotonic() + 0.01)
        self.assertEqual(instance._model.calls, [])


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                     "torch/transformers unavailable; real CPU Qwen equivalence NOT MEASURED")
class TinyQwenTests(unittest.TestCase):
    def test_real_cpu_qwen_greedy_equivalence_and_context_reset(self):
        import torch
        try:
            from transformers import Qwen3Config, Qwen3ForCausalLM
        except ImportError:
            self.skipTest("installed transformers lacks Qwen3 (project pins 5.20.2); equivalence NOT MEASURED here")

        torch.manual_seed(20261113)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                             max_position_embeddings=4096, bos_token_id=1, eos_token_id=None, pad_token_id=0,
                             attention_dropout=0.0)
        model = Qwen3ForCausalLM(config).to("cpu").eval()
        cached, fresh = planner(model=model, torch=torch), planner(False, model=model, torch=torch)
        prefill = []
        hook = model.register_forward_pre_hook(lambda _model, args, kwargs: prefill.append(
            kwargs["input_ids"].shape[-1]), with_kwargs=True)
        try:
            for user in ("observations=[]", "observations=[read]", "observations=[read,write]", "observations=[read,write]"):
                before = len(prefill)
                actual = generate(cached, user)
                cached_prefill = prefill[before]
                before = len(prefill)
                self.assertEqual(actual, generate(fresh, user))
                if user != "observations=[]":
                    self.assertLess(cached_prefill, prefill[before])
            self.assertIsNotNone(cached._prefix_cache.past_key_values)
            cached._context_tokens = 64
            with self.assertRaisesRegex(ValueError, "context limit"):
                generate(cached, "long " * 40)
            self.assertIsNone(cached._prefix_cache.past_key_values)
            cached._context_tokens = 4096
            before = len(prefill)
            actual = generate(cached, "observations=[]")
            first = prefill[before]
            before = len(prefill)
            self.assertEqual(actual, generate(fresh, "observations=[]"))
            self.assertEqual(first, prefill[before])
        finally:
            hook.remove()


if __name__ == "__main__":
    unittest.main()
