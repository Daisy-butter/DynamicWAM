#!/usr/bin/env python3
"""Execute the actual teardown methods with CPU-only lifecycle test doubles."""
from __future__ import annotations

import ast
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "src/dynamicwam/training/train/multirate_trainer.py"
ENTRY = ROOT / "scripts/train_multirate.py"


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


def main():
    logger = logging.getLogger("teardown_cpu")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    env = {"logger": logger, "torch": NS(no_grad=lambda: lambda function: function)}
    definitions(TRAINER, {"MultirateTrainer"}, env)
    Trainer = env["MultirateTrainer"]
    events = []

    class Iterator:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail

        def _shutdown_workers(self):
            events.append(self.name)
            if self.fail:
                raise RuntimeError(self.name)

    def make_trainer(fail=False):
        obj = Trainer.__new__(Trainer)
        obj.train_loader = NS(base_dataloader=NS(_iterator=Iterator("train_workers", fail)))
        obj.val_loader = NS(base_dataloader=NS(_iterator=Iterator("val_workers")))
        obj._run_loaders = [obj.train_loader, obj.train_loader]
        obj._active_data_iterator = NS(close=lambda: events.append("iterator_close"))
        obj.dataset = NS(close=lambda: events.append("train_dataset_close"))
        obj.val_dataset = NS(close=lambda: events.append("val_dataset_close"))
        return obj

    trainer = make_trainer()
    trainer._close_data_loaders()
    assert events == ["iterator_close", "train_workers", "val_workers", "train_dataset_close", "val_dataset_close"], events
    events.clear()
    trainer._close_data_loaders()
    assert not any("workers" in item or item == "iterator_close" for item in events)

    events.clear()
    trainer = make_trainer(fail=True)
    try:
        trainer._close_data_loaders()
        raise AssertionError("expected cleanup error")
    except RuntimeError as exc:
        assert str(exc) == "train_workers"
    assert "val_workers" in events and "val_dataset_close" in events

    trainer = make_trainer()
    trainer.scheduler, other = object(), object()
    trainer.accelerator = NS(_custom_objects=[other, trainer.scheduler])
    trainer.model = trainer.optimizer = trainer.native_teacher = object()
    trainer.release_training_resources()
    assert trainer.model is trainer.optimizer is trainer.native_teacher is None
    assert trainer.accelerator._custom_objects == [other]

    # Evaluate the real cursor-selection expression, not a copied implementation.
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MultirateTrainer")
    run = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_run_training")
    assignment = next(node for node in run.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "loader" for t in node.targets))
    expression = compile(ast.Expression(assignment.value), str(TRAINER), "eval")
    calls = []
    source_loader, clone = object(), object()
    state = NS(train_loader=source_loader, batches_in_epoch=0,
               accelerator=NS(skip_first_batches=lambda source, cursor: calls.append((source, cursor)) or clone))
    assert eval(expression, {"self": state}) is source_loader and not calls
    state.batches_in_epoch = 3
    assert eval(expression, {"self": state}) is clone and calls == [(source_loader, 3)]

    for failure in (False, True):
        events.clear()
        trainer = make_trainer(fail=failure)
        original = ValueError("original training failure")
        def run_body():
            if failure:
                raise original
        trainer._run_training = run_body
        try:
            trainer.run()
            assert not failure
        except ValueError as exc:
            assert exc is original
        assert "train_workers" in events and "val_workers" in events

    # Run the real entry main with no torch import or device initialization.
    for failure in (False, True):
        events.clear()
        original = ValueError("entry original")
        class DummyTrainer:
            def __init__(self, config, accelerator):
                pass

            def run(self):
                events.append("run")
                if failure:
                    raise original

            def release_training_resources(self):
                events.append("release")
                if failure:
                    raise RuntimeError("cleanup must not mask original")

        accel = NS(process_index=0, device="cuda:0",
                   wait_for_everyone=lambda: events.append("barrier"),
                   free_memory=lambda: events.append("free"),
                   end_training=lambda: events.append("end"))
        profile = NS(training={"log_level": "INFO"}, deepspeed={})
        namespace = dict(sys=sys, logging=NS(getLogger=lambda name: logger),
                         gc=NS(collect=lambda: events.append("collect")),
                         build_arg_parser=lambda _: NS(parse_args=lambda: NS(config="unused")),
                         load_multirate_profile=lambda _: profile,
                         build_accelerator=lambda *a, **kw: accel,
                         setup_logging=lambda *a, **kw: None, MultirateTrainer=DummyTrainer)
        definitions(ENTRY, {"finish_training", "main"}, namespace)
        utils = ModuleType("accelerate.utils")
        utils.set_seed = lambda seed: None
        torch = ModuleType("torch")
        torch.cuda = NS(synchronize=lambda device: events.append("synchronize"))
        with patch.dict(sys.modules, {"accelerate.utils": utils, "torch": torch}):
            try:
                assert namespace["main"]() == 0
                assert not failure
            except ValueError as exc:
                assert failure and exc is original
        expected = ["run", "release", "free", "collect", "end"] if failure else ["run", "synchronize", "barrier", "release", "free", "collect", "end"]
        assert events == expected, events
    print("PASS: owned iterator/worker cleanup, deduplication, idempotence, cleanup failures,")
    print("scheduler/model release, zero/resume cursor ownership, original-error preservation,")
    print("success-only synchronization/barrier and exactly-once end_training (CPU stubs).")


if __name__ == "__main__":
    main()
