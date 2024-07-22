import json
import logging
import multiprocessing as mp
import os
import signal
import tempfile
from queue import Empty
from typing import Dict, Any, Set

import attrs
import wandb


@attrs.define
class StringAsFileArtifact:
    string: str
    file_name: str
    artifact_info: Dict[str, Any]


class WandbClient:
    def __init__(self, queue: mp.Queue, client_index: int):
        self.queue = queue
        self.client_index = client_index

        self.Table = wandb.Table
        self.Image = wandb.Image
        self.Video = wandb.Video
        self.Audio = wandb.Audio
        self.Html = wandb.Html

        self._started = False
        self._closed = False

    def _check_is_start_and_notclosed(self):
        if not self._started:
            raise ValueError("WandbClient not been started yet. Call .init() first.")

        if self._closed:
            raise ValueError("WandbClient is closed.")

    def init(self):
        if self._started:
            raise ValueError("WandbClient is already started, cannot init again.")

        if self._closed:
            raise ValueError("WandbClient is closed, cannot init.")

        self._started = True

    def log(self, data: Dict[str, Any]):
        self._check_is_start_and_notclosed()
        self.queue.put((self.client_index, False, data))

    def log_artifact(self, artifact):
        self._check_is_start_and_notclosed()
        self.queue.put((self.client_index, False, artifact))

    def log_string_as_file_artifact(self, saa: StringAsFileArtifact):
        self._check_is_start_and_notclosed()
        self.queue.put((self.client_index, False, saa))

    def close(self):
        if not self._closed:
            self.queue.put((self.client_index, True, None))
        self._closed = True

    def __del__(self):
        if self._started and not self._closed:
            self.close()


class WandbServer:
    def __init__(self, queue: mp.Queue, **wandb_kwargs):
        self.queue = queue
        self._num_clients_created = 0
        self._open_clients: Set[int] = set()

        wandb.init(**wandb_kwargs)

    def finish(self):
        assert not self.any_open_clients()
        wandb.finish()

    def create_client(self):
        wc = WandbClient(self.queue, client_index=self._num_clients_created)
        self._num_clients_created += 1
        self._open_clients.add(wc.client_index)
        return wc

    def any_open_clients(self):
        return len(self._open_clients) > 0

    @property
    def num_closed_clients(self):
        return self._num_clients_created - len(self._open_clients)

    def log(self, timeout: int, verbose: bool = False):
        logged_data = []
        while True:
            try:
                client_ind, closing, data = self.queue.get(timeout=timeout)

                assert client_ind in self._open_clients

                if closing:
                    if verbose:
                        print(f"Closing client {client_ind}")

                    self._open_clients.remove(client_ind)
                    continue

                if verbose:
                    print(f"Logging [from client {client_ind}]: {data}")

                if isinstance(data, wandb.Artifact):
                    wandb.log_artifact(data)
                elif isinstance(data, StringAsFileArtifact):
                    td = tempfile.TemporaryDirectory()
                    with td as temp_dir:
                        with open(
                            os.path.join(temp_dir, data.file_name), "w"
                        ) as temp_file:
                            temp_file.write(data.string)

                        artifact = wandb.Artifact(**data.artifact_info)
                        artifact.add_file(os.path.join(temp_dir, data.file_name))

                        wandb.log_artifact(artifact)
                else:
                    try:
                        wandb.log(data)
                    except TypeError:
                        if isinstance(data, dict):
                            new_data = {}
                            for k, v in data.items():
                                try:
                                    json.dumps(v)
                                    new_data[k] = v
                                except:
                                    new_data[k] = str(v)

                            wandb.log(new_data)
                        else:
                            raise

                logged_data.append(data)
            except Empty:
                break

        return logged_data


class DelayedKeyboardInterrupt:
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug(
            "SIGINT received. Delaying KeyboardInterrupt as critical code is running."
        )

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)
