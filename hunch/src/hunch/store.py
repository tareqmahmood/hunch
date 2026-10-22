"""Where the proxy keeps question versions, families, examples and the states they were asked about.

A *state* is stored once, under its hash (`state_hash`): its encoder features and, optionally, its text. Examples
reference states by hash, so a state asked several questions costs one feature vector. A state's features are
deleted when no example references it any more.

Examples are kept per key: labelled ones under the question's family, newest `labelled_cap` per class; sampled ones
under the question version, newest `sampled_cap` in all. A key holds at most one example of each kind per state: a new
answer for a state replaces the old one, so repeated states don't crowd the window or weigh twice in a fit. Every
example carries a sequence number, so a question's examples come back in the order they were recorded. `lookup`
returns the stored answer for an exact (question, state), and `put_fit` / `get_fit` keep each version's fitted
student, so a restarted proxy answers without refitting first. Each version's latest audits (`add_audit`, `audits`)
are kept too, so a demotion outlasts a restart.

For labelled states, the store also keeps the teacher's answer per version (`put_teacher`), outside the rotating
sampled window, so it lasts as long as the label does. `teacher_pairs` returns each label with that version's teacher
answer: the teacher measured against ground truth.

The store records the *encoder fingerprint* the features were computed with (checkpoint revision, layer, matmul
precision). Features computed by a different encoder don't mix with them: `check_fingerprint` raises
`FingerprintMismatch`, and a store that keeps texts can be re-encoded (`state_texts`, `replace_features`), which
drops the kept fits, since they were trained on the old features.

Backends: `SQLiteStore` (a local file) and `RedisStore` (`redis://host:port/db`, shared by several processes).
`open_store` picks one from a string.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

LABELLED, SAMPLED = "labelled", "sampled"

Entry = tuple[int, str, str, int, list | None]  # (seq, kind, state hash, label, probs or None)


class FingerprintMismatch(RuntimeError):
    """The store's features were computed by a different encoder (revision, layer or precision)."""


@dataclass
class Examples:
    """A question version's training rows, oldest first."""

    X: np.ndarray  # [n, d] float32 features
    y: np.ndarray  # [n] class indices
    conf: np.ndarray  # [n] the answer's confidence: max probability, 1.0 for a label
    kinds: list[str]  # LABELLED or SAMPLED per row
    probs: list[list[float] | None]  # the teacher's probabilities per row (None for a label or a bare class)
    states: list[str]  # the state hash per row

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class Row:
    """One example to record: the state it is about, its class and the teacher's probabilities (or None)."""

    state: str
    label: int
    probs: list[float] | None = None


class Store(ABC):
    """The proxy's persistent state. Implementations are safe to call from several threads."""

    def __init__(self, sampled_cap: int = 2000, labelled_cap: int = 5000, store_text: bool = False):
        self.sampled_cap, self.labelled_cap, self.store_text = sampled_cap, labelled_cap, store_text

    # -------- encoder

    def check_fingerprint(self, fingerprint: dict) -> None:
        """Record the encoder fingerprint on first use; raise FingerprintMismatch if it differs from the recorded one."""
        have = self._get_fingerprint()
        if have is None:
            self._set_fingerprint(fingerprint)
        elif have != fingerprint:
            raise self._fingerprint_mismatch(have, fingerprint)

    @staticmethod
    def _fingerprint_mismatch(have: dict | None, fingerprint: dict) -> FingerprintMismatch:
        return FingerprintMismatch(f"the store's features come from {have}, this encoder is {fingerprint}")

    @staticmethod
    def _signature_mismatch(family: str, have: dict, signature: dict) -> ValueError:
        return ValueError(f"{family} has {have}; this version has {signature}: versions in a family need the same "
                          "type and options, or their labels mean different things")

    @abstractmethod
    def _get_fingerprint(self) -> dict | None: ...

    @abstractmethod
    def _set_fingerprint(self, fingerprint: dict) -> None: ...

    # -------- questions and families

    @abstractmethod
    def add_question(self, key: str, spec: dict, family: str, signature: dict) -> None:
        """Record a version and its family (raising ValueError if the family's signature differs)."""

    @abstractmethod
    def join_family(self, key: str, spec: dict, family: str, signature: dict) -> None:
        """Put a version that had no family into one, moving its labelled examples to the family."""

    @abstractmethod
    def questions(self) -> dict[str, tuple[dict, str]]:
        """version key -> (spec, family key)."""

    # -------- states and examples

    @abstractmethod
    def add_examples(self, key: str, kind: str, rows: list[Row], features: dict[str, np.ndarray],
                     texts: dict[str, str] | None = None) -> None:
        """Record examples under `key` (a family for labelled rows, a version for sampled ones), store any of their
        states not stored yet, and apply the caps. `features` (and `texts`) are by state hash."""

    @abstractmethod
    def examples(self, key: str, family: str | None = None) -> Examples:
        """A version's training rows: its family's labelled examples and its own sampled ones (and any labelled
        examples recorded under the version itself). A sampled answer is dropped for any state that also has a label."""

    @abstractmethod
    def get(self, key: str, kind: str, state: str) -> tuple[int, list | None] | None:
        """The example of `kind` recorded under `key` for a state: (label, probs or None), or None."""

    @abstractmethod
    def put_teacher(self, key: str, answers: dict[str, list[float]]) -> None:
        """Keep the teacher's probabilities for labelled states (by state hash) under a version."""

    @abstractmethod
    def teacher_pairs(self, key: str, family: str | None = None) -> list[tuple[str, int, list[float]]]:
        """(state, label, teacher's probabilities) for every labelled state (the family's, then the version's) that
        has a kept teacher answer for this version."""

    @abstractmethod
    def lookup(self, key: str, family: str | None, state: str) -> tuple[str, int, list | None] | None:
        """The stored answer for a state: (kind, label, probs or None), from a label (the family's, then the
        version's) before a sampled answer; None if there is neither."""

    @abstractmethod
    def put_fit(self, key: str, blob: bytes) -> None:
        """Keep a version's serialized fit."""

    @abstractmethod
    def get_fit(self, key: str) -> bytes | None:
        """A version's serialized fit, if one was kept."""

    @abstractmethod
    def add_audit(self, key: str, entry: list, window: int) -> None:
        """Append an audit (any JSON list) to a version's window, keeping the newest `window`."""

    @abstractmethod
    def audits(self, key: str) -> list[list]:
        """A version's audit window, oldest first."""

    @abstractmethod
    def clear_audits(self, key: str) -> None:
        """Empty a version's audit window."""

    @abstractmethod
    def counts(self, key: str, family: str | None = None) -> dict[str, int]:
        """Examples per kind, as `examples` would count them before dropping contested sampled answers."""

    @abstractmethod
    def num_states(self) -> int: ...

    @abstractmethod
    def texts(self, hashes: list[str]) -> dict[str, str]:
        """The kept texts of the given states (only those with one: texts are kept with `store_text`)."""

    @abstractmethod
    def state_texts(self) -> Iterator[tuple[str, str | None]]:
        """(state hash, text or None) for every stored state."""

    @abstractmethod
    def replace_features(self, features: dict[str, np.ndarray], fingerprint: dict) -> None:
        """Overwrite the given states' features, record the encoder that computed them, and drop every kept fit
        (trained on the old features)."""

    def close(self) -> None:
        pass


def _fp(fingerprint: dict) -> str:
    return json.dumps(fingerprint, sort_keys=True)


def _drop_contested(entries: list[Entry]) -> list[Entry]:
    """(seq, kind, state, label, probs) rows in order, without sampled answers for states that have a label."""
    labelled = {s for _, k, s, _, _ in entries if k == LABELLED}
    return sorted(e for e in entries if e[1] == LABELLED or e[2] not in labelled)


def _pairs(labels: list[tuple[str, int]], teacher: dict[str, list[float]]) -> list[tuple[str, int, list[float]]]:
    """Labels (the family's first) joined with kept teacher answers, one per state."""
    out = {}
    for state, label in labels:
        if state in teacher and state not in out:
            out[state] = (state, label, teacher[state])
    return list(out.values())


def _as_examples(entries: list[Entry], feats: dict[str, np.ndarray]) -> Examples:
    """Training rows from entries in order, with each state's features."""
    if not entries:
        return Examples(np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=int), np.zeros(0), [], [], [])
    X = np.stack([feats[s] for _, _, s, _, _ in entries]).astype(np.float32, copy=False)
    y = np.array([lab for *_, lab, _ in entries], dtype=int)
    conf = np.array([max(p) if p else 1.0 for *_, p in entries])
    return Examples(X, y, conf, [k for _, k, *_ in entries], [p for *_, p in entries], [s for _, _, s, _, _ in entries])


# ---------------------------------------------------------------- SQLite


class SQLiteStore(Store):
    """A local SQLite file (or ":memory:")."""

    def __init__(self, path: str | Path, **kw):
        super().__init__(**kw)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript("""
                pragma journal_mode = wal;
                create table if not exists meta (name text primary key, value text);
                create table if not exists questions (key text primary key, spec text, family text, created real);
                create table if not exists families (family text primary key, signature text);
                create table if not exists states (hash text primary key, feat blob, text text);
                create table if not exists examples (
                    id integer primary key autoincrement, key text, kind text, state text, label integer, probs text);
                create index if not exists examples_key on examples (key, kind, label, id);
                create index if not exists examples_state on examples (state, key, kind);
                create table if not exists fits (key text primary key, blob blob, created real);
                create table if not exists teacher (key text, state text, probs text, primary key (key, state));
                create table if not exists audits (id integer primary key autoincrement, key text, entry text);
                create index if not exists audits_key on audits (key, id);
            """)
            self.db.commit()

    def _get_fingerprint(self):
        with self.lock:
            row = self.db.execute("select value from meta where name = 'fingerprint'").fetchone()
        return json.loads(row[0]) if row else None

    def _set_fingerprint(self, fingerprint):
        with self.lock:
            self.db.execute("insert or replace into meta values ('fingerprint', ?)", (_fp(fingerprint),))
            self.db.commit()

    def add_question(self, key, spec, family, signature):
        with self.lock:
            self._add_question(key, spec, family, signature)
            self.db.commit()

    def _add_question(self, key, spec, family, signature):
        sig = json.dumps(signature, sort_keys=True)
        row = self.db.execute("select signature from families where family = ?", (family,)).fetchone()
        if row and row[0] != sig:
            raise self._signature_mismatch(family, json.loads(row[0]), signature)
        self.db.execute("insert or ignore into families values (?, ?)", (family, sig))
        self.db.execute("insert or ignore into questions values (?, ?, ?, ?)", (key, json.dumps(spec), family, time.time()))

    def join_family(self, key, spec, family, signature):
        with self.lock:
            self._add_question(key, spec, family, signature)
            self.db.execute("update questions set family = ? where key = ?", (family, key))
            moved = self.db.execute("update examples set key = ? where key = ? and kind = ?", (family, key, LABELLED))
            if moved.rowcount:
                # one label per state: the version's labels replace the family's for the same states
                self.db.execute("delete from examples where key = ? and kind = ? and id not in "
                                "(select max(id) from examples where key = ? and kind = ? group by state)",
                                (family, LABELLED, family, LABELLED))
                self._trim(family, LABELLED)
            self.db.commit()

    def questions(self):
        with self.lock:
            return {k: (json.loads(s), f or k) for k, s, f in self.db.execute("select key, spec, family from questions")}

    def add_examples(self, key, kind, rows, features, texts=None):
        with self.lock:
            self.db.executemany(
                "insert or ignore into states values (?, ?, ?)",
                [(s, np.asarray(features[s], dtype=np.float32).tobytes(),
                  (texts or {}).get(s) if self.store_text else None) for s in dict.fromkeys(r.state for r in rows)])
            self.db.executemany("delete from examples where state = ? and key = ? and kind = ?",
                                [(s, key, kind) for s in dict.fromkeys(r.state for r in rows)])
            self.db.executemany("insert into examples (key, kind, state, label, probs) values (?, ?, ?, ?, ?)",
                                [(key, kind, r.state, int(r.label), json.dumps(r.probs) if r.probs is not None else None)
                                 for r in rows])
            self._trim(key, kind)
            self.db.commit()

    def _trim(self, key, kind):
        """Apply the cap to key's examples of this kind, and delete states no example references any more."""
        if kind == SAMPLED:
            old = self.db.execute("select id, state from examples where key = ? and kind = ? order by id desc "
                                  "limit -1 offset ?", (key, kind, self.sampled_cap)).fetchall()
        else:
            old = []
            for (label,) in self.db.execute("select distinct label from examples where key = ? and kind = ?",
                                            (key, kind)).fetchall():
                old += self.db.execute("select id, state from examples where key = ? and kind = ? and label = ? "
                                       "order by id desc limit -1 offset ?", (key, kind, label, self.labelled_cap)).fetchall()
        if not old:
            return
        self.db.executemany("delete from examples where id = ?", [(i,) for i, _ in old])
        self.db.executemany("delete from states where hash = ? and not exists (select 1 from examples where state = ?)",
                            [(s, s) for s in {s for _, s in old}])

    def examples(self, key, family=None):
        with self.lock:
            entries = [(i, k, s, lab, json.loads(p) if p else None) for i, k, s, lab, p in self.db.execute(
                "select id, kind, state, label, probs from examples where key = ? or (key = ? and kind = ?)",
                (key, family or key, LABELLED))]
            entries = _drop_contested(entries)
            hashes = list(dict.fromkeys(s for _, _, s, _, _ in entries))
            feats = {}
            for i in range(0, len(hashes), 900):  # SQLite's bound-parameter limit
                part = hashes[i: i + 900]
                feats |= {h: np.frombuffer(f, dtype=np.float32) for h, f in self.db.execute(
                    f"select hash, feat from states where hash in ({','.join('?' * len(part))})", part)}
        return _as_examples(entries, feats)

    def get(self, key, kind, state):
        with self.lock:
            row = self.db.execute("select label, probs from examples where key = ? and kind = ? and state = ? "
                                  "order by id desc limit 1", (key, kind, state)).fetchone()
        return (row[0], json.loads(row[1]) if row[1] else None) if row else None

    def put_teacher(self, key, answers):
        with self.lock:
            self.db.executemany("insert or replace into teacher values (?, ?, ?)",
                                [(key, s, json.dumps(p)) for s, p in answers.items()])
            self.db.commit()

    def teacher_pairs(self, key, family=None):
        with self.lock:
            labels = self.db.execute("select state, label from examples where kind = ? and key in (?, ?) "
                                     "order by key = ? desc, id desc", (LABELLED, family or key, key, family or key)).fetchall()
            teacher = {s: json.loads(p) for s, p in self.db.execute("select state, probs from teacher where key = ?", (key,))}
        return _pairs(labels, teacher)

    def lookup(self, key, family, state):
        with self.lock:
            row = self.db.execute(
                "select kind, label, probs from examples where state = ? and "
                "((key = ? and kind = ?) or (key in (?, ?) and kind = ?)) order by kind = ? desc, key = ? desc, id desc "
                "limit 1", (state, key, SAMPLED, family or key, key, LABELLED, LABELLED, family or key)).fetchone()
        return (row[0], row[1], json.loads(row[2]) if row[2] else None) if row else None

    def put_fit(self, key, blob):
        with self.lock:
            self.db.execute("insert or replace into fits values (?, ?, ?)", (key, blob, time.time()))
            self.db.commit()

    def get_fit(self, key):
        with self.lock:
            row = self.db.execute("select blob from fits where key = ?", (key,)).fetchone()
        return bytes(row[0]) if row else None

    def add_audit(self, key, entry, window):
        with self.lock:
            self.db.execute("insert into audits (key, entry) values (?, ?)", (key, json.dumps(entry)))
            self.db.execute("delete from audits where key = ? and id not in "
                            "(select id from audits where key = ? order by id desc limit ?)", (key, key, window))
            self.db.commit()

    def audits(self, key):
        with self.lock:
            return [json.loads(e) for (e,) in self.db.execute("select entry from audits where key = ? order by id", (key,))]

    def clear_audits(self, key):
        with self.lock:
            self.db.execute("delete from audits where key = ?", (key,))
            self.db.commit()

    def counts(self, key, family=None):
        with self.lock:
            return dict(self.db.execute("select kind, count(*) from examples where key = ? or (key = ? and kind = ?) "
                                        "group by kind", (key, family or key, LABELLED)).fetchall())

    def num_states(self):
        with self.lock:
            return self.db.execute("select count(*) from states").fetchone()[0]

    def texts(self, hashes):
        out = {}
        with self.lock:
            for i in range(0, len(hashes), 900):  # SQLite's bound-parameter limit
                part = hashes[i: i + 900]
                out |= dict(self.db.execute(f"select hash, text from states where text is not null and hash in "
                                            f"({','.join('?' * len(part))})", part).fetchall())
        return out

    def state_texts(self):
        with self.lock:
            rows = self.db.execute("select hash, text from states").fetchall()
        yield from rows

    def replace_features(self, features, fingerprint):
        with self.lock:
            self.db.executemany("update states set feat = ? where hash = ?",
                                [(np.asarray(f, dtype=np.float32).tobytes(), h) for h, f in features.items()])
            self.db.execute("insert or replace into meta values ('fingerprint', ?)", (_fp(fingerprint),))
            self.db.execute("delete from fits")
            self.db.commit()

    def close(self):
        with self.lock:
            self.db.close()


# ---------------------------------------------------------------- Redis


class RedisStore(Store):
    """Redis (`redis://host:port/db`), shareable by several proxy processes. Keys live under `prefix`:
      {p}:fingerprint                 string, the encoder fingerprint
      {p}:questions                   hash, version key -> {"spec", "family"}
      {p}:families                    hash, family -> signature
      {p}:feat / {p}:text / {p}:refs  hashes by state hash: features (float32 bytes), text, example references
      {p}:ex:{key}:sampled            list of sampled examples, newest last
      {p}:ex:{key}:labelled:{label}   list of labelled examples of one class, newest last
      {p}:labels:{key}                set of the classes with labelled examples under key
      {p}:idx:{key}:{kind}            hash, state hash -> that state's example entry under key (one per state)
      {p}:teacher:{key}               hash, state hash -> the teacher's probabilities for a labelled state
      {p}:fits                        hash, version key -> serialized fit
      {p}:audits:{key}                list of a version's audits (JSON), newest last
      {p}:seq                         counter giving every example its sequence number
    Example entries are JSON: [seq, state hash, label, probs or null]. A state's reference count is the number of
    list entries naming it. Every change to a list, its index and the counts runs in one WATCH transaction (retried
    if a watched key changed), so concurrent writers can't count an entry twice or release a state still in use."""

    def __init__(self, url: str = "redis://localhost:6379/0", prefix: str = "hunch", client=None, **kw):
        super().__init__(**kw)
        if client is None:
            import redis

            client = redis.Redis.from_url(url)
        self.r, self.p = client, prefix

    def _k(self, *parts) -> str:
        return ":".join((self.p, *map(str, parts)))

    def _list(self, key, kind, label=None) -> str:
        return self._k("ex", key, kind) if kind == SAMPLED else self._k("ex", key, kind, label)

    def _get_fingerprint(self):
        v = self.r.get(self._k("fingerprint"))
        return json.loads(v) if v else None

    def _set_fingerprint(self, fingerprint):
        self.r.setnx(self._k("fingerprint"), _fp(fingerprint))
        if self._get_fingerprint() != json.loads(_fp(fingerprint)):  # another process recorded a different one first
            raise self._fingerprint_mismatch(self._get_fingerprint(), fingerprint)

    def add_question(self, key, spec, family, signature):
        sig = json.dumps(signature, sort_keys=True)
        self.r.hsetnx(self._k("families"), family, sig)
        have = self.r.hget(self._k("families"), family).decode()
        if have != sig:
            raise self._signature_mismatch(family, json.loads(have), signature)
        self.r.hsetnx(self._k("questions"), key, json.dumps({"spec": spec, "family": family}))

    def join_family(self, key, spec, family, signature):
        self.add_question(key, spec, family, signature)
        self.r.hset(self._k("questions"), key, json.dumps({"spec": spec, "family": family}))
        moved = self.r.hgetall(self._k("idx", key, LABELLED))
        if moved:
            # one label per state: the version's labels replace the family's for the same states
            fidx = self._k("idx", family, LABELLED)
            have = dict(zip(moved, self.r.hmget(fidx, list(moved))))
            pipe = self.r.pipeline()
            for state, prev in have.items():
                if prev is not None:
                    pipe.lrem(self._list(family, LABELLED, json.loads(prev)[2]), 1, prev)
                    pipe.hincrby(self._k("refs"), state, -1)
            pipe.hset(fidx, mapping=moved)
            pipe.delete(self._k("idx", key, LABELLED))
            pipe.execute()
        for label in [int(x) for x in self.r.smembers(self._k("labels", key))]:
            src, dst = self._list(key, LABELLED, label), self._list(family, LABELLED, label)
            entries = self.r.lrange(src, 0, -1)
            if entries:
                pipe = self.r.pipeline()
                pipe.rpush(dst, *entries)
                pipe.sadd(self._k("labels", family), label)
                pipe.delete(src)
                pipe.execute()
            self._trim(dst, self.labelled_cap, self._k("idx", family, LABELLED))
        self.r.delete(self._k("labels", key))

    def questions(self):
        out = {}
        for k, v in self.r.hgetall(self._k("questions")).items():
            d = json.loads(v)
            out[k.decode()] = (d["spec"], d["family"] or k.decode())
        return out

    def _transact(self, watch: list[str], body) -> list:
        """Run `body(pipe)` under WATCH on the given keys: it reads through `pipe`, calls `pipe.multi()` and queues its
        writes. Retried until no watched key changed in between; returns the writes' results."""
        from redis.exceptions import WatchError

        while True:
            with self.r.pipeline() as pipe:
                try:
                    pipe.watch(*watch)
                    if body(pipe) is False:
                        pipe.reset()
                        return []
                    return pipe.execute()
                except WatchError:
                    continue

    def add_examples(self, key, kind, rows, features, texts=None):
        if not rows:
            return
        seq0 = self.r.incrby(self._k("seq"), len(rows)) - len(rows)
        idx = self._k("idx", key, kind)
        states = list(dict.fromkeys(r.state for r in rows))
        lists = set()

        def write(pipe):
            old = dict(zip(states, pipe.hmget(idx, states)))
            pipe.multi()
            for s in states:  # a state's features are set while its count goes up, so they can't be released between
                pipe.hsetnx(self._k("feat"), s, np.asarray(features[s], dtype=np.float32).tobytes())
                if self.store_text and texts and s in texts:
                    pipe.hsetnx(self._k("text"), s, texts[s])
            for i, r in enumerate(rows):
                if old.get(r.state) is not None:  # one example per state: the new answer replaces the old one
                    prev = old.pop(r.state)
                    pipe.lrem(self._list(key, kind, json.loads(prev)[2]), 1, prev)
                    pipe.hincrby(self._k("refs"), r.state, -1)
                lst = self._list(key, kind, r.label)
                entry = json.dumps([seq0 + i, r.state, int(r.label), r.probs])
                pipe.rpush(lst, entry)
                pipe.hset(idx, r.state, entry)
                pipe.hincrby(self._k("refs"), r.state, 1)
                if kind == LABELLED:
                    pipe.sadd(self._k("labels", key), int(r.label))
                lists.add(lst)

        # the index says which entry each state has; trims change it in the same transaction as the list
        self._transact([idx], write)
        for lst in lists:
            self._trim(lst, self.sampled_cap if kind == SAMPLED else self.labelled_cap, idx)

    def _trim(self, lst: str, cap: int, idx: str) -> None:
        """Keep the newest `cap` entries of a list, dropping the others' entries in the index `idx` and releasing
        the states no entry names any more, all in one transaction."""
        refs = self._k("refs")

        def trim(pipe):
            n = pipe.llen(lst)
            if n <= cap:
                return False
            old = pipe.lrange(lst, 0, n - cap - 1)
            states = [json.loads(e)[1] for e in old]
            uniq = list(dict.fromkeys(states))
            indexed = dict(zip(uniq, pipe.hmget(idx, uniq)))
            left = {s: int(c or 0) for s, c in zip(uniq, pipe.hmget(refs, uniq))}
            for s in states:
                left[s] -= 1
            pipe.multi()
            pipe.ltrim(lst, n - cap, -1)
            for e, s in zip(old, states):
                pipe.hincrby(refs, s, -1)
                if indexed[s] == e:
                    pipe.hdel(idx, s)
            dead = [s for s in uniq if left[s] <= 0]
            if dead:
                pipe.hdel(self._k("feat"), *dead)
                pipe.hdel(self._k("text"), *dead)
                pipe.hdel(refs, *dead)

        # watching the counts means a writer adding an entry for a released state retries the release
        self._transact([lst, idx, refs], trim)

    def _entries(self, lists) -> list[Entry]:
        out = []
        for kind, lst in lists:
            for e in self.r.lrange(lst, 0, -1):
                seq, s, lab, p = json.loads(e)
                out.append((seq, kind, s, lab, p))
        return out

    def _lists(self, key, family):
        lists = [(SAMPLED, self._list(key, SAMPLED))]
        for k in dict.fromkeys([key, family or key]):
            lists += [(LABELLED, self._list(k, LABELLED, int(x))) for x in self.r.smembers(self._k("labels", k))]
        return lists

    def examples(self, key, family=None):
        entries = _drop_contested(self._entries(self._lists(key, family)))
        hashes = list(dict.fromkeys(s for _, _, s, _, _ in entries))
        feats = {}
        for i in range(0, len(hashes), 1000):
            part = hashes[i: i + 1000]
            feats |= {h: np.frombuffer(f, dtype=np.float32) for h, f in zip(part, self.r.hmget(self._k("feat"), part))}
        return _as_examples(entries, feats)

    def get(self, key, kind, state):
        e = self.r.hget(self._k("idx", key, kind), state)
        if e is None:
            return None
        _, _, lab, p = json.loads(e)
        return lab, p

    def put_teacher(self, key, answers):
        if answers:
            self.r.hset(self._k("teacher", key), mapping={s: json.dumps(p) for s, p in answers.items()})

    def teacher_pairs(self, key, family=None):
        labels = []
        for k in dict.fromkeys([family or key, key]):
            labels += [(s.decode(), json.loads(e)[2]) for s, e in self.r.hgetall(self._k("idx", k, LABELLED)).items()]
        teacher = {s.decode(): json.loads(p) for s, p in self.r.hgetall(self._k("teacher", key)).items()}
        return _pairs(labels, teacher)

    def lookup(self, key, family, state):
        for k, kind in ((family or key, LABELLED), (key, LABELLED), (key, SAMPLED)):
            e = self.r.hget(self._k("idx", k, kind), state)
            if e is not None:
                _, _, lab, p = json.loads(e)
                return kind, lab, p
        return None

    def put_fit(self, key, blob):
        self.r.hset(self._k("fits"), key, blob)

    def get_fit(self, key):
        return self.r.hget(self._k("fits"), key)

    def add_audit(self, key, entry, window):
        pipe = self.r.pipeline()
        pipe.rpush(self._k("audits", key), json.dumps(entry))
        pipe.ltrim(self._k("audits", key), -window, -1)
        pipe.execute()

    def audits(self, key):
        return [json.loads(e) for e in self.r.lrange(self._k("audits", key), 0, -1)]

    def clear_audits(self, key):
        self.r.delete(self._k("audits", key))

    def counts(self, key, family=None):
        out: dict[str, int] = {}
        for kind, lst in self._lists(key, family):
            n = self.r.llen(lst)
            if n:
                out[kind] = out.get(kind, 0) + n
        return out

    def num_states(self):
        return self.r.hlen(self._k("feat"))

    def texts(self, hashes):
        if not hashes:
            return {}
        return {h: t.decode() for h, t in zip(hashes, self.r.hmget(self._k("text"), hashes)) if t is not None}

    def state_texts(self):
        texts = self.r.hgetall(self._k("text"))
        for h in self.r.hkeys(self._k("feat")):
            t = texts.get(h)
            yield h.decode(), t.decode() if t is not None else None

    def replace_features(self, features, fingerprint):
        pipe = self.r.pipeline()
        for h, f in features.items():
            pipe.hset(self._k("feat"), h, np.asarray(f, dtype=np.float32).tobytes())
        pipe.set(self._k("fingerprint"), _fp(fingerprint))
        pipe.delete(self._k("fits"))
        pipe.execute()

    def close(self):
        self.r.close()


def open_store(where: str | Path | Store, **kw) -> Store:
    """A Store from a path (SQLite), a `redis://`, `rediss://` or `unix://` URL (an optional `?prefix=` names the key
    prefix), or a given Store. `kw` sets sampled_cap, labelled_cap and store_text."""
    if isinstance(where, Store):
        for name, value in kw.items():  # the caller's caps and text policy apply to a given store too
            setattr(where, name, value)
        return where
    s = str(where)
    if s.startswith(("redis://", "rediss://", "unix://")):
        from urllib.parse import parse_qs, urlsplit, urlunsplit

        u = urlsplit(s)
        q = parse_qs(u.query)
        prefix = q.pop("prefix", ["hunch"])[0]
        rest = "&".join(f"{k}={v[0]}" for k, v in q.items())
        return RedisStore(urlunsplit(u._replace(query=rest)), prefix=prefix, **kw)
    return SQLiteStore(s, **kw)
