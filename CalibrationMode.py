"""
Calibration Mode - Multi-Dimensional Search Engine

Searches across parameter space to find channels showing better-than-chance performance:
- Multiple hidden-bit sources (main RNG stream, hash-based, fractals)
- Various prediction methods and timescales
- Systematic experiment tracking and p-value ranking
"""

import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
import time
import os
import hashlib
from scipy.stats import binomtest, chisquare
from collections import deque
import json
import sqlite3
from tkinter import messagebox


# ============================================================================
# HIDDEN BIT SOURCES - Different ways to generate ground truth
# ============================================================================

class HiddenBitSource:
    """Base class for hidden-bit generation strategies"""
    def __init__(self, name):
        self.name = name
    
    def get_bit(self, rng_app):
        """Generate next hidden bit"""
        raise NotImplementedError


class MainStreamSource(HiddenBitSource):
    """Use next bit from main RNG stream - tests if stream predicts itself"""
    def __init__(self):
        super().__init__("Main RNG Stream")
    
    def get_bit(self, rng_app):
        # Do NOT consume the main stream. Prefer a non-destructive read of the
        # most recent bit produced by the main window. This avoids toggling the
        # stream state by calling get_random_bit() directly.
        try:
            # Prefer the recent window (latest visible bit)
            if hasattr(rng_app, 'window') and len(rng_app.window) > 0:
                return int(rng_app.window[-1])
            # Fallback to bit_stream if available
            if hasattr(rng_app, 'bit_stream') and len(rng_app.bit_stream) > 0:
                return int(rng_app.bit_stream[-1])
        except Exception:
            pass

        # As a last resort, fall back to a non-deterministic bit (should not happen
        # during normal operation when the main window is running).
        if hasattr(rng_app, 'get_random_bit') and callable(rng_app.get_random_bit):
            try:
                return int(rng_app.get_random_bit())
            except Exception:
                pass

        return int.from_bytes(os.urandom(1), "big") & 1


class NumpyPRNGSource(HiddenBitSource):
    """Use numpy PRNG - tests synchronicity between separate generators"""
    def __init__(self):
        super().__init__("Numpy PRNG")
    
    def get_bit(self, rng_app):
        return int(np.random.randint(0, 2))


class HashTimestampSource(HiddenBitSource):
    """Hash current timestamp - deterministic but unpredictable"""
    def __init__(self):
        super().__init__("Hash Timestamp")
        self.counter = 0  # Add counter for uniqueness
    
    def get_bit(self, rng_app):
        # Combine timestamp + counter for uniqueness
        self.counter += 1
        data = f"{time.time()}_{self.counter}".encode()
        hash_val = hashlib.sha256(data).digest()
        # Use different byte based on counter to get better distribution
        return hash_val[self.counter % 32] & 1


class MandelbrotSource(HiddenBitSource):
    """Mandelbrot iteration count - fractal pattern"""
    def __init__(self):
        super().__init__("Mandelbrot Fractal")
        self.trial_counter = 0
    
    def get_bit(self, rng_app):
        # Use walk and entropy as complex coordinates with better scaling
        self.trial_counter += 1
        walk = rng_app.cumulative_val if hasattr(rng_app, 'cumulative_val') else 0
        entropy = self.get_entropy(rng_app)
        
        # Scale to interesting region of Mandelbrot set (-2 to 2)
        # Add trial counter for variation
        real = (walk % 1000) / 250.0 - 2.0  # Range: -2 to 2
        imag = entropy * 2.0 - 1.0  # Range: -1 to 1
        
        # Add small offset based on trial counter
        offset = (self.trial_counter % 100) / 100.0 * 0.1
        c = complex(real + offset, imag)
        
        iterations = self.mandelbrot_iterations(c, max_iter=50)
        # Use multiple bits from iteration count for better distribution
        return (iterations + self.trial_counter) % 2
    
    def mandelbrot_iterations(self, c, max_iter=50):
        z = 0
        for n in range(max_iter):
            if abs(z) > 2:
                return n
            z = z*z + c
        return max_iter
    
    def get_entropy(self, rng_app):
        # Compute a simple entropy estimate from the most recent p_history entry.
        # Returns a value in [0,1] where 1 is maximal uncertainty (p=0.5).
        if not hasattr(rng_app, 'p_history') or not rng_app.p_history:
            return 1.0
        p = rng_app.p_history[-1]
        try:
            if p <= 0 or p >= 1:
                return 0.0
            return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        except Exception:
            return 1.0
# ============================================================================
# EXPERIMENT DESCRIPTOR - Encapsulates all parameters for one calibration run
# ============================================================================

class ExperimentDescriptor:
    """Describes a single calibration experiment configuration"""
    def __init__(self, exp_id, method, hidden_source, params=None):
        self.id = exp_id
        self.method = method  # Oracle prediction method
        self.hidden_source = hidden_source  # HiddenBitSource instance
        self.params = params or {}
        self.timestamp = time.time()
    
    def to_dict(self):
        return {
            'id': self.id,
            'method': self.method,
            'hidden_source': self.hidden_source.name,
            'params': json.dumps(self.params),
            'timestamp': self.timestamp
        }


# ============================================================================
# ADAPTIVE ORACLE - Learns RNG state → answer mappings
# ============================================================================

class AdaptiveOracle:
    """Oracle that learns RNG state → answer mappings from feedback"""
    
    def __init__(self, fft_weight=0.5):
        self.training_data = []  # List of {rng_state, answer, correct}
        self.similarity_threshold = 0.7
        self.fft_weight = fft_weight  # Boost FFT importance (was 0.3)
        
    def predict(self, rng_state):
        """Make prediction based on learned patterns or default strategy"""
        # Search for similar past states
        similar = self.find_similar_states(rng_state)
        
        if similar:
            # Use most common answer from similar states
            answers = [s['answer'] for s in similar]
            return max(set(answers), key=answers.count)
        else:
            # Default: random guess (50/50) instead of walk-based
            # Walk-based was biased because walk is often negative early on
            return int(np.random.randint(0, 2))
    
    def learn(self, rng_state, correct_answer, was_correct):
        """Store this RNG state → answer association"""
        self.training_data.append({
            'rng_state': rng_state,
            'answer': correct_answer,
            'was_correct': was_correct,
            'timestamp': time.time()
        })
        
        # Keep only last 200 examples (short-term memory)
        if len(self.training_data) > 200:
            self.training_data.pop(0)
    
    def find_similar_states(self, current_state):
        """Find past RNG states similar to current one"""
        similar = []
        for record in self.training_data:
            if self.compute_similarity(current_state, record['rng_state']) > self.similarity_threshold:
                similar.append(record)
        return similar
    
    def compute_similarity(self, state1, state2):
        """Compute similarity between two RNG states (0-1)"""
        # Walk direction match
        walk_sim = 1.0 if (state1.get('walk', 0) > 0) == (state2.get('walk', 0) > 0) else 0.0
        
        # Entropy similarity
        e1 = state1.get('entropy', 1.0)
        e2 = state2.get('entropy', 1.0)
        entropy_sim = max(0, 1.0 - abs(e1 - e2))
        
        # FFT anomaly match
        fft_sim = 1.0 if state1.get('fft_anomaly', False) == state2.get('fft_anomaly', False) else 0.0
        
        # Weighted average - FFT gets higher weight (0.5 vs 0.3)
        walk_weight = 0.3
        entropy_weight = 0.2
        return (walk_sim * walk_weight + entropy_sim * entropy_weight + fft_sim * self.fft_weight)


# ============================================================================
# CALIBRATION MODE - Runs experiments and tracks results
# ============================================================================

class CalibrationMode:
    """Automated self-calibration system with experiment tracking"""
    
    def __init__(self, rng_app, experiment=None):
        self.rng_app = rng_app
        self.oracle = AdaptiveOracle(fft_weight=0.5)  # Boost FFT weight
        # Probe vs Adaptive (learning) mode. 'adaptive' = existing behavior, 'probe' = no learning
        self.mode = 'adaptive'
        
        # Experiment configuration
        self.experiment = experiment or ExperimentDescriptor(
            exp_id="default",
            method="adaptive_walk",
            hidden_source=MainStreamSource()  # Use main RNG stream by default
        )
        
        # Calibration state
        self.active = False
        self.trials_run = 0
        self.hits = 0
        self.recent_results = deque(maxlen=10)
        
        # Current trial
        self.hidden_bit = None
        self.prediction = None
        # Translations log file
        self.log_dir = os.environ.get('CALIBRATION_LOG_DIR', 'rng_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.translations_path = os.path.join(self.log_dir, 'calibration_translations.jsonl')
        # Pending lagged probes: deque of dicts with keys: hidden, target_total, config
        self.pending_probes = deque()
        # Exploration state
        self.exploration_active = False
        self.exploration_tasks = []  # list of dicts: {source, lag, trials_left, config_id}
        self.exploration_progress = {}  # config_id -> {'scheduled': int, 'completed': int, 'total': int}
        # Queue for sequential config runs (source x mode combinations)
        self.exploration_queue = deque()
        self.exploration_summaries = []  # list of per-config summary dicts
        self.current_config = None
        
    def start_calibration(self):
        """Start automated calibration"""
        self.active = True
        self.trials_run = 0
        self.hits = 0
        self.recent_results.clear()
        print("[CALIBRATION] Started automated self-calibration")
    
    def stop_calibration(self):
        """Stop calibration"""
        self.active = False
        print(f"[CALIBRATION] Stopped. Trials: {self.trials_run}, Accuracy: {self.get_accuracy():.1%}")
    
    def run_trial(self):
        """Run one automated calibration trial"""
        if not self.active:
            return None
        
        # 1. Generate hidden bit using configured source
        self.hidden_bit = self.experiment.hidden_source.get_bit(self.rng_app)
        
        # 2. Get current RNG state
        rng_state = self.get_rng_state()
        
        # 3. Oracle makes prediction
        raw_prediction = self.oracle.predict(rng_state)
        
        # ===== PHASE INVERTER =====
        # Fractal sources apply opposite transformation
        # Invert oracle prediction to match fractal polarity
        # This tests: Do different mathematical structures have different consciousness coupling phases?
        if "Fractal" in self.experiment.hidden_source.name or \
           "Mandelbrot" in self.experiment.hidden_source.name:
            self.prediction = 1 - raw_prediction  # NOT gate (phase inversion)
            print(f"[PHASE-INVERT] Fractal source detected: raw={raw_prediction} → inverted={self.prediction}")
        else:
            self.prediction = raw_prediction
        # ===========================
        
        # 4. Compare prediction to truth
        correct = (self.prediction == self.hidden_bit)
        
        # 5. Oracle learns from result (only in adaptive/learning mode)
        if self.mode == 'adaptive':
            # CRITICAL: Oracle learns the INVERTED prediction succeeded for fractals
            self.oracle.learn(rng_state, self.hidden_bit, correct)
        else:
            # In probe mode, record the translation (hidden -> prediction/universe label)
            universe_label = self.get_universe_label(rng_state, self.hidden_bit, raw_prediction)
            self.record_translation({
                'trial': self.trials_run + 1,
                'hidden': int(self.hidden_bit),
                'predicted': int(self.prediction),
                'raw_prediction': int(raw_prediction),
                'universe_label': universe_label,
                'rng_state': rng_state,
                'timestamp': time.time()
            })
        
        # 6. Record result
        self.trials_run += 1
        if correct:
            self.hits += 1
        
        self.recent_results.append(correct)

        # After each trial, also process pending lagged probes (resolve any whose target bit has appeared)
        self.process_pending_probes()
        
        return {
            'trial': self.trials_run,
            'hidden': self.hidden_bit,
            'predicted': self.prediction,
            'raw_prediction': raw_prediction,  # Track both for analysis
            'phase_inverted': self.prediction != raw_prediction,
            'correct': correct,
            'rng_state': rng_state
        }

    def schedule_lagged_probe(self, hidden_source, lag, config_id=None):
        """Schedule a lagged probe: read hidden bit now, resolve when main window reaches target index.

        lag is measured in bits (1 = next bit). target_total is the absolute total-bit index when the prediction should be read.
        """
        # Use the main app's monotonic total_bits counter to compute an absolute target
        cur_total = getattr(self.rng_app, 'total_bits', 0)
        target_total = cur_total + int(lag)
        hidden = hidden_source.get_bit(self.rng_app)
        entry = {
            'hidden': int(hidden),
            'target_total': int(target_total),
            'config': config_id,
            'timestamp': time.time()
        }
        self.pending_probes.append(entry)
        return entry

    def process_pending_probes(self):
        """Resolve pending probes whose target bit is available in the main window."""
        if not hasattr(self.rng_app, 'window'):
            return
        window = list(self.rng_app.window)
        cur_total = getattr(self.rng_app, 'total_bits', 0)
        resolved = []
        while self.pending_probes:
            entry = self.pending_probes[0]
            # Resolve when the main RNG has produced enough bits (absolute count)
            if entry.get('target_total') is not None and entry['target_total'] <= cur_total:
                self.pending_probes.popleft()
                # Compute the index into the current window for the target bit.
                # window holds the most recent WINDOW_SIZE bits; target_total refers to absolute count.
                # We map target_total to an index from the right: offset = cur_total - entry['target_total']
                offset_from_right = cur_total - entry['target_total']
                # If offset_from_right is within the window length, we can index it.
                if offset_from_right < len(window):
                    pred = int(window[-1 - offset_from_right])
                else:
                    # The requested bit has scrolled out of the window; fall back to most recent bit
                    pred = int(window[-1])
                correct = (pred == entry['hidden'])
                # Record translation event
                rec = {
                    'trial': None,
                    'hidden': entry['hidden'],
                    'predicted': pred,
                    'universe_label': pred,
                    'config': entry['config'],
                    'resolved_at_total': entry.get('target_total', None),
                    'correct': correct,
                    'timestamp': time.time()
                }
                self.record_translation(rec)
                try:
                    print(f"[EXPLORER] Resolved config={entry.get('config')} target_total={entry.get('target_total')} pred={pred} hidden={entry.get('hidden')} correct={correct}")
                except Exception:
                    pass
                # Update stats
                self.trials_run += 1
                if correct:
                    self.hits += 1
                self.recent_results.append(correct)
                # Return a resolved result for UI consumption
                resolved.append(rec)
                # Update exploration progress if applicable
                cfg = entry.get('config')
                if cfg and cfg in self.exploration_progress:
                    self.exploration_progress[cfg]['completed'] += 1
            else:
                break
        return resolved

    def run_exploration_suite(self, sources, lags, trials_per_config=10, spacing=1, config_prefix='expl'):
        """Run an automated exploration suite over multiple sources and lag values.

        - sources: list of HiddenBitSource instances
        - lags: iterable of integer lags (bits)
        - trials_per_config: number of probe trials per (source, lag) combination
        - spacing: spacing in bits between scheduled trials to avoid collisions
        Returns a list of summaries for each config.
        """
        summaries = []
        # Create exploration tasks (will be scheduled over time by step_exploration)
        for s_i, source in enumerate(sources):
            for lag in lags:
                config_id = f"{config_prefix}_{s_i}_lag{lag}"
                task = {
                    'source': source,
                    'lag': int(lag),
                    'trials_left': int(trials_per_config),
                    'spacing': int(spacing),
                    'config_id': config_id,
                    'next_offset': 0  # offset to space subsequent trials
                }
                self.exploration_tasks.append(task)
                self.exploration_progress[config_id] = {'scheduled': 0, 'completed': 0, 'total': trials_per_config}
                summaries.append({'config_id': config_id, 'source': source.name, 'lag': lag, 'scheduled': trials_per_config})

        # Set exploration active flag so UI can show progress
        self.exploration_active = True
        return summaries

    def queue_explore_configs(self, sources, lags, trials_per_config=10, modes=('adaptive','probe'), config_prefix='expl'):
        """Enqueue configs for sequential exploration across modes.

        Each enqueued item is: {config_id, source, mode, lags, trials_per_config, spacing}
        """
        for s_i, source in enumerate(sources):
            for mode in modes:
                for lag in lags:
                    config_id = f"{config_prefix}_{s_i}_{mode}_lag{lag}"
                    self.exploration_queue.append({
                        'config_id': config_id,
                        'source': source,
                        'mode': mode,
                        'lag_list': [lag],
                        'trials_per_config': int(trials_per_config),
                        'spacing': 1
                    })

    def start_next_config(self):
        """Start the next queued config (runs one source/mode/lag set)."""
        if not self.exploration_queue:
            return None

        cfg = self.exploration_queue.popleft()
        self.current_config = cfg

        # Apply experiment and mode
        self.experiment = ExperimentDescriptor(exp_id=cfg['config_id'], method='explore', hidden_source=cfg['source'])
        self.mode = cfg['mode']

        # Reset per-config stats
        self.start_calibration()

        # Run exploration suite for this single config (lag list contains one lag)
        self.exploration_tasks.clear()
        self.exploration_progress.clear()
        self.run_exploration_suite([cfg['source']], cfg['lag_list'], trials_per_config=cfg['trials_per_config'], spacing=cfg['spacing'], config_prefix=cfg['config_id'])
        return cfg

    def finalize_current_config(self):
        """Called when a config finishes: record summary and persist it."""
        if not self.current_config:
            return
        cfg = self.current_config
        # Compute binomial CI for this config's results
        trials = self.trials_run
        hits = self.hits
        bt = binomtest(k=hits, n=trials) if trials > 0 else None
        conf_level = 0.95
        try:
            ci_low, ci_high = bt.proportion_ci(confidence_level=conf_level) if bt is not None else (0.0, 1.0)
        except Exception:
            ci_low, ci_high = (max(0, (hits / trials) - 0.05) if trials else 0.0, min(1, (hits / trials) + 0.05) if trials else 1.0)

        # Labeling using candidate threshold
        candidate_threshold = 0.55
        if ci_low <= 0.5 <= ci_high:
            label = 'uncoupled'
        elif ci_low >= candidate_threshold:
            label = 'candidate'
        else:
            label = 'inconclusive'

        summary = {
            'config_id': cfg['config_id'],
            'source': cfg['source'].name,
            'mode': cfg['mode'],
            'trials': trials,
            'hits': hits,
            'accuracy': self.get_accuracy(),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
            'conf_level': conf_level,
            'label': label,
            'timestamp': time.time()
        }
        self.exploration_summaries.append(summary)
        # Persist
        out_path = os.path.join(self.log_dir, 'calibration_explore_results.jsonl')
        try:
            with open(out_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(summary) + "\n")
        except Exception:
            print(f"[CALIBRATION] Failed to write explore summary to {out_path}")

        self.current_config = None

    def run_queued_exploration_blocking(self, max_ticks_per_config=10000, tick_advance=1):
        """Run queued exploration configs synchronously (useful for tests/CI).

        This will sequentially start each queued config and drive scheduling and
        pending resolution without relying on the tkinter UI loop. It advances
        the main RNG's `total_bits` by `tick_advance` and, if available,
        consumes bits from rng_app.get_random_bit() to populate rng_app.window so
        pending probes can resolve.
        """
        # Ensure we have an rng_app with minimal required attributes
        if not hasattr(self.rng_app, 'total_bits'):
            setattr(self.rng_app, 'total_bits', 0)
        if not hasattr(self.rng_app, 'window'):
            setattr(self.rng_app, 'window', deque(maxlen=1024))

        results = []
        while self.exploration_queue:
            cfg = self.start_next_config()
            print(f"[CALIBRATION] Blocking start config {cfg['config_id']}")

            ticks = 0
            # Continue until per-config scheduled trials completed and no pending probes
            while True:
                # Safety guard
                ticks += 1
                if ticks > max_ticks_per_config:
                    print(f"[CALIBRATION] Max ticks exceeded for config {cfg['config_id']}")
                    break

                # Schedule next probe if any
                scheduled = self.step_exploration()

                # Advance RNG by tick_advance bits so pending probes can resolve
                for _ in range(tick_advance):
                    # Prefer using provided get_random_bit if available
                    bit = None
                    if hasattr(self.rng_app, 'get_random_bit') and callable(self.rng_app.get_random_bit):
                        try:
                            bit = int(self.rng_app.get_random_bit())
                        except Exception:
                            bit = int.from_bytes(os.urandom(1), 'big') & 1
                    else:
                        bit = int.from_bytes(os.urandom(1), 'big') & 1

                    # Append to window and increment total_bits
                    if isinstance(self.rng_app.window, deque):
                        self.rng_app.window.append(bit)
                    else:
                        self.rng_app.window = deque(list(self.rng_app.window) + [bit], maxlen=1024)
                    self.rng_app.total_bits = getattr(self.rng_app, 'total_bits', 0) + 1

                # Try to resolve pending probes
                resolved = self.process_pending_probes()

                # If all scheduled and none pending, finalize config
                completed = sum(v['completed'] for v in self.exploration_progress.values())
                total = sum(v['total'] for v in self.exploration_progress.values())
                pending = len(self.pending_probes)
                if total > 0 and completed >= total and pending == 0 and not self.exploration_active:
                    # finalize and break to next queued config
                    self.finalize_current_config()
                    results.append(self.exploration_summaries[-1])
                    break

            # Small pause to be polite (no-op in tests, but keeps console readable)
        return results

    def step_exploration(self):
        """Schedule a single probe from exploration tasks (called per UI tick)."""
        if not self.exploration_active:
            return None

        # Find the next task with trials_left > 0
        for task in self.exploration_tasks:
            if task['trials_left'] <= 0:
                continue

            # Schedule one probe for this task
            source = task['source']
            lag = task['lag']
            cfg = task['config_id']

            # Determine realistic absolute target_total based on current total_bits + lag + next_offset
            cur_total = getattr(self.rng_app, 'total_bits', 0)
            target_total = cur_total + lag + task.get('next_offset', 0)

            hidden = source.get_bit(self.rng_app)
            self.pending_probes.append({
                'hidden': int(hidden),
                'target_total': int(target_total),
                'config': cfg,
                'timestamp': time.time()
            })
            try:
                print(f"[EXPLORER] Scheduled {cfg} target_total={target_total} hidden={hidden}")
            except Exception:
                pass

            task['trials_left'] -= 1
            task['next_offset'] += task.get('spacing', 1)
            # Update scheduled counter
            self.exploration_progress[cfg]['scheduled'] += 1

            # If all tasks are done, we just return None; do NOT clear exploration_active.
            # exploration_active remains True until all probes are resolved and the config is finalized in update_ui.
            return {'config': cfg, 'target_total': target_total}

        # No tasks with trials left: just return None but keep exploration_active True 
        # so process_pending_probes continues to be called in update_ui until resolve.
        return None

    def get_universe_label(self, rng_state, hidden_bit, raw_prediction):
        """Automated 'Universe' oracle — returns a deterministic label per trial.

        For now this is a lightweight deterministic hash-based label so it doesn't
        require human input and doesn't mutate any internal training state.
        """
        data = f"{rng_state.get('walk',0)}|{rng_state.get('p',0.5)}|{hidden_bit}|{raw_prediction}".encode()
        h = hashlib.sha256(data).digest()
        # Return a simple bit label (0/1) as integer
        return int(h[0] & 1)

    def record_translation(self, entry):
        """Persist a translation entry (JSON lines)."""
        try:
            with open(self.translations_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            # Best-effort logging; don't crash calibration loop
            print(f"[CALIBRATION] Failed to write translation to {self.translations_path}")

    def get_translation_count(self):
        """Return number of recorded translations (fast count)."""
        if not os.path.exists(self.translations_path):
            return 0
        try:
            # Count lines
            with open(self.translations_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception:
            return 0
    
    def get_rng_state(self):
        """Extract current RNG state"""
        return {
            'walk': self.rng_app.cumulative_val,
            'entropy': self.calculate_entropy(),
            'fft_anomaly': self.check_recent_fft_anomaly(),
            'p': self.rng_app.p_history[-1] if self.rng_app.p_history else 0.5
        }
    
    def calculate_entropy(self):
        """Calculate current entropy"""
        if not self.rng_app.p_history:
            return 1.0
        p = self.rng_app.p_history[-1]
        if p <= 0 or p >= 1:
            return 0
        return -(p * np.log2(p) + (1-p) * np.log2(1-p))
    
    def check_recent_fft_anomaly(self):
        """Check if FFT anomaly occurred recently"""
        if not self.rng_app.anomaly_db:
            return False
        
        now = time.time()
        for anomaly in reversed(self.rng_app.anomaly_db[-10:]):
            if now - anomaly['timestamp'] > 3:
                break
            if anomaly['type'].startswith('FFT-'):
                return True
        return False
    
    def get_accuracy(self):
        """Get overall accuracy"""
        if self.trials_run == 0:
            return 0.5
        return self.hits / self.trials_run
    
    def get_recent_accuracy(self):
        """Get accuracy of last 10 trials"""
        if not self.recent_results:
            return 0.5
        return sum(self.recent_results) / len(self.recent_results)
    
    def get_stats(self):
        """Get current calibration statistics"""
        return {
            'trials': self.trials_run,
            'hits': self.hits,
            'accuracy': self.get_accuracy(),
            'recent_accuracy': self.get_recent_accuracy(),
            'training_examples': len(self.oracle.training_data)
        }

    def run_calibration_phase(self, n_trials=2000, conf_level=0.95, candidate_threshold=0.55, write_summary=True):
        """Run a true calibration (probe mode) phase.

        - Switches to probe mode (no learning).
        - Runs exactly `n_trials` probe trials (writing per-trial translations as already implemented).
        - Computes binomial confidence interval for the observed accuracy (Clopper-Pearson via binomtest).
        - Labels channel:
            - 'uncoupled' if the CI contains 0.5
            - 'candidate' if CI lower bound >= candidate_threshold
            - otherwise 'inconclusive'
        - Optionally writes a JSON summary line to `calibration_phases.jsonl` in `log_dir`.

        Returns a dict with summary statistics and label.
        """
        # Ensure probe mode and active state during the run
        prev_mode = self.mode
        prev_active = self.active
        self.mode = 'probe'
        self.start_calibration()

        trials_done = 0
        hits = 0

        for i in range(n_trials):
            res = self.run_trial()
            if res is None:
                break
            trials_done += 1
            if res.get('correct'):
                hits += 1

        # Compute CI
        # Use scipy.stats.binomtest to get Clopper-Pearson CI
        bt = binomtest(k=hits, n=trials_done)
        try:
            ci_low, ci_high = bt.proportion_ci(confidence_level=conf_level)
        except Exception:
            # Fallback: approximate Wilson interval
            ci_low, ci_high = (max(0, (hits / trials_done) - 0.05), min(1, (hits / trials_done) + 0.05))

        accuracy = hits / trials_done if trials_done else 0.0

        # Labeling
        if ci_low <= 0.5 <= ci_high:
            label = 'uncoupled'
        elif ci_low >= candidate_threshold:
            label = 'candidate'
        else:
            label = 'inconclusive'

        summary = {
            'exp_id': self.experiment.id if self.experiment else 'unknown',
            'hidden_source': self.experiment.hidden_source.name if self.experiment else None,
            'trials': trials_done,
            'hits': hits,
            'accuracy': accuracy,
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
            'conf_level': conf_level,
            'label': label,
            'timestamp': time.time()
        }

        if write_summary:
            phases_path = os.path.join(self.log_dir, 'calibration_phases.jsonl')
            try:
                with open(phases_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(summary) + "\n")
            except Exception:
                print(f"[CALIBRATION] Failed to write phase summary to {phases_path}")

        # Restore previous mode/active state
        self.mode = prev_mode
        if not prev_active:
            self.stop_calibration()

        return summary


class CalibrationWindow:
    """Popup window for multi-dimensional calibration search"""
    
    def __init__(self, parent, rng_app):
        self.rng_app = rng_app
        self.calibration = CalibrationMode(rng_app)
        
        # Create popup window
        self.window = tk.Toplevel(parent)
        self.window.title("Calibration Search Engine")
        self.window.geometry("550x700")
        self.window.configure(bg="#121212")
        self.window.transient(parent)
        
        self.build_ui()
        
    def build_ui(self):
        """Build the calibration UI"""
        
        # Header
        header = tk.Label(self.window, text="🔬 CALIBRATION SEARCH", 
                         font=("Arial", 16, "bold"), bg="#121212", fg="#00ff00")
        header.pack(pady=10)
        
        # Description
        desc = tk.Label(self.window, 
                       text="Search parameter space for channels showing better-than-chance performance",
                       font=("Arial", 9), bg="#121212", fg="#aaaaaa", wraplength=500)
        desc.pack(pady=5)
        
        # Controls
        controls = tk.Frame(self.window, bg="#1e1e1e", bd=2, relief=tk.RAISED)
        controls.pack(fill=tk.X, padx=10, pady=10)
        
        # Hidden-bit source selection
        source_frame = tk.Frame(controls, bg="#1e1e1e")
        source_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(source_frame, text="Hidden-Bit Source:", bg="#1e1e1e",
                fg="#aaaaaa", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        self.source_var = tk.StringVar(value="Main RNG Stream")
        source_menu = tk.OptionMenu(source_frame, self.source_var,
                                   "Main RNG Stream", "Numpy PRNG", 
                                   "Numpy Lag 3",
                                   "Hash Timestamp", "Mandelbrot Fractal")
        source_menu.config(bg="#2d2d2d", fg="white", highlightthickness=0)
        source_menu.pack(side=tk.LEFT, padx=5)
        
        # Mode selection (Adaptive vs Probe)
        tk.Label(source_frame, text="Mode:", bg="#1e1e1e",
            fg="#aaaaaa", font=("Arial", 10)).pack(side=tk.LEFT, padx=(15,5))
        self.mode_var = tk.StringVar(value="Adaptive")
        mode_menu = tk.OptionMenu(source_frame, self.mode_var, "Adaptive", "Probe")
        mode_menu.config(bg="#2d2d2d", fg="white", highlightthickness=0)
        mode_menu.pack(side=tk.LEFT, padx=5)
        
        self.start_btn = tk.Button(controls, text="START CALIBRATION", 
                                   command=self.toggle_calibration,
                                   bg="#008000", fg="white", font=("Arial", 12, "bold"),
                                   width=20, height=2)
        self.start_btn.pack(pady=10)
        
        # Exploration controls: run automated suite
        exp_frame = tk.Frame(controls, bg="#1e1e1e")
        exp_frame.pack(fill=tk.X, padx=10, pady=4)

        tk.Label(exp_frame, text="Lags (csv):", bg="#1e1e1e", fg="#aaaaaa").pack(side=tk.LEFT)
        self.lags_entry = tk.Entry(exp_frame, width=12, bg="#2d2d2d", fg="white")
        self.lags_entry.insert(0, "1,2,3")
        self.lags_entry.pack(side=tk.LEFT, padx=4)

        tk.Label(exp_frame, text="Trials/config:", bg="#1e1e1e", fg="#aaaaaa").pack(side=tk.LEFT, padx=(8,4))
        self.trials_entry = tk.Entry(exp_frame, width=6, bg="#2d2d2d", fg="white")
        self.trials_entry.insert(0, "10")
        self.trials_entry.pack(side=tk.LEFT)

        self.explore_btn = tk.Button(exp_frame, text="Run Explore", command=self.run_explore_ui,
                         bg="#444499", fg="white", font=("Arial", 10))
        self.explore_btn.pack(side=tk.LEFT, padx=6)

        self.explore_all_btn = tk.Button(exp_frame, text="Explore All Sources & Modes", command=self.run_explore_all_ui,
                         bg="#664488", fg="white", font=("Arial", 10))
        self.explore_all_btn.pack(side=tk.LEFT, padx=6)
        
        # Stats Display
        stats_frame = tk.LabelFrame(self.window, text="Calibration Stats",
                                   bg="#1e1e1e", fg="#ffaa00", font=("Arial", 10, "bold"))
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.stats_label = tk.Label(stats_frame, text="Not running", 
                                    font=("Courier", 10), bg="#1e1e1e", fg="#00ff00",
                                    justify=tk.LEFT, anchor=tk.W)
        self.stats_label.pack(fill=tk.X, padx=10, pady=10)
        
        # Last hidden bit display (prominent)
        self.last_hidden_label = tk.Label(stats_frame, text="Hidden: -", 
                          font=("Courier", 12, "bold"), bg="#1e1e1e", fg="#00ffff",
                          justify=tk.LEFT, anchor=tk.W)
        self.last_hidden_label.pack(fill=tk.X, padx=10)

        # Translation count
        self.trans_count_label = tk.Label(stats_frame, text="Translations: 0", 
                          font=("Courier", 10), bg="#1e1e1e", fg="#ffaa00",
                          justify=tk.LEFT, anchor=tk.W)
        self.trans_count_label.pack(fill=tk.X, padx=10, pady=(0,6))
        
        # Exploration status label
        self.explore_status_label = tk.Label(stats_frame, text="Exploring: 0/0 completed | Pending: 0",
                             font=("Courier", 10), bg="#1e1e1e", fg="#ffffff",
                             justify=tk.LEFT, anchor=tk.W)
        self.explore_status_label.pack(fill=tk.X, padx=10)

        # Pending probe debug (first few)
        self.pending_list_label = tk.Label(stats_frame, text="Pending: []",
                           font=("Courier", 9), bg="#1e1e1e", fg="#ff5555",
                           justify=tk.LEFT, anchor=tk.W)
        self.pending_list_label.pack(fill=tk.X, padx=10)
        
        # Recent Trials
        trials_frame = tk.LabelFrame(self.window, text="Recent Trials",
                                    bg="#1e1e1e", fg="#ffaa00", font=("Arial", 10, "bold"))
        trials_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tree_frame = tk.Frame(trials_frame, bg="#1e1e1e")
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.trials_tree = ttk.Treeview(tree_frame, 
                                       columns=('Trial', 'Hidden', 'Predicted', 'Result'),
                                       height=10, show='headings')
        self.trials_tree.heading('Trial', text='#')
        self.trials_tree.heading('Hidden', text='Hidden')
        self.trials_tree.heading('Predicted', text='Predicted')
        self.trials_tree.heading('Result', text='Result')
        
        self.trials_tree.column('Trial', width=60)
        self.trials_tree.column('Hidden', width=80)
        self.trials_tree.column('Predicted', width=80)
        self.trials_tree.column('Result', width=100)
        
        self.trials_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, 
                              command=self.trials_tree.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.trials_tree.configure(yscrollcommand=scroll.set)
        
        # Start update loop
        self.update_ui()
    
    def toggle_calibration(self):
        """Start or stop calibration"""
        if not self.calibration.active:
            # Create experiment with selected source
            source_map = {
                "Main RNG Stream": MainStreamSource(),
                "Numpy PRNG": NumpyPRNGSource(),
                "Numpy Lag 3": NumpyPRNGSource(),
                "Hash Timestamp": HashTimestampSource(),
                "Mandelbrot Fractal": MandelbrotSource()
            }
            
            sel = self.source_var.get()
            source = source_map[sel]
            
            method = "adaptive_oracle"
            params = {}
            if sel == "Numpy Lag 3":
                method = "lagged_probe"
                params = {'lag': 3}

            experiment = ExperimentDescriptor(
                exp_id=f"exp_{int(time.time())}",
                method=method,
                hidden_source=source,
                params=params
            )
            
            # Update calibration experiment
            # If the selected source is the Main RNG Stream, ensure the main app is running
            if isinstance(source, MainStreamSource) and not getattr(self.rng_app, 'running', False):
                # Alert the user and refuse to start calibration with main stream
                try:
                    messagebox.showerror("Calibration Error", "Main window is not running — cannot use Main RNG Stream for calibration.")
                except Exception:
                    print("[CALIBRATION] Main window not running — cannot use Main RNG Stream for calibration.")
                return

            self.calibration.experiment = experiment
            
            self.calibration.start_calibration()
            # Set mode from UI
            selected = self.mode_var.get()
            # Map display to internal value
            self.calibration.mode = 'probe' if selected == 'Probe' else 'adaptive'
            self.start_btn.config(text="STOP CALIBRATION", bg="#cc0000")
        else:
            self.calibration.stop_calibration()
            self.start_btn.config(text="START CALIBRATION", bg="#008000")
    
    def update_ui(self):
        """Update UI with current stats"""
        # NO mode override here; we respect the mode set by the explorer or the user manually
        if self.calibration.active and self.rng_app.running and self.calibration.experiment.method != 'explore':
            if self.calibration.experiment.method == 'lagged_probe':
                # For manual lagged runs, schedule one probe per tick
                lag = self.calibration.experiment.params.get('lag', 1)
                self.calibration.schedule_lagged_probe(self.calibration.experiment.hidden_source, lag, config_id="manual_lag")
                result = None # Probes are resolved via process_pending_probes
            else:
                # Run a trial only if we aren't in automated exploration mode
                result = self.calibration.run_trial()
            
            if result:
                # Add to tree
                result_str = "✓ HIT" if result['correct'] else "✗ MISS"
                self.trials_tree.insert('', 0, values=(
                    result['trial'],
                    result['hidden'],
                    result['predicted'],
                    result_str
                ))
                # Update last hidden bit display
                try:
                    self.last_hidden_label.config(text=f"Hidden: {int(result['hidden'])}")
                except Exception:
                    pass

                # Update translation count
                try:
                    self.trans_count_label.config(text=f"Translations: {self.calibration.get_translation_count()}")
                except Exception:
                    pass
                
                # Keep only last 50
                if len(self.trials_tree.get_children()) > 50:
                    self.trials_tree.delete(self.trials_tree.get_children()[-1])
            
            # Update stats
            stats = self.calibration.get_stats()
            text = f"Trials: {stats['trials']}\n"
            text += f"Mode: {'Probe' if self.calibration.mode=='probe' else 'Adaptive'}\n"
            text += f"Hits: {stats['hits']}\n"
            text += f"Overall Accuracy: {stats['accuracy']:.1%}\n"
            text += f"Recent Accuracy (last 10): {stats['recent_accuracy']:.1%}\n"
            text += f"Training Examples: {stats['training_examples']}\n"
            self.stats_label.config(text=text)
        
        # Advance exploration scheduling one step per UI tick if active
        if self.calibration.exploration_active:
            try:
                scheduled = self.calibration.step_exploration()
                if scheduled:
                    # update UI status label
                    pending = len(self.calibration.pending_probes)
                    completed = sum(v['completed'] for v in self.calibration.exploration_progress.values())
                    total = sum(v['total'] for v in self.calibration.exploration_progress.values())
                    self.explore_status_label.config(text=f"Exploring: {completed}/{total} completed | Pending: {pending}")
                # Also try to resolve any pending probes now that time has advanced
                resolved = self.calibration.process_pending_probes()
                for r in resolved:
                    # Show resolved lagged probes in the trials tree
                    result_str = "✓ HIT" if r['correct'] else "✗ MISS"
                    self.trials_tree.insert('', 0, values=(
                        self.calibration.trials_run,
                        r['hidden'],
                        r['predicted'],
                        result_str
                    ))
                    # Update last hidden and translations count
                    try:
                        self.last_hidden_label.config(text=f"Hidden: {int(r['hidden'])}")
                        self.trans_count_label.config(text=f"Translations: {self.calibration.get_translation_count()}")
                    except Exception:
                        pass
            except Exception:
                pass

        # Also update exploration status even if not scheduling
        try:
            pending = len(self.calibration.pending_probes)
            completed = sum(v['completed'] for v in self.calibration.exploration_progress.values())
            total = sum(v['total'] for v in self.calibration.exploration_progress.values())
            self.explore_status_label.config(text=f"Exploring: {completed}/{total} completed | Pending: {pending}")
            # Show first few pending probes with distance to resolve
            try:
                cur_total = getattr(self.rng_app, 'total_bits', 0)
                sample = list(self.calibration.pending_probes)[:6]
                parts = []
                for e in sample:
                    tgt = e.get('target_total') or e.get('target_index') or None
                    dist = tgt - cur_total if tgt is not None else None
                    parts.append(f"{e.get('config') or '?'}:h{e.get('hidden')} in {dist}")
                self.pending_list_label.config(text="Pending: [" + ", ".join(parts) + "]")
            except Exception:
                pass
            # Try to resolve pending probes even when not scheduling
            try:
                resolved = self.calibration.process_pending_probes()
                for r in resolved:
                    result_str = "✓ HIT" if r['correct'] else "✗ MISS"
                    self.trials_tree.insert('', 0, values=(
                        self.calibration.trials_run,
                        r['hidden'],
                        r['predicted'],
                        result_str
                    ))
                    try:
                        self.last_hidden_label.config(text=f"Hidden: {int(r['hidden'])}")
                        self.trans_count_label.config(text=f"Translations: {self.calibration.get_translation_count()}")
                    except Exception:
                        pass
            except Exception:
                pass
            # If exploration has finished (all scheduled and none pending), finalize
            try:
                if total > 0 and completed >= total and pending == 0 and self.calibration.exploration_active:
                    # Finalize this config
                    self.calibration.exploration_active = False
                    self.calibration.finalize_current_config()
                    self.calibration.stop_calibration() # Ensure trials_run/active state resets
                    
                    # If more queued configs exist, start the next one automatically
                    if self.calibration.exploration_queue:
                        next_cfg = self.calibration.start_next_config()
                        try:
                            # Use a non-blocking notification or console log to avoid dialog hell
                            print(f"[CALIBRATION] Starting next config: {next_cfg['config_id']}")
                        except Exception:
                            pass
                    else:
                        # ALL EXPLORATION DONE
                        try:
                            self.show_explore_summary_popup()
                        except Exception as e:
                            print(f"[CALIBRATION] Failed to show summary popup: {e}")
                        # Update UI button state
                        try:
                            self.start_btn.config(text="START CALIBRATION", bg="#008000")
                        except Exception:
                            pass
            except Exception:
                pass
        except Exception:
            pass
        
        # Schedule next update
        self.window.after(100, self.update_ui)

    def run_explore_ui(self):
        """Parse UI inputs and run the exploration suite."""
        # Parse lags
        raw = self.lags_entry.get().strip()
        try:
            lags = [int(x.strip()) for x in raw.split(',') if x.strip()]
        except Exception:
            messagebox.showerror("Invalid input", "Lags must be comma-separated integers")
            return

        try:
            trials = int(self.trials_entry.get().strip())
        except Exception:
            messagebox.showerror("Invalid input", "Trials must be an integer")
            return

        # Build sources list (currently offer same options as the dropdown)
        source_map = {
            "Main RNG Stream": MainStreamSource(),
            "Numpy PRNG": NumpyPRNGSource(),
            "Hash Timestamp": HashTimestampSource(),
            "Mandelbrot Fractal": MandelbrotSource()
        }

        # If the user selected Main RNG, ensure main app is running
        selected = self.source_var.get()
        sources = [source_map[selected]] if selected in source_map else list(source_map.values())
        # If any selected source requires the main app running, enforce it
        # Require the main RNG to be running for exploration (predictions come from main stream)
        if not getattr(self.rng_app, 'running', False):
            try:
                messagebox.showerror("Calibration Error", "Main window is not running — start the main RNG before exploring.")
            except Exception:
                print("[CALIBRATION] Main window not running — cannot explore.")
            return

        self.calibration.run_exploration_suite(sources, lags, trials_per_config=trials)
        # Give user basic feedback
        messagebox.showinfo("Explore started", f"Scheduled exploration: {len(sources)} sources x {len(lags)} lags x {trials} trials each")

    def run_explore_all_ui(self):
        """Build and enqueue all sources x modes configs and start first one."""
        raw = self.lags_entry.get().strip()
        try:
            lags = [int(x.strip()) for x in raw.split(',') if x.strip()]
        except Exception:
            messagebox.showerror("Invalid input", "Lags must be comma-separated integers")
            return

        try:
            trials = int(self.trials_entry.get().strip())
        except Exception:
            messagebox.showerror("Invalid input", "Trials must be an integer")
            return

        source_map = {
            "Main RNG Stream": MainStreamSource(),
            "Numpy PRNG": NumpyPRNGSource(),
            "Hash Timestamp": HashTimestampSource(),
            "Mandelbrot Fractal": MandelbrotSource()
        }

        # Require main RNG running
        if not getattr(self.rng_app, 'running', False):
            try:
                messagebox.showerror("Calibration Error", "Main window is not running — start the main RNG before exploring.")
            except Exception:
                print("[CALIBRATION] Main window not running — cannot explore.")
            return

        sources = list(source_map.values())
        # Enqueue configs for both modes
        self.calibration.queue_explore_configs(sources, lags, trials_per_config=trials, modes=('adaptive','probe'))
        print(f"[CALIBRATION] Enqueued {len(self.calibration.exploration_queue)} configs for Explore All")
        # Start first config if none currently active
        if not self.calibration.exploration_active and self.calibration.exploration_queue:
            cfg = self.calibration.start_next_config()
            try:
                messagebox.showinfo("Explore All", f"Starting exploration: {cfg['config_id']}")
            except Exception:
                print(f"[CALIBRATION] Starting exploration: {cfg['config_id']}")

    def show_explore_summary_popup(self):
        """Popup that shows a table of all exploration summaries and allows CSV export."""
        summaries = list(self.calibration.exploration_summaries)
        if not summaries:
            try:
                messagebox.showinfo("Explore Summary", "No exploration summaries available.")
            except Exception:
                print("[CALIBRATION] No exploration summaries available.")
            return

        popup = tk.Toplevel(self.window)
        popup.title("Exploration Summary")
        popup.geometry("700x400")

        header = tk.Label(popup, text="Exploration Results", font=("Arial", 12, "bold"))
        header.pack(pady=6)

        cols = ("Config", "Source", "Mode", "Trials", "Hits", "Accuracy", "CI Low", "CI High", "Label", "Timestamp")
        tree = ttk.Treeview(popup, columns=cols, show='headings')
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=100)
        tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        for s in summaries:
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(s.get('timestamp', time.time())))
            tree.insert('', 'end', values=(
                s.get('config_id'), s.get('source'), s.get('mode'), s.get('trials'), s.get('hits'), f"{s.get('accuracy',0):.1%}",
                f"{s.get('ci_low',0):.3f}", f"{s.get('ci_high',1):.3f}", s.get('label',''), ts
            ))

        btn_frame = tk.Frame(popup)
        btn_frame.pack(fill=tk.X, pady=6)

        def export_csv():
            out_csv = os.path.join(self.calibration.log_dir, 'calibration_explore_results.csv')
            try:
                with open(out_csv, 'w', encoding='utf-8') as f:
                    f.write(','.join(cols) + '\n')
                    for s in summaries:
                        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(s.get('timestamp', time.time())))
                        row = [
                            str(s.get('config_id','')), str(s.get('source','')), str(s.get('mode','')),
                            str(s.get('trials','')), str(s.get('hits','')),
                            f"{s.get('accuracy',0):.6f}", f"{s.get('ci_low',0):.6f}", f"{s.get('ci_high',1):.6f}", s.get('label',''), ts
                        ]
                        f.write(','.join(row) + '\n')
                try:
                    messagebox.showinfo("Export CSV", f"Exported to {out_csv}")
                except Exception:
                    print(f"[CALIBRATION] Exported to {out_csv}")
            except Exception as e:
                try:
                    messagebox.showerror("Export Failed", str(e))
                except Exception:
                    print(f"[CALIBRATION] Export failed: {e}")

        exp_btn = tk.Button(btn_frame, text="Export CSV", command=export_csv, bg="#2266aa", fg="white")
        exp_btn.pack(side=tk.RIGHT, padx=6)
