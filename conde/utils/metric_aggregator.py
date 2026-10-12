from collections import defaultdict

def nested_defaultdict():
    return defaultdict(nested_defaultdict)

class MetricAggregator:
    def __init__(self):
        self.sum = nested_defaultdict()
        self.count = nested_defaultdict()

    def update(self, stats_dict, n=1):
        self._update_recursive(self.sum, self.count, stats_dict, n)

    def _update_recursive(self, sum_dict, count_dict, stats, n):
        for k, v in stats.items():
            if isinstance(v, dict):
                self._update_recursive(sum_dict[k], count_dict[k], v, n)
            else:
                sum_dict[k] = sum_dict.get(k, 0.0) + float(v) * n
                count_dict[k] = count_dict.get(k, 0) + n

    def compute(self):
        return self._compute_recursive(self.sum, self.count)

    def _compute_recursive(self, sum_dict, count_dict):
        out = {}
        for k in sum_dict.keys():
            if isinstance(sum_dict[k], dict):
                out[k] = self._compute_recursive(sum_dict[k], count_dict[k])
            else:
                out[k] = sum_dict[k] / max(count_dict[k], 1)
        return out

    def reset(self):
        self.sum = nested_defaultdict()
        self.count = nested_defaultdict()

    def print(self):
        stats = self.compute()
        line = self._compact_str(stats)
        print(line)
    def _compact_str(self, d, prefix=""):
        """
        Converts nested dicts to compact one-line strings.
        Example:
           total=1.23 | decomp=2.34 | group: a=1.0 b=2.0
        """
        parts = []
        for k, v in d.items():
            if isinstance(v, dict):
                sub = " ".join(f"{sk}={sv:.3f}" for sk, sv in v.items())
                parts.append(f"{k}: {sub}")
            else:
                parts.append(f"{k}={v:.3f}")
        return " | ".join(parts)