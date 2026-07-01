import json, os
from config.settings import MEMORY_FILE, MEMORY_MAX

class Memory:
    def __init__(self):
        self._file = MEMORY_FILE
        self._data = self._load()

    def _load(self):
        if os.path.exists(self._file):
            try:
                return json.loads(open(self._file).read())
            except: pass
        return []

    def save(self, role: str, content: str):
        self._data.append({"role": role, "content": content})
        if len(self._data) > MEMORY_MAX:
            self._data = self._data[-MEMORY_MAX:]
        try:
            with open(self._file, "w") as f:
                json.dump(self._data, f, indent=2)
        except: pass

    def get_history(self):  return list(self._data)
    def clear(self):
        self._data = []
        try:
            with open(self._file, "w") as f:
                json.dump(self._data, f, indent=2)
        except: pass
    def __len__(self):      return len(self._data)
