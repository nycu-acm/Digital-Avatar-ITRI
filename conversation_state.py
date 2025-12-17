from typing import Dict

# Stores the latest full AI reply text per avatar session (keyed by BaseReal.sessionid, int)
session_last_reply: Dict[int, str] = {}


