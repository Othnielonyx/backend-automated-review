import os
import re
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

samples = [
    # Good codes
    ("def add(a, b):\n    return a + b\n# Addition function", 1),
    ("# Calculate area\ndef area(length, width):\n    return length * width", 1),
    ("# Sum numbers\ndef sum_list(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total", 1),
    # Bad codes
    ("def add(a,b):return a+b", 0),
    ("def foo():pass", 0),
    ("def buggy(x): return x*2 #TODO: fix later", 0),
]

texts, labels = zip(*samples)


def extract_features(code_text):
    num_functions = len(re.findall(r'def ', code_text))
    num_comments = len(re.findall(r'#', code_text))
    num_todos = len(re.findall(r'TODO', code_text))
    total_lines = len(code_text.splitlines())
    comment_ratio = num_comments / total_lines if total_lines > 0 else 0
    return [num_functions, num_comments, num_todos, total_lines, comment_ratio]


X = np.array([extract_features(code) for code in texts])
y = np.array(labels)


model = LogisticRegression()
model.fit(X, y)


joblib.dump(model, 'ml_model.joblib')
print("✅ ML model trained and saved as 'ml_model.joblib'")
