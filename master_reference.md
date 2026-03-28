# Python Reference: Artificial Intelligence & Data Science
<!-- Copilot/Gemini Context File — Use this as a knowledge base for code suggestions, explanations, and completions -->

---

## Table of Contents
1. [Data Structures](#1-data-structures)
2. [Set Operations](#2-set-operations)
3. [Functions & Modules](#3-functions--modules)
4. [Recursion Basics](#4-recursion-basics)
5. [File Handling](#5-file-handling)
6. [OOP Fundamentals](#6-oop-fundamentals)
7. [OOP Advanced](#7-oop-advanced)
8. [Visualization](#8-visualization)
9. [NumPy Basics](#9-numpy-basics)

---

## 1. Data Structures

**Subtopics:** Lists · Tuples · Sets · Dictionaries · Comprehensions

---

### Lists

**Concept:** Python's primary dynamic array. Ordered, mutable, holds heterogeneous types. Uses contiguous memory blocks holding references to objects — supports O(1) append, O(n) insert/search, zero-based indexing, and slicing.

```python
features = [0.5, 0.8, 0.1]
features.append(0.9)         # O(1) — adds to end
features.insert(0, 1.0)      # O(n) — shifts others right
popped_val = features.pop()  # removes & returns last element
features.sort(reverse=True)  # in-place sort
print(features)              # [1.0, 0.8, 0.5, 0.1]
```

**AI/ML Use:** Accumulating epoch losses, batching tokenized sequences before converting to arrays.

**Common Mistake:** Never modify a list while iterating over it — it skips elements. Iterate over a copy: `for x in my_list[:]`.

---

### Tuples

**Concept:** Ordered, **immutable** collection. Fixed after creation — hashable if all elements are hashable. Slightly less memory overhead than lists, marginally faster for fixed data access.

```python
tensor_shape = (3, 224, 224)  # Channels, Height, Width
# tensor_shape[0] = 1         # TypeError — immutable

num_channels = tensor_shape[0]
total_pixels = tensor_shape[1] * tensor_shape[2]
print(f"Channels: {num_channels}, Pixels: {total_pixels}")
# Channels: 3, Pixels: 50176
```

**AI/ML Use:** Defining tensor shapes, RGB boundaries, fixed hyperparameters passed to constructors.

**Common Mistake:** Single-element tuple requires trailing comma — `(5,)` not `(5)`. The latter is just an int.

---

### Sets

**Concept:** Unordered, mutable collection of **unique, hashable** elements. Backed by a hash table — O(1) average membership testing. No indexing or slicing.

```python
predicted_classes = {0, 1, 1, 2, 0}
print(predicted_classes)       # {0, 1, 2} — duplicates dropped

predicted_classes.add(3)
predicted_classes.remove(0)    # KeyError if missing
predicted_classes.discard(0)   # Safe — no error if missing
```

**AI/ML Use:** Building unique vocabularies, ensuring feature uniqueness, computing overlap between predicted and ground-truth labels.

**Common Mistake:** `{}` creates an empty **dict**, not a set. Use `set()` for an empty set.

---

### Dictionaries

**Concept:** Mutable key-value store. Keys must be unique and hashable. Values can be any type. Since Python 3.7, insertion order is preserved. O(1) average lookup.

```python
hyperparameters = {"learning_rate": 0.01, "batch_size": 32}
hyperparameters["epochs"] = 100

lr = hyperparameters.get("learning_rate", 0.001)   # safe retrieval
momentum = hyperparameters.get("momentum", 0.9)    # returns default 0.9

print(hyperparameters.keys())
# dict_keys(['learning_rate', 'batch_size', 'epochs'])
```

**AI/ML Use:** Model config management, mapping feature names to label indices, hyperparameter grids for grid-search CV.

**Common Mistake:** Lists and dicts cannot be dict keys — they're mutable, so their hash would change, corrupting the hash table.

---

### Comprehensions (List / Dict / Set)

**Concept:** Concise, C-optimized syntax to generate collections from iterables. Faster than equivalent `for` loops.

```python
raw_data = [1, 2, -3, 4, -5]

# List comprehension
squares = [x**2 for x in raw_data if x > 0]        # [1, 4, 16]

# Set comprehension
abs_set = {abs(x) for x in raw_data}               # {1, 2, 3, 4, 5}

# Dict comprehension
val_map = {x: abs(x) for x in raw_data if x < 0}  # {-3: 3, -5: 5}
```

**AI/ML Use:** Preprocessing pipelines, token filtering, building reverse-index mappings for one-hot encoding.

**Common Mistake:** Deeply nested comprehensions kill readability. If it needs line breaks or nested ternaries, use a regular `for` loop.

---

### Quick Comparison Table — Data Structures

| Feature       | List              | Tuple             | Set                        | Dictionary                     |
|---------------|-------------------|-------------------|----------------------------|--------------------------------|
| Ordering      | Ordered           | Ordered           | Unordered                  | Insertion-Ordered (Python 3.7+)|
| Mutability    | Mutable           | Immutable         | Mutable                    | Mutable                        |
| Duplicates    | Allowed           | Allowed           | Not allowed                | Keys: No, Values: Yes          |
| Indexing      | Integer index     | Integer index     | None                       | Key-based                      |
| Lookup Speed  | O(n)              | O(n)              | O(1) average               | O(1) average                   |

---

### Gotchas — Data Structures

```python
# GOTCHA 1: List multiplication creates shared references, not independent copies
matrix = [[0] * 3] * 3      # All 3 rows point to SAME list object
matrix[0][0] = 1            # Corrupts ALL rows: [[1,0,0],[1,0,0],[1,0,0]]
matrix = [[0]*3 for _ in range(3)]  # CORRECT — independent rows

# GOTCHA 2: Tuple hashability depends on ALL elements being immutable
bad_key = (1, [2, 3])       # TypeError — list inside tuple is unhashable

# GOTCHA 3: Dict union (Python 3.10+) — right side wins on key collision
merged = dict_A | dict_B    # dict_B values overwrite dict_A on shared keys

# GOTCHA 4: Tuple unpacking — length must match exactly
a, b = (1, 2, 3)            # ValueError — too many values
a, *b = (1, 2, 3)           # CORRECT — a=1, b=[2,3]
```

---

### Cheat Sheet — Data Structures

```python
L = []           # List:   append(), pop(), sort() — mutable, ordered, O(n) lookup
T = (1, 2)       # Tuple:  count(), index()        — immutable, hashable
S = {1, 2}       # Set:    add(), discard()         — no duplicates, O(1) lookup
D = {'k': 'v'}   # Dict:   keys(), values(), get() — key-based O(1) lookup
[x for x in L]   # Comprehension: [expr for item in iterable if condition]
```

---

## 2. Set Operations

**Subtopics:** Union & Intersection · Difference & Symmetric Difference · Frozenset

---

### Union & Intersection

**Concept:** Union returns all unique elements across sets (OR logic). Intersection returns only elements present in ALL sets (AND logic).

```python
model_A_errors = {101, 102, 105}
model_B_errors = {102, 108, 105}

all_errors    = model_A_errors | model_B_errors   # {101, 102, 105, 108}
common_errors = model_A_errors & model_B_errors   # {102, 105}

# Method equivalents (accept any iterable, not just sets)
model_A_errors.union(model_B_errors)
model_A_errors.intersection(model_B_errors)
```

**AI/ML Use:** Intersection = consensus misclassifications across ensemble models. Union = full superset of anomalies flagged by multiple detectors.

**Common Mistake:** `|` and `&` operators require **both operands to be sets**. Use `.union()` / `.intersection()` when one operand is a list.

---

### Difference & Symmetric Difference

**Concept:** Difference = elements in A but not B (NOT logic). Symmetric difference = elements in either A or B but **not both** (XOR logic).

```python
train_ids = {1, 2, 3, 4}
val_ids   = {3, 4, 5, 6}

only_train      = train_ids - val_ids   # {1, 2}
unique_to_either = train_ids ^ val_ids  # {1, 2, 5, 6}

# Method equivalents
train_ids.difference(val_ids)
train_ids.symmetric_difference(val_ids)
```

**AI/ML Use:** Difference ensures no data leakage between train/val splits. Symmetric difference finds IDs exclusive to one temporal split.

**Common Mistake:** `A - B` ≠ `B - A`. Difference is **not commutative**.

---

### Frozenset

**Concept:** Immutable, read-only variant of `set`. Because Python guarantees immutability, frozensets are **hashable** — usable as dictionary keys or nested inside other sets.

```python
stop_words = frozenset(["the", "and", "a", "of"])
# stop_words.add("an")  # AttributeError — no mutation allowed

# Can be used as a dict key
vectorizer_configs = {stop_words: "basic_english_filter"}
print(vectorizer_configs[stop_words])  # basic_english_filter
```

**AI/ML Use:** Locking preprocessing vocabularies or static feature subsets to prevent accidental mutation during parallelized pipeline runs.

**Common Mistake:** `frozenset(1, 2, 3)` raises TypeError. Constructor takes exactly **one iterable**: `frozenset([1, 2, 3])`.

---

### Quick Comparison Table — Set Operations

| Operation         | Operator | Method                        | Logic                                    |
|-------------------|----------|-------------------------------|------------------------------------------|
| Union             | `A \| B` | `A.union(B)`                  | OR — elements in A, B, or both           |
| Intersection      | `A & B`  | `A.intersection(B)`           | AND — elements in both A and B           |
| Difference        | `A - B`  | `A.difference(B)`             | NOT — elements in A, absent from B       |
| Sym. Difference   | `A ^ B`  | `A.symmetric_difference(B)`   | XOR — in A or B, but not both            |

---

### Gotchas — Set Operations

```python
# GOTCHA 1: Operators enforce strict set types; methods accept any iterable
set_A | [1, 2]          # TypeError
set_A.union([1, 2])     # Works fine

# GOTCHA 2: .symmetric_difference() only takes ONE argument
A ^ B ^ C               # Works — chaining operators is fine
A.symmetric_difference(B, C)  # TypeError — chain methods instead

# GOTCHA 3: Empty frozenset is a singleton in memory (Python optimization)
frozenset() is frozenset()  # True — same object reused
```

---

### Cheat Sheet — Set Operations

```python
A, B = {1, 2}, {2, 3}
A | B   # Union         -> {1, 2, 3}
A & B   # Intersection  -> {2}
A - B   # Difference    -> {1}
A ^ B   # Sym. Diff     -> {1, 3}
fs = frozenset([1, 2])  # Immutable, hashable — safe as dict key
```

---

## 3. Functions & Modules

**Subtopics:** Defining Functions & Scope (LEGB) · `*args` and `**kwargs` · `lambda`, `map`, `filter` · Imports & `__name__`

---

### Defining Functions & Scope (LEGB)

**Concept:** LEGB = the order Python resolves variable names: **L**ocal → **E**nclosing → **G**lobal → **B**uilt-in.

```python
multiplier = 2   # Global scope

def scaling_pipeline(data):
    offset = 10  # Enclosing scope for apply_scale

    def apply_scale(x):
        return (x * multiplier) + offset  # reads Global and Enclosing

    return [apply_scale(d) for d in data]

print(scaling_pipeline([1, 2]))  # [12, 14]
```

**AI/ML Use:** Encapsulating custom loss functions and data augmentation pipelines. Prevents local training loop variables from corrupting global hyperparameters.

**Common Mistake:** `UnboundLocalError` — if Python sees any assignment to a name inside a function, it treats it as local everywhere in that function. Use `global variable_name` to explicitly modify a global.

---

### `*args` and `**kwargs`

**Concept:** `*args` collects extra positional arguments into a **tuple**. `**kwargs` collects extra keyword arguments into a **dict**.

```python
def build_model(architecture, *layers, **hyperparams):
    print(f"Arch: {architecture}")
    print(f"Layers (tuple): {layers}")
    print(f"Hyperparams (dict): {hyperparams}")

build_model("CNN", 64, 128, 256, learning_rate=0.01, dropout=0.5)
# Arch: CNN
# Layers (tuple): (64, 128, 256)
# Hyperparams (dict): {'learning_rate': 0.01, 'dropout': 0.5}
```

**AI/ML Use:** Framework constructors (Scikit-Learn, TensorFlow) use this pattern so layer initializers can absorb varied hyperparameter configs without long hardcoded signatures.

**Common Mistake:** Parameter order is strict: `(required, *args, defaults, **kwargs)`. `**kwargs` must always be last.

---

### `lambda`, `map`, and `filter`

**Concept:** `lambda` creates anonymous single-expression functions. `map` applies a function to every element. `filter` returns elements where the function evaluates to `True`.

```python
probabilities = [0.1, 0.8, 0.4, 0.9]

predictions = list(map(lambda p: p > 0.5, probabilities))
# [False, True, False, True]

high_conf = list(filter(lambda p: p > 0.85, probabilities))
# [0.9]
```

**AI/ML Use:** Fast element-wise transforms — mapping string labels to integers, filtering noisy data points before matrix ingestion.

**Common Mistake:** `map()` and `filter()` return **lazy iterators** in Python 3, not lists. Always wrap with `list()` to materialize results.

---

### Imports & `__name__ == "__main__"`

**Concept:** `if __name__ == "__main__":` is an execution guard — the block runs only when the script is executed directly, not when imported as a module.

```python
import math
from math import sqrt as square_root

def calculate_distance(p1, p2):
    return square_root((p1 - p2)**2)

if __name__ == "__main__":
    print(f"Test Distance: {calculate_distance(10, 5)}")  # 5.0
```

**AI/ML Use:** Ensures training loops and data parsers don't accidentally run when an API server imports the file for prediction functions only.

**Common Mistake:** Never name your script the same as a library (e.g., `math.py`, `numpy.py`). Python checks local directories first — your script shadows the real library.

---

### Gotchas — Functions & Modules

```python
# GOTCHA 1: Mutable default arguments are created ONCE at function definition
def append_to(item, my_list=[]):   # Bug — same list reused every call
    my_list.append(item)
    return my_list

def append_to(item, my_list=None): # Correct
    if my_list is None:
        my_list = []
    my_list.append(item)
    return my_list

# GOTCHA 2: Lambda late binding in loops
funcs = [lambda: i for i in range(3)]
funcs[0]()  # Returns 2, not 0 — all lambdas capture the same 'i'
# Fix: lambda i=i: i  (bind at creation time)

# GOTCHA 3: Shadowing built-ins
list = [1, 2, 3]   # Now list() constructor is broken in this scope
dict = {}           # Same problem with dict()
```

---

### Cheat Sheet — Functions & Modules

```python
def func(a, b=2, *args, **kwargs): pass  # Strict parameter order

# LEGB: Local -> Enclosing -> Global -> Built-in

mapped   = list(map(lambda x: x*2, [1, 2, 3]))   # [2, 4, 6]
filtered = list(filter(lambda x: x > 0, [-1, 1])) # [1]

if __name__ == "__main__": pass   # Execution guard
```

---

## 4. Recursion Basics

**Subtopics:** Base Case & Recursive Case · Call Stack Concept · Simple Examples

---

### Base Case & Recursive Case

**Concept:** A function that calls itself to solve progressively smaller subproblems. Requires two parts:
- **Base case** — stopping condition, returns a concrete value
- **Recursive case** — calls itself with a strictly smaller input

```python
def factorial(n):
    if n == 1 or n == 0:     # Base case
        return 1
    return n * factorial(n - 1)  # Recursive case

print(factorial(5))  # 120
```

**AI/ML Use:** Traversing decision trees to compute Gini impurity, parsing nested JSON configs, DFS in reinforcement learning grids.

**Common Mistake:** Forgetting the base case, or recursive case not moving toward it → infinite recursion → memory exhaustion.

---

### Call Stack Concept

**Concept:** Each recursive call pushes a new **frame** onto the call stack (allocating memory for local variables). When the base case is hit, the stack **unwinds** — frames are popped in reverse order, returning computed values up the chain.

```python
def countdown(n):
    print(f"Pushing: {n}")
    if n <= 0:
        print("Base case! Unwinding...")
        return
    countdown(n - 1)
    print(f"Popping: {n}")

countdown(2)
# Pushing: 2 → Pushing: 1 → Pushing: 0
# Base case! Unwinding...
# Popping: 1 → Popping: 2
```

**AI/ML Use:** Backpropagation mirrors call stack unwinding — computing gradients backward through layers using the chain rule.

**Common Mistake:** `RecursionError` (stack overflow) — Python's default limit is ~1000 frames. Refactor deep problems into iterative loops.

---

### Quick Comparison Table — Recursion vs Iteration

| Feature         | Recursion                              | Iteration (Loops)                  |
|-----------------|----------------------------------------|------------------------------------|
| Mechanics       | Function calls itself                  | Loop repeats a code block          |
| State Tracking  | Managed by the call stack              | Managed via loop variables         |
| Memory Overhead | High (new frame per call)              | Minimal (reuses same memory)       |
| Best Use Cases  | Trees, graphs, divide-and-conquer      | Sequential arrays, long pipelines  |

---

### Gotchas — Recursion

```python
# GOTCHA 1: Naive Fibonacci is O(2^n) — recomputes same subproblems
def fib(n):
    if n <= 1: return n
    return fib(n-1) + fib(n-2)   # Exponentially slow

# Fix: use memoization
from functools import lru_cache
@lru_cache(maxsize=None)
def fib(n):
    if n <= 1: return n
    return fib(n-1) + fib(n-2)

# GOTCHA 2: Missing return keyword when unwinding
def factorial(n):
    if n == 0: return 1
    factorial(n - 1)    # Bug — return value is dropped, returns None
    return n * factorial(n - 1)  # Correct
```

---

### Cheat Sheet — Recursion

```python
def recurse(n):
    if base_condition(n):   # Base case — stops recursion
        return concrete_value
    return recurse(n - 1)   # Recursive case — moves toward base

# Max recursion depth in Python: ~1000 frames (sys.getrecursionlimit())
```

---

## 5. File Handling

**Subtopics:** `open()` & Context Managers · File Modes · Reading/Writing Methods · `seek()` & `tell()`

---

### `open()` and Context Managers (`with`)

**Concept:** `open()` creates a data stream between Python and a file on disk. The `with` statement (context manager) automatically closes the stream on exit — even if an exception occurs.

```python
with open("dataset_log.txt", "w") as file:
    file.write("Epoch 1: Validation Loss 0.5\n")
# File is automatically closed here
```

**AI/ML Use:** Critical for iterating over large image directories or writing training logs. Unclosed files hit OS open-file limits and crash long training runs.

**Common Mistake:** Using `f = open(...)` without `f.close()` causes memory leaks and OS-level file locks.

---

### File Modes

**Concept:** The mode string controls read/write permissions and where the cursor starts.

```python
with open("test.txt", "w+") as f:
    f.write("Raw Data")   # w+ truncates existing content on open
    f.seek(0)
    print(f.read())       # Raw Data

with open("test.txt", "a+") as f:
    f.write(" More")      # a+ appends — existing content preserved
    f.seek(0)
    print(f.read())       # Raw Data More
```

**AI/ML Use:** `a` mode for incrementally logging epoch metrics without overwriting history. `r+` for updating serialized state files.

**Common Mistake:** Using `w` or `w+` when you meant to append — **it instantly deletes all existing content**.

---

### Quick Comparison Table — File Modes

| Mode | Read | Write | Truncates Existing? | Creates New? | Cursor Starts |
|------|------|-------|---------------------|--------------|---------------|
| `r`  | ✅   | ❌    | No                  | ❌ (Error)   | Beginning     |
| `w`  | ❌   | ✅    | **Yes**             | ✅           | Beginning     |
| `a`  | ❌   | ✅    | No                  | ✅           | End           |
| `r+` | ✅   | ✅    | No                  | ❌ (Error)   | Beginning     |
| `w+` | ✅   | ✅    | **Yes**             | ✅           | Beginning     |
| `a+` | ✅   | ✅    | No                  | ✅           | End           |

---

### Reading/Writing Methods

**Concept:**
- `read()` — entire file as one string
- `readline()` — one line per call
- `readlines()` — all lines as a list (each item has `\n` attached)
- `write()` — writes a single string
- `writelines()` — writes a list of strings (no auto newlines added)

```python
with open("temp.csv", "w") as f:
    f.writelines(["id,label\n", "1,cat\n", "2,dog\n"])

with open("temp.csv", "r") as f:
    header = f.readline()    # "id,label\n"
    rest   = f.readlines()   # ['1,cat\n', '2,dog\n']
```

**AI/ML Use:** Stream gigabyte-scale datasets line-by-line with `for line in file:` — avoids loading the whole file into RAM.

**Common Mistake:** `readline()` / `readlines()` do **not** strip `\n`. Always call `.strip()` before type conversion.

---

### `seek()` and `tell()`

**Concept:** Files have an invisible cursor. `tell()` returns its current byte position. `seek(offset)` jumps the cursor to a specific byte.

```python
with open("cursor.txt", "w+") as f:
    f.write("Machine Learning")
    print(f.tell())        # 16 — cursor is at EOF

    f.seek(0)              # Jump back to start
    print(f.read(7))       # "Machine"
```

**AI/ML Use:** Random-access reading in large binary datasets — jump directly to the byte offset of a specific image tensor without parsing preceding data.

**Common Mistake:** In `a+` mode, `seek()` works for reading, but any `write()` call **snaps the cursor back to EOF** regardless of seek position.

---

### Gotchas — File Handling

```python
# GOTCHA 1: Append mode write always goes to EOF even after seek()
with open("file.txt", "a+") as f:
    f.seek(0)
    f.write("new")   # Still appends to end — seek is ignored for writes

# GOTCHA 2: Always specify encoding to avoid cross-platform issues
with open("file.txt", "r", encoding="utf-8") as f:  # Correct
    data = f.read()

# GOTCHA 3: r+ overwrites bytes in-place, does not insert
# File contains: "Machine"
with open("file.txt", "r+") as f:
    f.write("AI")   # Result: "AIchine" — not "AI Machine"
```

---

### Cheat Sheet — File Handling

```python
with open("file.txt", "w+", encoding="utf-8") as f:
    f.write("Text")           # Single string
    f.writelines(["a", "b"])  # List of strings — no auto newlines
    f.seek(0)                 # Jump cursor to byte 0
    pos  = f.tell()           # Current byte position
    data = f.read()           # Read from cursor to EOF
# 'with' handles f.close() automatically
```

---

## 6. OOP Fundamentals

**Subtopics:** Classes, `__init__`, `self` · Instance vs Class Attributes · Instance/Class/Static Methods · Properties

---

### Classes, Objects, `__init__`, and `self`

**Concept:** A class is a blueprint; an object is its instantiated realization in memory. `__init__` initializes object state at creation. `self` explicitly refers to the current instance.

```python
class Model:
    def __init__(self, architecture):
        self.architecture = architecture  # instance attribute

    def display(self):
        print(f"Running: {self.architecture}")

resnet = Model("ResNet50")
resnet.display()  # Running: ResNet50
```

**AI/ML Use:** PyTorch layers, Scikit-Learn scalers, and custom DataLoaders are all OOP classes — encapsulation prevents models from bleeding state into each other.

**Common Mistake:** Forgetting `self` as the first parameter of an instance method → `TypeError: takes 0 positional arguments but 1 was given`.

---

### Instance vs Class Attributes

**Concept:** Instance attributes (`self.x`) are unique per object. Class attributes are defined at the class level and **shared across all instances**.

```python
class DataProcessor:
    version = "1.0"          # Class attribute — shared globally

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name  # Instance attribute — unique

proc1 = DataProcessor("Images")
proc2 = DataProcessor("Text")

DataProcessor.version = "1.1"
print(proc1.version, proc2.dataset_name)  # 1.1 Text
```

**AI/ML Use:** Instance attrs = per-model learning rates. Class attrs = global constants like version numbers or config seeds.

**Common Mistake:** Setting a **mutable** object (e.g., `[]`) as a class attribute — appending to it from one instance mutates it for ALL instances.

---

### Instance, Class, and Static Methods

**Concept:**
- **Instance method** — takes `self`, operates on the specific object
- **Class method** — takes `cls` via `@classmethod`, operates on the class itself
- **Static method** — takes neither, via `@staticmethod`, pure utility logic

```python
class Scaler:
    scale_factor = 2.0

    def __init__(self, data):
        self.data = data

    def apply_scale(self):              # Instance method
        return [x * self.scale_factor for x in self.data]

    @classmethod
    def set_factor(cls, new_val):       # Class method
        cls.scale_factor = new_val

    @staticmethod
    def validate(data):                 # Static method
        return all(isinstance(x, (int, float)) for x in data)

Scaler.set_factor(3.0)
s = Scaler([10, 20])
if Scaler.validate(s.data):
    print(s.apply_scale())  # [30.0, 60.0]
```

**AI/ML Use:** Class methods → `Model.from_pretrained("bert-base")`. Static methods → math utilities like tensor dimension validation.

**Common Mistake:** Trying to access `self.data` inside a `@staticmethod` → `NameError`. Static methods have no object reference.

---

### Properties

**Concept:** `@property` lets a method be accessed like an attribute. Pairs with `@x.setter` to intercept assignments with validation logic — without exposing raw `get_x()` / `set_x()` methods.

```python
class Threshold:
    def __init__(self):
        self._val = 0.5          # Protected backing variable

    @property
    def value(self):             # Getter
        return self._val

    @value.setter
    def value(self, new_val):    # Setter with validation
        if 0 <= new_val <= 1:
            self._val = new_val
        else:
            raise ValueError("Must be between 0 and 1.")

t = Threshold()
t.value = 0.9     # Triggers setter
print(t.value)    # 0.9
```

**AI/ML Use:** Clamping hyperparameters (e.g., blocking negative learning rates) at the property level before they corrupt a model.

**Common Mistake:** `self.value = new_val` inside the setter → infinite recursion. Always assign to the backing variable (`self._val`).

---

### Gotchas — OOP Fundamentals

```python
# GOTCHA 1: obj.__dict__ stores all instance attrs — external code can inject new ones
obj.__dict__['injected'] = "dangerous"

# GOTCHA 2: Mutable class attribute shared across ALL instances
class Model:
    layers = []   # Shared list

m1 = Model()
m1.layers.append("Dense")
m2 = Model()
print(m2.layers)  # ['Dense'] — m2 is affected!

# GOTCHA 3: Python has NO true private variables
# _var  = convention only (not enforced)
# __var = name-mangled to _ClassName__var (obscured, but still accessible)
```

---

### Cheat Sheet — OOP Fundamentals

```python
class MLTask:
    epochs = 100                       # Class attribute (shared)

    def __init__(self, data):
        self.data = data               # Instance attribute (unique)

    def process(self): pass            # Instance method — needs self
    
    @classmethod
    def update(cls): pass              # Class method — needs cls

    @staticmethod
    def is_valid(): pass               # Static method — needs neither

    @property
    def size(self): return len(self.data)  # Accessed as obj.size
```

---

## 7. OOP Advanced

**Subtopics:** Inheritance & `super()` · Method Overriding & Polymorphism · Abstraction & Encapsulation · Dunder Methods

---

### Inheritance, `super()`, and Multiple Inheritance

**Concept:** A child class inherits attributes and methods from a parent class. `super()` proxies calls to the parent's constructor or methods. Multiple inheritance allows inheriting from several parents simultaneously.

```python
class BaseLayer:
    def __init__(self, units):
        self.units = units

class DenseLayer(BaseLayer):
    def __init__(self, units, activation):
        super().__init__(units)       # Calls parent __init__
        self.activation = activation

layer = DenseLayer(64, "relu")
print(f"Units: {layer.units}, Activation: {layer.activation}")
# Units: 64, Activation: relu
```

**AI/ML Use:** PyTorch — `Conv2D`, `Dropout`, `DenseLayer` all inherit from a shared `BaseLayer` that handles weight registration and backprop hooks.

**Common Mistake:** In multiple inheritance `class C(A, B)`, Python resolves method conflicts **left to right** (MRO). Misunderstanding this causes the wrong method to run silently. Check with `C.mro()`.

---

### Method Overriding and Polymorphism

**Concept:** A child class redefines a parent's method (overriding). Polymorphism lets a single loop treat different object types through a shared interface — the interpreter picks the correct overridden method at runtime.

```python
class Metric:
    def compute(self, y_true, y_pred): pass

class Accuracy(Metric):
    def compute(self, y_true, y_pred):
        return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

class Loss(Metric):
    def compute(self, y_true, y_pred):
        return sum(abs(t - p) for t, p in zip(y_true, y_pred))

for m in [Accuracy(), Loss()]:
    print(m.compute([1, 0, 1], [1, 0, 0]))
# 0.6666... (Accuracy)
# 1         (Loss)
```

**AI/ML Use:** Training loops call `.compute()` without caring which metric object it is — polymorphism handles dispatch.

**Common Mistake:** Overriding a method but changing its parameter signature breaks the polymorphic contract → crashes in any loop using the parent's expected signature.

---

### Abstraction and Encapsulation

**Concept:** Encapsulation restricts direct access to internal state (using `_var` conventions or properties). Abstraction hides complex implementation behind a simple interface.

```python
class Optimizer:
    def __init__(self, weights):
        self._weights = weights           # Encapsulated

    def _calculate_gradients(self):       # Private logic — abstracted
        return [w * 0.1 for w in self._weights]

    def step(self):                       # Clean public interface
        grads = self._calculate_gradients()
        self._weights = [w - g for w, g in zip(self._weights, grads)]
        print(f"Updated: {self._weights}")

opt = Optimizer([1.0, 2.0])
opt.step()  # Updated: [0.9, 1.8]
```

**AI/ML Use:** Practitioners call `optimizer.step()` or `model.fit()` — complex partial derivatives and memory allocation are fully hidden.

**Common Mistake:** Directly mutating `model._learning_rate = 0.5` bypasses validation and logging, leaving the model in a broken state.

---

### Dunder Methods (`__str__`, `__repr__`, `__len__`)

**Concept:** "Magic methods" define how objects interact with Python's built-in syntax. `__str__` → user-friendly string for `print()`. `__repr__` → unambiguous developer string for debugging/REPL. `__len__` → defines `len(obj)`.

```python
class Dataset:
    def __init__(self, name, size):
        self.name = name
        self.size = size

    def __len__(self):   return self.size
    def __str__(self):   return f"Dataset: '{self.name}'"
    def __repr__(self):  return f"Dataset(name='{self.name}', size={self.size})"

data = Dataset("ImageNet_Lite", 5000)
print(data)       # Dataset: 'ImageNet_Lite'    (__str__)
repr(data)        # Dataset(name='ImageNet_Lite', size=5000)  (__repr__)
len(data)         # 5000  (__len__)
```

**AI/ML Use:** `__len__` is **mandatory** for PyTorch custom `Dataset` classes (required by `DataLoader`). `__repr__` is critical for MLflow logging to capture model states.

**Common Mistake:** Skipping `__repr__` → objects display as useless memory addresses `<__main__.Dataset object at 0x...>`.

---

### Gotchas — OOP Advanced

```python
# GOTCHA 1: super() in multiple inheritance delegates to A first, not both
class C(A, B):
    def method(self):
        super().method()  # Only calls A.method(), not B.method()
# Inspect order with: C.mro()

# GOTCHA 2: Jupyter vs script display behavior
# Jupyter end-of-cell: automatically calls __repr__
# print() in any context: calls __str__ (falls back to __repr__ if __str__ absent)

# GOTCHA 3: Name mangling — not true privacy
class Foo:
    def __init__(self):
        self.__secret = 42   # Mangled to _Foo__secret

obj = Foo()
obj._Foo__secret  # Accessible — mangling just obfuscates, not hides
```

---

### Cheat Sheet — OOP Advanced

```python
class Parent:
    def action(self): pass

class Child(Parent):
    def action(self):       # Overrides parent
        super().action()    # Still calls parent version
        pass

    def __str__(self):  pass   # Controls print(obj)
    def __repr__(self): pass   # Controls Jupyter display & logs
    def __len__(self):  pass   # Controls len(obj)
```

---

## 8. Visualization

**Subtopics:** Matplotlib Basics · Seaborn Basics · Subplots, Labels, Titles, Saving

---

### Matplotlib Basics

**Concept:** Low-level foundational plotting library. Architecture separates **Figure** (canvas) from **Axes** (individual plot). Always use the object-oriented API: `fig, ax = plt.subplots()`.

```python
import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [4, 5, 6]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y, label="Accuracy", color="blue", marker="o")
ax.bar(x, y, alpha=0.3, color="orange")
ax.scatter([1.5], [5.5], color="red", s=100)

ax.set_title("Training Metrics")
ax.set_xlabel("Epoch")
ax.set_ylabel("Score")
ax.legend()

plt.savefig("plot.png", dpi=300)   # Save BEFORE show()
plt.show()
```

**AI/ML Use:** Epoch loss curves, decision boundary scatter plots, anomaly dashboards.

**Common Mistake:** Mixing `plt.plot()` (stateful API) with `ax.plot()` (OOP API) causes state bleed across plots. Always use `ax` methods.

---

### Seaborn Basics

**Concept:** High-level statistical visualization wrapper built on top of Matplotlib. Integrates natively with Pandas DataFrames. Handles statistical aggregations automatically.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data_matrix = np.array([[1.0, 0.82], [0.82, 1.0]])

fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(data_matrix, annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Feature Correlation Heatmap")
plt.show()
```

**AI/ML Use:** `sns.heatmap()` for confusion matrices. `sns.pairplot()` for full EDA feature correlation in one call.

**Common Mistake:** Seaborn doesn't replace Matplotlib for layout control. Rotating tick labels, adjusting limits, saving files — all still require Matplotlib `ax` commands.

---

### Subplots, Labels, Titles, Saving

**Concept:** `plt.subplots(nrows, ncols)` creates a grid of axes. `plt.tight_layout()` prevents overlap. **Always `savefig()` before `show()`** — `show()` flushes the figure buffer.

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

axes[0].scatter([1, 2], [3, 4], color='purple')
axes[0].set_title("Scatter")

sns.boxplot(data=[[1, 2, 3], [4, 5, 6]], ax=axes[1])
axes[1].set_title("Boxplot")

plt.tight_layout()
plt.savefig("report.png", dpi=300)   # Save first
plt.show()                           # Then display
```

**AI/ML Use:** Side-by-side train loss vs. validation accuracy comparison — required for research reports and business analytics.

**Common Mistake:** `plt.savefig()` AFTER `plt.show()` produces a blank file — the buffer is already cleared.

---

### Quick Comparison Table — Visualization

| Feature              | Matplotlib                              | Seaborn                                    |
|----------------------|-----------------------------------------|--------------------------------------------|
| Abstraction Level    | Low-level, granular control             | High-level, automated statistics           |
| Primary Use Cases    | Custom plots, line/scatter/bar          | EDA, heatmaps, pairplots, boxplots         |
| Ideal Input          | Arrays, Python lists                    | Pandas DataFrames, arrays                  |
| Syntax               | `ax.plot(x, y)`                         | `sns.scatterplot(data=df, x=..., y=...)`   |

---

### Gotchas — Visualization

```python
# GOTCHA 1: Forgetting plt.show() in scripts → only prints object reference
# Always call plt.show() at the end in non-Jupyter environments

# GOTCHA 2: plt.subplots(2, 2) returns a 2D NumPy array of axes, not a list
fig, axes = plt.subplots(2, 2)
axes[0, 1].plot(x, y)    # Correct — 2D indexing
axes.flatten()[1].plot()  # Alternative — flatten to 1D first

# GOTCHA 3: Seaborn without ax= hijacks the entire figure
sns.heatmap(data)         # No ax= → destroys existing subplot grid
sns.heatmap(data, ax=axes[1])  # Correct — bind to specific axis
```

---

### Cheat Sheet — Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(x, y)                        # Matplotlib on left
sns.heatmap(matrix, ax=ax[1])           # Seaborn on right — bind with ax=

ax[0].set(title="Trend", xlabel="Epoch")
plt.tight_layout()
plt.savefig("output.png", dpi=300)      # Save BEFORE show
plt.show()
```

---

## 9. NumPy Basics

**Subtopics:** Array Creation, dtype, shape, reshape · Indexing, Slicing, View vs Copy · Broadcasting · Vectorized Operations

---

### Array Creation, `dtype`, `shape`, `reshape`

**Concept:** NumPy `ndarray` = homogeneous, contiguous C memory blocks. Far faster than Python lists. `dtype` enforces type uniformity. `shape` reveals dimensional layout. `reshape` changes layout without copying data.

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
print(arr.shape)    # (6,)
print(arr.dtype)    # float32

matrix = arr.reshape(2, 3)
# [[1. 2. 3.]
#  [4. 5. 6.]]

# Common reshape pattern for Scikit-Learn
col_vector = arr.reshape(-1, 1)   # -1 infers the row count automatically
```

**AI/ML Use:** Images as 3D tensors `(H × W × C)`. `reshape(-1, 1)` is standard for converting 1D feature arrays into 2D column vectors for regression models.

**Common Mistake:** `len(arr)` returns only the outermost dimension count. Use `arr.shape` to inspect the full dimensional structure.

---

### Indexing, Slicing, and View vs Copy

**Concept:** Basic slicing returns a **view** — a window into the same memory. Modifying a view **corrupts the original array**. Use `.copy()` to decouple.

```python
import numpy as np

base = np.array([10, 20, 30, 40])

view = base[1:3]       # VIEW — shares memory with base
view[0] = 999
print(base)            # [10 999 30 40] — base is corrupted!

safe = base[1:3].copy()  # COPY — independent memory, base is safe
safe[0] = 0
print(base)            # [10 999 30 40] — base unchanged
```

**AI/ML Use:** Views enable zero-copy ROI extraction from large image tensors or mini-batch subsetting from NLP datasets without RAM duplication.

**Common Mistake:** Mutating a slice during exploratory analysis corrupts ground-truth data. Always `.copy()` before modifying.

---

### Broadcasting

**Concept:** NumPy automatically "stretches" smaller arrays along missing/size-1 dimensions to match a larger array's shape — enabling element-wise operations without explicit loops.

```python
import numpy as np

matrix = np.array([[10, 20],
                   [30, 40]])   # Shape (2, 2)
vector = np.array([1, 2])       # Shape (2,)

result = matrix + vector        # vector broadcast as [[1,2],[1,2]]
# [[11 22]
#  [31 42]]
```

**Broadcasting Rules:** Shapes are compared right-to-left. Dimensions must either match or one of them must be 1.

**AI/ML Use:** Batch normalization, applying scalar learning rates, adding 1D bias vectors to multi-dimensional weight matrices.

**Common Mistake — Silent Shape Bug:**
```python
a = np.ones((10,))      # Shape (10,)
b = np.ones((10, 1))    # Shape (10, 1)
result = a + b          # Shape becomes (10, 10) — NOT (10,)!
# No error, silent corruption. Fix: a.reshape(-1, 1) + b
```

---

### Vectorized Operations & Functions

**Concept:** NumPy pushes computations into pre-compiled C code, bypassing the slow Python interpreter. Use NumPy functions — never Python built-ins — on arrays.

```python
import numpy as np

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print(x * y)           # [4 10 18]  — element-wise multiplication
print(np.dot(x, y))    # 32         — dot product (1*4 + 2*5 + 3*6)

matrix = np.array([[1, 2], [3, 4]])
np.sum(matrix, axis=0)   # [4 6]    — sum down rows (vertical)
np.mean(matrix, axis=1)  # [1.5 3.5] — mean across columns (horizontal)
np.max(matrix)           # 4        — global max
np.std(matrix)           # standard deviation
```

**Axis Reference:** `axis=0` = collapse rows (vertical). `axis=1` = collapse columns (horizontal).

**AI/ML Use:** `np.dot` = forward propagation engine. `mean` + `std` = normalization/standardization of input data.

**Common Mistake:** Using Python's `math.sqrt()` or `sum()` on NumPy arrays forces unpacking to Python objects → destroys performance. Always use `np.sqrt()`, `np.sum()`, etc.

---

### Quick Comparison Table — Python Lists vs NumPy Arrays

| Dimension            | Python Lists                        | NumPy Arrays                          |
|----------------------|-------------------------------------|---------------------------------------|
| Memory               | Scattered references, fragmented    | Contiguous C memory blocks            |
| Data Types           | Heterogeneous (mixed types OK)      | Homogeneous (one type enforced)       |
| Sizing               | Dynamically resizable               | Fixed size at creation                |
| Computation          | Slow Python loops required          | Fast compiled vectorization           |
| Slicing              | Returns independent copy            | Returns view (shared memory!)         |

---

### Gotchas — NumPy

```python
# GOTCHA 1: axis direction is counterintuitive
matrix = np.array([[1, 2], [3, 4]])
np.sum(matrix, axis=0)   # [4, 6]  — collapses ROWS (vertical sum)
np.sum(matrix, axis=1)   # [3, 7]  — collapses COLUMNS (horizontal sum)

# GOTCHA 2: + behaves differently on lists vs arrays
[1, 2] + [4, 5]               # [1, 2, 4, 5] — concatenation
np.array([1, 2]) + np.array([4, 5])  # [5, 7] — element-wise addition

# GOTCHA 3: Fancy Indexing always creates a COPY (not a view)
arr = np.array([10, 20, 30, 40])
view = arr[1:3]         # Slice → VIEW (shared memory)
copy = arr[[1, 3]]      # Fancy Index → COPY (independent memory)
```

---

### Cheat Sheet — NumPy

```python
import numpy as np

arr = np.array([[1, 2], [3, 4]], dtype=np.float32)

arr.shape              # (2, 2)
arr.reshape(-1, 1)     # Infer rows, force 1 column
arr[0, :]              # Row 0, all columns
arr[:, 1]              # All rows, column 1
arr[1:3].copy()        # Safe copy — decoupled from source

np.sum(arr, axis=0)    # [4, 6]  — vertical (collapse rows)
np.mean(arr, axis=1)   # [1.5, 3.5] — horizontal (collapse cols)
np.dot(A, B)           # Matrix multiplication

# Broadcasting rule: rightmost dimensions must match OR be 1
```

---

*End of Reference — 9 Topics Covered*
