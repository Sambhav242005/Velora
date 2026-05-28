from pathlib import Path

TEXT = """
Cloud computing is a way to use computing resources through the internet. It gives users access to servers, storage, databases, software, and networking without buying all hardware directly.

A private cloud is used by one organization. It gives more control, security, and customization. Private cloud deployment needs virtualization, storage management, networking, identity management, monitoring, backup, and automation.

Infrastructure as a Service gives virtual machines and storage. Platform as a Service gives a development platform. Software as a Service gives ready-to-use applications. These service models help companies choose how much control they want.

A queue is a linear data structure that follows first in first out. The element inserted first is removed first. A queue is used in CPU scheduling, printer queues, network packet handling, and customer service systems.

A stack is a linear data structure that follows last in first out. The element inserted last is removed first. Stack is used in function calls, undo operations, expression evaluation, and browser history.

Machine learning is a field where computers learn patterns from data. A dependent variable is the output we want to predict. Independent variables are inputs used to make predictions. In linear regression the model can be written as y = mx + b.

A RESTful API uses HTTP methods such as GET, POST, PUT, PATCH, and DELETE. Status code 200 means success. Status code 201 means created. Status code 404 means not found. Status code 500 means internal server error.

Training a small language model requires a tokenizer, a clean dataset, a model architecture, an optimizer, a learning rate schedule, checkpointing, and evaluation. A safe trainer should save progress on interrupt, crash, and out of memory errors.
""".strip()

out = Path("data/raw/sample.txt")
out.parent.mkdir(parents=True, exist_ok=True)
# Repeat sample text so the local smoke test has enough tokens.
out.write_text((TEXT + "\n\n") * 2000, encoding="utf-8")
print(f"Wrote {out}")
