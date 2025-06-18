# 🛠️ Language Processing – Pascal Compiler Project

## Grade: 18/20 ⭐️
📚 This document presents information related to the Practical Assignment for the Language Processing course, part of the 2nd semester of the 3rd year of the Bachelor's degree in Computer Engineering at the University of Minho, during the 2024/2025 academic year.

This repository contains the practical assignment developed for the Language Processing course, focused on building a compiler for the Pascal Standard language. The compiler supports variable declarations, arithmetic expression evaluation, control flow commands (if, while, for), and optionally, the implementation of subprograms (procedures and functions).

The main objective was to implement a complete compilation pipeline, composed of the following stages:

- Lexical Analysis

- Syntactic Analysis

- Semantic Analysis

- Code Generation targeting a virtual machine provided to students

This report describes in detail the functionalities implemented by the group, the technical decisions made, challenges encountered, and solutions adopted. Wherever relevant, code examples, diagrams, and test cases are presented to illustrate the operation and validation of the developed compiler.

# Setup

Start by cloning this repository, and creating a Python virtual environment:

```
$ python -m venv .venv
```

To run the project, start by running:

```
$ source .venv/bin/activate
$ pip install .
```

To compile pascal code to machine code:

```
$ python src/parser.py -c <test_path> 
```

To visualize parsed code tree:

```
$ python src/parser.py -v <test_path> 
```


To exit the virtual environment, you can run:

```
$ deactivate
```

## Authors
- A104276 - Afonso Dionísio
- A104356 - João Lobo
- A104439 - Rita Camacho
