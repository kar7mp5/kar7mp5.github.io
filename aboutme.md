---
layout: default
title: About Me
---

# About Me

## Education

- **Inha University**, Computer Science Engineering (2023.03 ~ Present)

## Experiences

- **Google Machine Learning Bootcamp, 5th** (2024.07-11)
- **Undergraduate research student in nursing department** (2024.10-12)
- **Undergraduate research student in aerospace engineering** (2025.01~Present)

## OpenSource

### Python Libraries

- **Korean News Scraper** - 2024.03 - [Project Link](https://github.com/kar7mp5/korean-news-scraper)

  Developed and deployed a Python library for collecting data to train Large Language Models (LLM).
  This was my first Python library, and while there were many areas to improve, it was a great experience learning about Python deployment and automation.

- **Notion News Crawler** - 2024.07 - [Project Link](https://github.com/kar7mp5/notion-news-crawler), [Blog](https://kar7mp5.tistory.com/)

  Developed a crawler for collecting news by category from Notion.
  During development, access to Notion's database was challenging, leading to the creation of a Python library for better integration.
  I set up a server running on a Raspberry Pi that works every 4 hours to upload relevant news to Notion.

## Projects

### Genetic Algorithm-Based Autonomous Drone Simulation

**`Jan 2024 – Feb 2024`** \| **[GitHub Repository](https://github.com/kar7mp5/Drone_Simulation)**

In this project, I developed a 2D drone simulator from scratch using Pygame, designed to navigate a drone to a target destination while avoiding obstacles. To discover the optimal flight path, a **Genetic Algorithm** was implemented to solve the control problem.

#### Key Implementations

- **Custom Physics Engine**: Instead of simple coordinate translation, I built a physics engine where the drone's movement is realistically simulated by calculating acceleration and velocity based on the thrust from its four individual rotors.
- **Pygame Environment**: Developed the entire simulation environment—including the drone, obstacles, and target visuals—from scratch using the Pygame library.
- **Genetic Algorithm**:
  - **Genes**: A "gene" was defined as a sequence of force commands applied to each of the drone's rotors.
  - **Fitness Function**: The fitness function was designed to award a higher score for closer proximity to the target, guiding the population toward the goal over successive generations.
  - **Evolution**: More efficient flight paths were generated with each new generation by applying selection, crossover, and mutation operators.

#### Core Technologies

- `Python`, `Pygame`, `NumPy`

### MinGPT: An LLM Implementation Based on 'Attention Is All You Need'

**`Mar 2024 – Jun 2024`** \| **[GitHub Repository](https://github.com/kar7mp5/MinGPT)**

This project is a from-scratch implementation of a GPT model based on the "Attention Is All You Need" paper. The primary goal was to develop a deep, functional understanding of the Transformer architecture by translating its fundamental components into code.

#### Key Implementations

- Implemented the core mechanisms of the Transformer, including **Self-Attention** and **Multi-Head Attention**.
- Applied **Positional Encoding** to enable the model to understand the sequence and order of tokens.
- Designed the complete **Transformer Block** structure, combining the attention layers with feed-forward networks.
- Built the model's core logic **from scratch** using fundamental PyTorch operations instead of high-level APIs like `nn.Transformer` to solidify foundational knowledge.

#### Core Technologies

- `Python`, `PyTorch`

### SurvivalRL: Reinforcement Learning

`Mar 2025 – Apr 2025` \| [**GitHub Repository**](https://github.com/kar7mp5/SurvivalRL) \| [**Tistory Blog**](https://kar7mp5.tistory.com/entry/Reinforcement-Learning-Python-Matplotlib%EC%9C%BC%EB%A1%9C-%EC%83%9D%ED%83%9C%EA%B3%84-%EA%B0%95%ED%99%94%ED%95%99%EC%8A%B5-%EC%8B%9C%EB%AE%AC%EB%A0%88%EC%9D%B4%EC%85%98-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0)

This is a project where an AI agent learns to survive on its own in a 3D survival shooting game. Using the PPO (Proximal Policy Optimization) algorithm, the agent discovers the optimal strategy for shooting enemies and dodging bullets.

#### Objective

To develop an AI that can survive by eliminating enemies in a 3D environment.

#### Core Technologies

- `ML-Agents`, `PyTorch`, `PPO Algorithm`

## Awards

- **Minister of Science and ICT Award, Korea Code Fair Hackathon** (2022.12)
- **TOP 6 in Data Analysis, Big Data & AI Competition with AWS, KT AICE** (2023.04~07)
- **3rd Place, Silver Prize, Inha University Capstone Design Competition** (2023.05~10)
- **Team Excellence Award, Inha University Carbon Neutral Academy 2nd Cohort** (2024.06)
- **Individual Excellence Award, Inha University Carbon Neutral Academy 2nd Cohort** (2024.06)
- **1st Place, Grand Prize, Inha University Capstone Design Competition** (2024.05~10)
- **Kaggle**: Top 3.7% (81st out of 2,234) in Binary Classification of Insurance Cross Selling (2024.07)
