import random
from typing import List, Tuple, Dict

# Labeled query dataset for training and testing the hybrid router:
# Label 0: Simple Factual -> On-device execution (sub-100ms)
# Label 1: Moderate Reasoning -> On-device execution with self-verification/retry
# Label 2: Complex Multi-step -> Direct to cloud execution (Claude/Gemini)
RAW_QUERIES: List[Tuple[str, int]] = [
    # Simple Factual (0)
    ("What is the capital of France?", 0),
    ("Who wrote Romeo and Juliet?", 0),
    ("What is the boiling point of water in Celsius?", 0),
    ("How many continents are there?", 0),
    ("What is the chemical symbol for gold?", 0),
    ("Name the largest ocean on Earth.", 0),
    ("When did World War II end?", 0),
    ("What is the speed of light?", 0),
    ("Who painted the Mona Lisa?", 0),
    ("What is the square root of 144?", 0),
    ("How many days are in a leap year?", 0),
    ("What is the capital of Japan?", 0),
    ("What is the currency of the UK?", 0),
    ("How many planets are in our solar system?", 0),
    ("What is the height of Mount Everest?", 0),
    ("Who discovered gravity?", 0),
    ("What is the largest mammal?", 0),
    ("What is the primary language spoken in Brazil?", 0),
    ("List the colors of the rainbow.", 0),
    ("Who is the president of the US?", 0),
    
    # Moderate Reasoning (1)
    ("Compare the weather in New York and Los Angeles during winter.", 1),
    ("Can you explain how photosynthesis works in simple terms?", 1),
    ("Summarize this text: The quick brown fox jumps over the lazy dog.", 1),
    ("Draft a polite email asking for feedback on my project.", 1),
    ("How does a diesel engine differ from a petrol engine?", 1),
    ("If a store offers a 20% discount on a $50 shirt, what is the final price?", 1),
    ("What are the main differences between a cat and a dog as pets?", 1),
    ("Rewrite this sentence in a professional tone: i can't make it to the meeting.", 1),
    ("Explain the concept of inflation to a 10 year old.", 1),
    ("Is it better to exercise in the morning or evening?", 1),
    ("Compare vegetarian and vegan diets.", 1),
    ("What are three ways to save water at home?", 1),
    ("How do clouds form?", 1),
    ("Explain the difference between a solid, liquid, and gas.", 1),
    ("Write a 3-sentence description of a futuristic city.", 1),
    
    # Complex Multi-step / Hard Reasoning (2)
    ("Write a Python script that scrapes headlines from a news website and sends an email digest.", 2),
    ("A farmer has a rectangular field of length 200m and width 100m. He wants to divide it into three equal zones for wheat, corn, and barley, but wheat needs 20% more water. Calculate the dimensions and optimal irrigation pipe layout.", 2),
    ("Analyze the macroeconomic impacts of raising interest rates in a developing country with high debt.", 2),
    ("Design an system architecture for a real-time chat application with group messaging and file attachments, specifying DB choices and protocol selections.", 2),
    ("Debug this C++ code block: there's a segmentation fault in my linked list traversal when removing nodes.", 2),
    ("Prove that the square root of 2 is irrational.", 2),
    ("Write a detailed essay analyzing the symbolism of the green light in F. Scott Fitzgerald's The Great Gatsby.", 2),
    ("Develop a detailed business model canvas and 5-year financial projection for a B2B SaaS startup in the logistics space.", 2),
    ("How would you optimize a SQL query joining 5 tables with over 10 million rows? Explain indexing and query execution plans.", 2),
    ("Create a step-by-step training schedule for a marathon, including dietary recommendations, heart-rate zones, and active recovery weeks.", 2),
    ("Explain the quantum mechanical principles behind semiconductors and how they enable modern transistor scaling.", 2),
    ("Design a microservices-based deployment configuration in Kubernetes including Istio virtual service routing and rate limiters.", 2)
]

def load_router_dataset(ratio: float = 0.8) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Loads and splits the query complexity dataset into train and test sets.
    Automatically duplicates/synthesizes queries to expand dataset size for training.
    """
    # Expand dataset to represent a larger sample size (e.g. 300 queries)
    expanded_queries = []
    
    # Templates to expand the dataset
    simple_templates = ["What is the capital of {}", "Tell me about {}", "Define the word {}", "How many {} in a {}"]
    moderate_templates = ["Compare {} and {}", "Explain how {} works", "Write a summary of {}", "How do I {} at home"]
    complex_templates = ["Write a Python program to {}", "Design a system architecture for {}", "Analyze the impact of {} on {}", "Debug this code: {}"]
    
    topics = [
        ("Berlin", "Germany", "Europe"),
        ("Tokyo", "Japan", "Asia"),
        ("gravity", "relativity", "physics"),
        ("photosynthesis", "cellular respiration", "biology"),
        ("interest rates", "inflation", "macroeconomics"),
        ("SQL index", "NoSQL database", "system design"),
        ("C++ pointer", "Python list", "software bugs"),
        ("a car", "a bicycle", "transportation")
    ]
    
    # Build a base set
    for query, label in RAW_QUERIES:
        expanded_queries.append((query, label))
        
    # Synthesize extra queries
    for topic_tuple in topics:
        expanded_queries.append((simple_templates[0].format(topic_tuple[0]), 0))
        expanded_queries.append((simple_templates[1].format(topic_tuple[1]), 0))
        expanded_queries.append((simple_templates[2].format(topic_tuple[2]), 0))
        
        expanded_queries.append((moderate_templates[0].format(topic_tuple[0], topic_tuple[1]), 1))
        expanded_queries.append((moderate_templates[1].format(topic_tuple[2]), 1))
        expanded_queries.append((moderate_templates[2].format(topic_tuple[1]), 1))
        
        expanded_queries.append((complex_templates[0].format("calculate Fibonacci series using dynamic programming"), 2))
        expanded_queries.append((complex_templates[1].format("an e-commerce shopping cart microservice"), 2))
        expanded_queries.append((complex_templates[2].format("decentralized finance protocols", "traditional banking"), 2))
        expanded_queries.append((complex_templates[3].format("null pointer dereference in C assembly"), 2))

    # Shuffle
    random.seed(42)
    random.shuffle(expanded_queries)
    
    split_idx = int(len(expanded_queries) * ratio)
    
    train_queries = expanded_queries[:split_idx]
    test_queries = expanded_queries[split_idx:]
    
    train_x = [q[0] for q in train_queries]
    train_y = [q[1] for q in train_queries]
    test_x = [q[0] for q in test_queries]
    test_y = [q[1] for q in test_queries]
    
    return train_x, train_y, test_x, test_y
