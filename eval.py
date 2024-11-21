import pandas as pd
from sympy import symbols, simplify
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
    function_exponentiation, convert_xor
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from Levenshtein import distance
import matplotlib.pyplot as plt


# === Symbolic Expression Functions ===

def clean_expression(expr_str):
    """
    Clean the expression string to make it suitable for parsing.
    """
    if isinstance(expr_str, str):
        expr_str = expr_str.strip().replace('×', '*').replace('÷', '/').replace(' ', '')
        return expr_str
    return str(expr_str)


def are_expressions_equal(expr1_str, expr2_str):
    """
    Compare two symbolic expressions for equality.
    Returns True if equal, False otherwise.
    """
    try:
        w, x, y, z = symbols('w x y z')
        expr1_str = clean_expression(expr1_str)
        expr2_str = clean_expression(expr2_str)
        transformations = (
            standard_transformations +
            (implicit_multiplication_application, function_exponentiation, convert_xor)
        )
        expr1 = parse_expr(expr1_str, transformations=transformations)
        expr2 = parse_expr(expr2_str, transformations=transformations)
        return simplify(expr1 - expr2) == 0
    except Exception as e:
        print(f"Error comparing expressions: {expr1_str} vs {expr2_str}: {str(e)}")
        return False


def evaluate_symbolic_accuracy(csv_path):
    """
    Evaluate symbolic expression accuracy from a CSV file.
    """
    try:
        df = pd.read_csv(csv_path)
        results = {"total": 0, "correct": 0}

        for _, row in df.iterrows():
            ground_truth = row['Ground Truth']
            response = row['Final Response']
            results["total"] += 1
            if are_expressions_equal(ground_truth, response):
                results["correct"] += 1

        accuracy = (results["correct"] / results["total"]) * 100 if results["total"] > 0 else 0
        print(f"Symbolic Expression Accuracy: {accuracy:.2f}%")
        return results
    except Exception as e:
        print(f"Error evaluating symbolic accuracy: {str(e)}")
        return None


# === BLEU Score Functions ===

def calculate_bleu_scores(df):
    """
    Calculate BLEU scores for reference and candidate pairs in a DataFrame.
    """
    smooth = SmoothingFunction()
    scores = []
    for _, row in df.iterrows():
        reference = str(row['Ground Truth']).strip()
        candidate = str(row['Final Response']).strip()
        score = sentence_bleu([reference], candidate, smoothing_function=smooth.method1)
        scores.append(score)
    return scores


def plot_bleu_histogram(df, title="BLEU Score Distribution", color='blue'):
    """
    Plot a histogram of BLEU scores.
    """
    scores = calculate_bleu_scores(df)
    plt.hist(scores, bins=20, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel('BLEU Score')
    plt.ylabel('Frequency')
    plt.show()


# === Levenshtein Similarity Functions ===

def calculate_levenshtein_similarity(df):
    """
    Calculate average Levenshtein similarity for reference and candidate pairs.
    """
    similarities = []
    for _, row in df.iterrows():
        ref = str(row['Ground Truth']).strip()
        hyp = str(row['Final Response']).strip()
        max_len = max(len(ref), len(hyp))
        similarity = (max_len - distance(ref, hyp)) / max_len if max_len > 0 else 1.0
        similarities.append(similarity * 100)
    return sum(similarities) / len(similarities)


def plot_levenshtein_histogram(df, title="Levenshtein Similarity Distribution", color='blue'):
    """
    Plot a histogram of Levenshtein similarities.
    """
    similarities = calculate_levenshtein_similarity(df)
    plt.hist(similarities, bins=20, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel('Similarity (%)')
    plt.ylabel('Frequency')
    plt.show()


# === Main Function ===

if __name__ == "__main__":
    # Example CSV file paths (replace with your file paths)
    csv_path = "output.csv"

    # Evaluate symbolic accuracy
    print("Evaluating symbolic accuracy...")
    evaluate_symbolic_accuracy(csv_path)

    # Calculate and plot BLEU scores
    print("Calculating BLEU scores...")
    df = pd.read_csv(csv_path)
    plot_bleu_histogram(df)

    # Calculate and plot Levenshtein similarities
    print("Calculating Levenshtein similarities...")
    plot_levenshtein_histogram(df)
