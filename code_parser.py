from radon.raw import analyze
from radon.metrics import h_visit
from radon.complexity import cc_visit
import ast

def extract_metrics(code_string):
    """
    Parses raw Python code and calculates the 21 metrics required by the JM1 model.
    """
    try:
        # 1. Raw Metrics
        raw_metrics = analyze(code_string)
        
        # 2. Halstead Metrics
        # Get the master object, then extract the '.total' HalsteadReport
        halstead_master = h_visit(code_string)
        halstead = halstead_master.total 
        
        # 3. Cyclomatic Complexity
        blocks = cc_visit(code_string)
        total_cc = sum([block.complexity for block in blocks]) if blocks else 1

        # 4. AST Parsing for Branch count approximation
        tree = ast.parse(code_string)
        branch_count = sum(isinstance(node, (ast.If, ast.While, ast.For, ast.Try)) for node in ast.walk(tree))

        # 5. Map to the exact 21 features the JM1 model expects
        metrics_dict = {
            'Lines of Code (LOC)': raw_metrics.loc,
            'Cyclomatic Complexity (v(g))': total_cc,
            'Essential Complexity (ev(g))': max(1, total_cc - 2), 
            'Design Complexity (iv(g))': max(1, total_cc - 1),    
            'Halstead Length (n)': halstead.vocabulary,        # RESTORED to .vocabulary
            'Halstead Volume (v)': halstead.volume,
            'Halstead Program Length (l)': halstead.length,    # RESTORED to .length
            'Halstead Difficulty (d)': halstead.difficulty,
            'Halstead Intelligence (i)': halstead.volume / max(halstead.difficulty, 1), 
            'Halstead Effort (e)': halstead.effort,
            'Halstead Delivered Bugs (b)': halstead.bugs,
            'Halstead Time Estimator (t)': halstead.time,
            'Lines of Executable Code': raw_metrics.sloc,
            'Lines of Comments': raw_metrics.comments,
            'Lines of Blank Space': raw_metrics.blank,
            'Lines of Code & Comments': 0, 
            'Unique Operators': halstead.h1,
            'Unique Operands': halstead.h2,
            'Total Operators': halstead.N1,
            'Total Operands': halstead.N2,
            'Branch Count': branch_count
        }
        
        return metrics_dict
        
    except Exception as e:
        return {"error": str(e)}

# --- Quick Test ---
if __name__ == "__main__":
    sample_code = """
def calculate_discount(price, is_member):
    if is_member:
        return price * 0.9
    else:
        return price
    """
    print(extract_metrics(sample_code))